"""Extract frame-level WavLM (1024-d) and Fbank (40-d, 25/10 ms) features.

    python scripts/extract_features.py --manifest manifests/iemocap.csv \
        --out features/iemocap --device cuda

Writes features/<dataset>/{wavlm,fbank}/<utt_id>.npy (float16). The two streams keep
their own lengths; nothing is aligned or resampled across streams.
"""
import argparse
import os

import _path  # noqa: F401
from tqdm import tqdm

from afea.data import read_manifest
from afea.features import WavLMExtractor, compute_fbank, load_audio, save_feature


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--streams", nargs="+", default=["wavlm", "fbank"], choices=["wavlm", "wavlm_all", "fbank"],
                    help="wavlm_all = every hidden layer, [L, M, 1024] (~64 GB for IEMOCAP)")
    ap.add_argument("--wavlm_checkpoint", default="microsoft/wavlm-large")
    ap.add_argument("--wavlm_layer", type=int, default=None,
                    help="hidden_states index; default = final layer (last_hidden_state)")
    ap.add_argument("--no_cmvn", action="store_true", help="disable per-utterance Fbank CMVN")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    for s in args.streams:
        os.makedirs(os.path.join(args.out, s), exist_ok=True)
    need_wavlm = any(s.startswith("wavlm") for s in args.streams)
    wavlm = WavLMExtractor(args.wavlm_checkpoint, args.wavlm_layer, args.device) if need_wavlm else None

    for row in tqdm(rows):
        targets = {s: os.path.join(args.out, s, row["utt_id"] + ".npy") for s in args.streams}
        if not args.overwrite and all(os.path.exists(p) for p in targets.values()):
            continue
        wav = load_audio(row["path"])
        if "fbank" in targets:
            save_feature(targets["fbank"], compute_fbank(wav, cmvn=not args.no_cmvn), fp16=not args.fp32)
        if "wavlm" in targets:
            save_feature(targets["wavlm"], wavlm(wav), fp16=not args.fp32)
        if "wavlm_all" in targets:
            save_feature(targets["wavlm_all"], wavlm.all_layers(wav), fp16=not args.fp32)


if __name__ == "__main__":
    main()
