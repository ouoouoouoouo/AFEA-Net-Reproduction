"""Build a CSV manifest.

    python scripts/prepare_data.py --dataset iemocap --root /data/IEMOCAP_full_release \
        --out manifests/iemocap.csv

    # Parquet corpus used as a cross-corpus test set (labels mapped onto IEMOCAP's 4 classes)
    python scripts/prepare_data.py --dataset thaiser --root /home/ouo/dataset/thai-ser/data \
        --out manifests/thaiser.csv --wav_dir data/thaiser_wav
"""
import argparse
from collections import Counter

import _path  # noqa: F401

from afea.data import write_manifest
from afea.data.iemocap import build_iemocap_manifest
from afea.data.parquet_ser import PRESETS, build_parquet_manifest, parse_label_map
from afea.data.ravdess import build_ravdess_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iemocap", "ravdess", "thaiser", "parquet"], required=True,
                    help="thaiser / parquet: any Parquet SER corpus, mapped onto IEMOCAP's 4 classes")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_strict", action="store_true", help="warn instead of failing on count mismatch")
    pq = ap.add_argument_group("Parquet corpora (see scripts/inspect_parquet.py)")
    pq.add_argument("--wav_dir", default=None, help="where decoded audio is written (default data/<dataset>_wav)")
    pq.add_argument("--audio_col", default=None,
                    help="one column, or several separated by commas (first non-empty wins); "
                         "thaiser default: mic_con,mic_clip,mic_middle,mic_zoom")
    pq.add_argument("--label_col", default=None, help="thaiser default: majority_emo (assigned_emo = acted intent)")
    pq.add_argument("--speaker_col", default=None)
    pq.add_argument("--id_col", default=None)
    pq.add_argument("--label_map", default=None,
                    help="e.g. 'neutral=neu,angry=ang,happy=hap,sad=sad' (integer labels: '0=neu,...'); "
                         "unlisted labels are dropped; default: common English emotion names")
    args = ap.parse_args()
    if args.dataset in ("thaiser", "parquet"):
        preset = PRESETS.get(args.dataset, {})   # explicit flags override the preset
        cols = {k: getattr(args, k) or preset.get(k) for k in ("audio_col", "label_col", "speaker_col", "id_col")}
        rows = build_parquet_manifest(args.root, args.wav_dir or f"data/{args.dataset}_wav", cols["audio_col"],
                                      cols["label_col"], cols["speaker_col"], cols["id_col"],
                                      parse_label_map(args.label_map), prefix=args.dataset + "_")
    else:
        build = build_iemocap_manifest if args.dataset == "iemocap" else build_ravdess_manifest
        rows = build(args.root, strict=not args.no_strict)
    write_manifest(rows, args.out)
    print(f"{len(rows)} utterances -> {args.out}")
    print("per class:", dict(Counter(r["label"] for r in rows)))
    print("per fold :", dict(sorted(Counter(r["fold"] for r in rows).items())))


if __name__ == "__main__":
    main()
