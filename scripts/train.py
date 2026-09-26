"""Cross-validated training of AFEA-Net and the paper's ablations.

    # full model (Table 4, last row)
    python scripts/train.py --dataset iemocap --manifest manifests/iemocap.csv \
        --feat_root features/iemocap --out runs/iemocap_afea

    # ablations
    --model wavlm | --model fbank          single-stream BiLSTM + max pooling
    --no_align                             w/o L_ali
    --afea_layers 0                        w/o AFEA (concatenate + FCN)
    --afea_layers {1,2,4} --no_con         AFEA-l
    --no_con                               w/o L_con
"""
import argparse
import os

import _path  # noqa: F401
import torch

from afea.config import DATASET_PRESETS, TRAIN_DEFAULTS
from afea.data import read_manifest
from afea.trainer import run_cv


def parse_args():
    d = TRAIN_DEFAULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASET_PRESETS), required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feat_root", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", choices=["afea", "wavlm", "fbank"], default="afea")
    ap.add_argument("--afea_layers", type=int, default=d["afea_layers"])
    ap.add_argument("--no_align", action="store_true")
    ap.add_argument("--no_con", action="store_true")
    ap.add_argument("--margin", type=float, default=None, help="default: dataset preset")
    ap.add_argument("--con_weights", type=float, nargs="+", default=None,
                    help="one weight per AFEA layer; default: dataset preset (alpha beta gamma)")
    ap.add_argument("--epochs", type=int, default=d["epochs"])
    ap.add_argument("--batch_size", type=int, default=d["batch_size"])
    ap.add_argument("--lr", type=float, default=d["lr"])
    ap.add_argument("--weight_decay", type=float, default=d["weight_decay"])
    ap.add_argument("--lstm_hidden", type=int, default=d["lstm_hidden"],
                    help="per-direction hidden size; D = 2 * lstm_hidden")
    ap.add_argument("--lstm_dropout", type=float, default=d["lstm_dropout"])
    ap.add_argument("--isa_hidden", type=int, default=d["isa_hidden"])
    ap.add_argument("--fcn_hidden", type=int, default=d["fcn_hidden"])
    ap.add_argument("--fcn_dropout", type=float, default=d["fcn_dropout"])
    ap.add_argument("--val_ratio", type=float, default=d["val_ratio"])
    ap.add_argument("--select", choices=["val", "last"], default=d["select"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[d["seed"]])
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--max_wavlm_frames", type=int, default=None)
    ap.add_argument("--max_fbank_frames", type=int, default=None)
    ap.add_argument("--cache", action="store_true", help="keep loaded features in RAM")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--wandb", default=None, metavar="PROJECT",
                    help="log to Weights & Biases under this project (off by default)")
    ap.add_argument("--wandb_name", default=None, help="run name (default: basename of --out)")
    ap.add_argument("--wandb_group", default=None, help="e.g. iemocap_ablations")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main():
    args = parse_args()
    preset = DATASET_PRESETS[args.dataset]
    cfg = vars(args).copy()
    cfg["num_classes"] = preset["num_classes"]
    cfg["margin"] = args.margin if args.margin is not None else preset["margin"]
    cfg["use_align"] = not args.no_align
    cfg["use_con"] = not args.no_con and args.model == "afea" and args.afea_layers > 0
    weights = args.con_weights if args.con_weights is not None else preset["con_weights"]
    if cfg["use_con"] and len(weights) != args.afea_layers:
        raise SystemExit(f"--con_weights needs {args.afea_layers} values for {args.afea_layers} AFEA "
                         f"layers (preset has {len(weights)}); pass them or use --no_con")
    cfg["con_weights"] = list(weights)
    cfg["max_frames"] = {"wavlm": args.max_wavlm_frames, "fbank": args.max_fbank_frames}
    if args.select == "val" and args.val_ratio <= 0:
        raise SystemExit("--select val requires --val_ratio > 0")
    if args.wandb and not args.wandb_name and args.out:
        cfg["wandb_name"] = os.path.basename(os.path.normpath(args.out))
    rows = read_manifest(args.manifest)
    run_cv(rows, cfg, folds=args.folds, seeds=args.seeds, device=args.device, out_dir=args.out)


if __name__ == "__main__":
    main()
