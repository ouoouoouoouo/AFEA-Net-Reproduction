"""Build a CSV manifest.

    python scripts/prepare_data.py --dataset iemocap --root /data/IEMOCAP_full_release \
        --out manifests/iemocap.csv
"""
import argparse
from collections import Counter

import _path  # noqa: F401

from afea.data import write_manifest
from afea.data.iemocap import build_iemocap_manifest
from afea.data.ravdess import build_ravdess_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["iemocap", "ravdess"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_strict", action="store_true", help="warn instead of failing on count mismatch")
    args = ap.parse_args()
    build = build_iemocap_manifest if args.dataset == "iemocap" else build_ravdess_manifest
    rows = build(args.root, strict=not args.no_strict)
    write_manifest(rows, args.out)
    print(f"{len(rows)} utterances -> {args.out}")
    print("per class:", dict(Counter(r["label"] for r in rows)))
    print("per fold :", dict(sorted(Counter(r["fold"] for r in rows).items())))


if __name__ == "__main__":
    main()
