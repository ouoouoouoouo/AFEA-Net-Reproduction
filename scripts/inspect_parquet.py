"""Show the layout of a Parquet speech-emotion corpus before building a manifest.

    python scripts/inspect_parquet.py /home/ouo/dataset/thai-ser/data/
"""
import argparse

import _path  # noqa: F401

from afea.data.parquet_ser import inspect_parquet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="a .parquet file or a directory searched recursively")
    ap.add_argument("--rows", type=int, default=3)
    args = ap.parse_args()
    print(inspect_parquet(args.root, args.rows))


if __name__ == "__main__":
    main()
