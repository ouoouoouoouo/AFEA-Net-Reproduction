"""Tiny CSV manifest helpers shared by all datasets.

A manifest row describes one utterance:
    utt_id, path, label, label_id, fold, speaker
`fold` is the cross-validation group (IEMOCAP session, RAVDESS actor group).
"""

import csv
from typing import Dict, List

FIELDS = ["utt_id", "path", "label", "label_id", "fold", "speaker"]


def write_manifest(rows: List[Dict], out_path: str) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})


def read_manifest(path: str) -> List[Dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["label_id"] = int(row["label_id"])
        row["fold"] = int(row["fold"])
    return rows
