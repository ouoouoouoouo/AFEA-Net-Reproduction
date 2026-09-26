"""RAVDESS (speech, audio-only) 8-class manifest builder.

File name format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
e.g. 03-01-06-01-02-01-12.wav. Speech = vocal channel 01, audio-only = modality 03.
1440 files, 24 actors. The paper does not say how the 5 folds are built; we use
actor-disjoint folds (assumption A-RAV-1 in ASSUMPTIONS.md).
"""

import glob
import os
from collections import Counter
from typing import Dict, List

# Class order follows Fig. 6 of the paper: 0-sad, 1-hap, 2-ang, 3-neu, 4-calm, 5-fea, 6-dis, 7-sur.
CLASSES = ["sad", "hap", "ang", "neu", "calm", "fea", "dis", "sur"]
CODE_TO_CLASS = {"01": "neu", "02": "calm", "03": "hap", "04": "sad",
                 "05": "ang", "06": "fea", "07": "dis", "08": "sur"}
EXPECTED_TOTAL = 1440
NUM_FOLDS = 5


def actor_fold(actor: int, num_folds: int = NUM_FOLDS) -> int:
    """Actors 1..24 -> folds 1..5 as contiguous blocks of 5 actors (sizes 5,5,5,5,4)."""
    return min((actor - 1) // 5, num_folds - 1) + 1


def build_ravdess_manifest(root: str, strict: bool = True) -> List[Dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)):
        parts = os.path.splitext(os.path.basename(path))[0].split("-")
        if len(parts) != 7 or parts[0] != "03" or parts[1] != "01":
            continue
        label = CODE_TO_CLASS[parts[2]]
        actor = int(parts[6])
        rows.append({"utt_id": os.path.splitext(os.path.basename(path))[0], "path": path,
                     "label": label, "label_id": CLASSES.index(label),
                     "fold": actor_fold(actor), "speaker": f"Actor_{actor:02d}"})
    if len(rows) != EXPECTED_TOTAL:
        msg = f"RAVDESS speech count {len(rows)} != {EXPECTED_TOTAL}; counts {dict(Counter(r['label'] for r in rows))}"
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
    return rows
