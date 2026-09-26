"""IEMOCAP 4-class manifest builder.

The paper (Sec. 4.1, Table 2) uses 5531 utterances from four classes:
happy (= happy + excited) 1636, angry 1103, sad 1084, neutral 1708.
We use the utterance-level majority label shipped in
`SessionX/dialog/EmoEvaluation/*.txt` and merge `exc` into `hap`.
Builds refuse to succeed unless the counts match the paper exactly
(pass `strict=False` to only warn).
"""

import glob
import os
import re
from collections import Counter
from typing import Dict, List

# Class order follows the paper's confusion matrix (Fig. 6a): 0-sad, 1-hap, 2-ang, 3-neu.
CLASSES = ["sad", "hap", "ang", "neu"]
RAW_TO_CLASS = {"sad": "sad", "hap": "hap", "exc": "hap", "ang": "ang", "neu": "neu"}
EXPECTED_COUNTS = {"sad": 1084, "hap": 1636, "ang": 1103, "neu": 1708}
EXPECTED_TOTAL = 5531

_LINE_RE = re.compile(r"^\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]\s+(Ses\S+)\s+(\S+)\s+\[")


def parse_emo_evaluation(txt_path: str) -> List[Dict]:
    """Return [{utt_id, raw_label, start, end}] for every turn in one EmoEvaluation file."""
    items = []
    with open(txt_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = _LINE_RE.match(line.strip())
            if m:
                start, end, utt_id, raw = m.groups()
                items.append({"utt_id": utt_id, "raw_label": raw,
                              "start": float(start), "end": float(end)})
    return items


def build_iemocap_manifest(root: str, strict: bool = True) -> List[Dict]:
    """Scan an `IEMOCAP_full_release` directory and return 4-class manifest rows."""
    rows = []
    for session in range(1, 6):
        sess_dir = os.path.join(root, f"Session{session}")
        eval_files = sorted(glob.glob(os.path.join(sess_dir, "dialog", "EmoEvaluation", "*.txt")))
        if not eval_files:
            raise FileNotFoundError(f"No EmoEvaluation files under {sess_dir}")
        for txt in eval_files:
            if os.path.basename(txt).startswith("."):  # macOS resource forks
                continue
            for item in parse_emo_evaluation(txt):
                label = RAW_TO_CLASS.get(item["raw_label"])
                if label is None:
                    continue
                utt_id = item["utt_id"]
                dialog = utt_id.rsplit("_", 1)[0]
                wav = os.path.join(sess_dir, "sentences", "wav", dialog, utt_id + ".wav")
                # speaker = session + gender of the *speaking* actor (last id segment, e.g. F000)
                speaker = f"Ses{session:02d}{utt_id.rsplit('_', 1)[1][0]}"
                rows.append({"utt_id": utt_id, "path": wav, "label": label,
                             "label_id": CLASSES.index(label), "fold": session,
                             "speaker": speaker})
    check_counts(rows, strict=strict)
    return rows


def check_counts(rows: List[Dict], strict: bool = True) -> Counter:
    counts = Counter(r["label"] for r in rows)
    ok = len(rows) == EXPECTED_TOTAL and all(counts[c] == n for c, n in EXPECTED_COUNTS.items())
    if not ok:
        msg = (f"IEMOCAP 4-class counts {dict(counts)} (total {len(rows)}) do not match the paper "
               f"{EXPECTED_COUNTS} (total {EXPECTED_TOTAL}).")
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)
    return counts
