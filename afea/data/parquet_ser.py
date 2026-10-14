"""Speech-emotion corpora stored as Parquet files (e.g. Hugging Face exports such as Thai SER).

The column layout differs between exports, so nothing is hard-coded:
  * `inspect_parquet(root)` prints the schema, a few rows and the label distribution;
  * `build_parquet_manifest(...)` decodes every audio cell to <wav_dir>/<utt_id>.wav, so the
    usual feature extractor can read it, and maps the corpus labels onto a target label set.

Used for cross-corpus testing: the model is trained on IEMOCAP, so labels are mapped onto
IEMOCAP's 4 classes and utterances whose emotion has no IEMOCAP counterpart are dropped
(assumptions X-1 ... X-4 in ASSUMPTIONS.md).
"""

import glob
import io
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

from .iemocap import CLASSES as IEMOCAP_CLASSES

# Lower-cased corpus label -> IEMOCAP 4-class label. Labels not listed here are dropped.
# "frustrated" and the like are dropped on purpose: IEMOCAP's 4-class setup excludes them.
DEFAULT_LABEL_MAP = {
    "neutral": "neu", "neu": "neu", "normal": "neu",
    "angry": "ang", "anger": "ang", "ang": "ang",
    "happy": "hap", "happiness": "hap", "hap": "hap", "joy": "hap", "excited": "hap", "exc": "hap",
    "sad": "sad", "sadness": "sad",
}
AUDIO_NAMES = ("audio", "wav", "speech", "waveform", "file", "path", "audio_path", "filepath")
LABEL_NAMES = ("emotion", "label", "labels", "emotion_label", "emo", "class", "category",
               "majority_emo", "assigned_emo", "emotion_name")
SPEAKER_NAMES = ("speaker", "speaker_id", "spk", "spk_id", "actor", "client_id")
ID_NAMES = ("id", "utt_id", "utterance_id", "name", "file_name", "filename", "key")


def parquet_files(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .parquet files under {root}")
    return files


def read_parquet(root: str):
    import pandas as pd
    return pd.concat([pd.read_parquet(f).assign(_file=os.path.basename(f)) for f in parquet_files(root)],
                     ignore_index=True)


def _is_audio_cell(v) -> bool:
    if isinstance(v, dict):
        return "bytes" in v or "array" in v
    return isinstance(v, (bytes, bytearray))


def detect_columns(df) -> Dict[str, Optional[str]]:
    cols = [c for c in df.columns if c != "_file"]
    lower = {c.lower(): c for c in cols}

    def by_name(names):
        return next((lower[n] for n in names if n in lower), None)

    first = df.iloc[0]
    audio = next((c for c in cols if _is_audio_cell(first[c])), None) or by_name(AUDIO_NAMES)
    return {"audio": audio, "label": by_name(LABEL_NAMES), "speaker": by_name(SPEAKER_NAMES),
            "id": by_name(ID_NAMES)}


def _short(v, n=80):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_short(x, 30)}" for k, x in v.items()) + "}"
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, np.ndarray):
        return f"<array {v.shape} {v.dtype}>"
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def inspect_parquet(root: str, n_rows: int = 3) -> str:
    files = parquet_files(root)
    df = read_parquet(root)
    out = [f"{len(files)} parquet file(s), {len(df)} rows", "files: " + ", ".join(os.path.basename(f) for f in files[:10]),
           "", "columns:"]
    for c in df.columns:
        if c == "_file":
            continue
        out.append(f"  {c:24s} {str(df[c].dtype):10s} e.g. {_short(df[c].iloc[0])}")
    out += ["", f"first {n_rows} rows:"]
    for i in range(min(n_rows, len(df))):
        out.append("  " + " | ".join(f"{c}={_short(df[c].iloc[i], 40)}" for c in df.columns if c != "_file"))
    out += ["", "low-cardinality columns (value counts):"]
    for c in df.columns:
        try:
            vc = df[c].value_counts()
        except TypeError:  # dict / bytes cells
            continue
        if 1 < len(vc) <= 30:
            out.append(f"  {c}: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    det = detect_columns(df)
    out += ["", "auto-detected: " + ", ".join(f"{k}={v}" for k, v in det.items())]
    return "\n".join(out)


def decode_audio(cell, base_dir: str = ""):
    """-> (float32 mono waveform, sample rate) from an HF Audio dict, raw bytes or a file path."""
    import soundfile as sf
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return decode_audio(cell["bytes"])
        if cell.get("array") is not None:
            return np.asarray(cell["array"], dtype=np.float32), int(cell["sampling_rate"])
        if cell.get("path"):
            return decode_audio(cell["path"], base_dir)
        raise ValueError("audio cell has neither bytes, array nor path")
    if isinstance(cell, (bytes, bytearray)):
        try:
            wav, sr = sf.read(io.BytesIO(cell), dtype="float32", always_2d=True)
        except Exception:  # e.g. mp3 with an old libsndfile
            import torchaudio
            t, sr = torchaudio.load(io.BytesIO(cell))
            wav = t.T.numpy()
        return wav.mean(axis=1).astype(np.float32), int(sr)
    if isinstance(cell, str):
        path = cell if os.path.isabs(cell) else os.path.join(base_dir, cell)
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        return wav.mean(axis=1), int(sr)
    raise TypeError(f"unsupported audio cell type {type(cell)}")


def parse_label_map(spec: Optional[str]) -> Dict[str, str]:
    """'neutral=neu,angry=ang' -> dict; keys are lower-cased (integer labels work too: '0=neu')."""
    if not spec:
        return dict(DEFAULT_LABEL_MAP)
    out = {}
    for item in spec.split(","):
        k, v = item.split("=")
        out[k.strip().lower()] = v.strip()
    return out


def _safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def build_parquet_manifest(root: str, wav_dir: str, audio_col: Optional[str] = None,
                           label_col: Optional[str] = None, speaker_col: Optional[str] = None,
                           id_col: Optional[str] = None, label_map: Optional[Dict[str, str]] = None,
                           classes: Sequence[str] = IEMOCAP_CLASSES, prefix: str = "",
                           overwrite: bool = False, log=print) -> List[Dict]:
    """Decode audio to <wav_dir>/<utt_id>.wav and return manifest rows (fold = 1: test-only corpus)."""
    import soundfile as sf
    df = read_parquet(root)
    det = detect_columns(df)
    audio_col, label_col = audio_col or det["audio"], label_col or det["label"]
    speaker_col, id_col = speaker_col or det["speaker"], id_col or det["id"]
    if audio_col is None or label_col is None:
        raise SystemExit(f"could not detect the audio / label columns (detected {det}); "
                         "pass --audio_col / --label_col (see scripts/inspect_parquet.py)")
    log(f"columns: audio={audio_col} label={label_col} speaker={speaker_col} id={id_col}")
    label_map = label_map if label_map is not None else dict(DEFAULT_LABEL_MAP)
    os.makedirs(wav_dir, exist_ok=True)
    base_dir = root if os.path.isdir(root) else os.path.dirname(root)

    rows, dropped, seen = [], Counter(), set()
    for i in range(len(df)):
        r = df.iloc[i]
        raw = r[label_col]
        key = str(raw).strip().lower()
        if key not in label_map:
            dropped[key] += 1
            continue
        label = label_map[key]
        if label not in classes:
            raise SystemExit(f"label map sends {key!r} to {label!r}, which is not one of {list(classes)}")
        uid = _safe_id(prefix + (str(r[id_col]) if id_col else f"{r['_file']}_{i}"))
        if uid in seen:
            uid = _safe_id(f"{uid}_{i}")
        seen.add(uid)
        path = os.path.abspath(os.path.join(wav_dir, uid + ".wav"))
        if overwrite or not os.path.exists(path):
            wav, sr = decode_audio(r[audio_col], base_dir)
            sf.write(path, wav, sr)
        rows.append({"utt_id": uid, "path": path, "label": label, "label_id": list(classes).index(label),
                     "fold": 1, "speaker": str(r[speaker_col]) if speaker_col else ""})
    if dropped:
        log("dropped (no target class): " + ", ".join(f"{k}={v}" for k, v in dropped.most_common()))
    if not rows:
        raise SystemExit("no utterance mapped to a target class; check --label_col / --label_map")
    return rows
