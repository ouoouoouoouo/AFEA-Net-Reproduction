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
SPEAKER_NAMES = ("speaker", "speaker_id", "spk", "spk_id", "actor", "actor_id", "client_id")
ID_NAMES = ("id", "utt_id", "utterance_id", "audio_id", "name", "file_name", "filename", "key")
# Raw audio bytes in these formats are written to disk as they are (no decoding, smallest files).
COPY_FORMATS = (".wav", ".flac", ".ogg")

# THAI SER (VISTEC / AIResearch) Hugging Face export: every utterance is recorded by several
# microphones; mic_con is the one named by audio_id, the others are fallbacks when it is missing.
# majority_emo = annotators' majority vote (perceived emotion, like IEMOCAP's labels); assigned_emo is the
# emotion the actor was asked to perform. See ASSUMPTIONS.md X-2 / X-3.
PRESETS = {
    "thaiser": {"audio_col": "mic_con,mic_clip,mic_middle,mic_zoom", "label_col": "majority_emo",
                "speaker_col": "actor_id", "id_col": "audio_id"},
}


def parquet_files(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    files = sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no .parquet files under {root}")
    return files


def _pq():
    import pyarrow.parquet as pq
    return pq


def _is_audio_cell(v) -> bool:
    if isinstance(v, dict):
        return "bytes" in v or "array" in v
    return isinstance(v, (bytes, bytearray))


def _first_rows(path: str, n: int) -> List[Dict]:
    """Only the first record batch of one file is read (the corpus may be tens of GB)."""
    batch = next(_pq().ParquetFile(path).iter_batches(batch_size=max(n, 1)), None)
    return batch.to_pylist()[:n] if batch is not None else []


def detect_columns(columns: Sequence[str], first: Dict) -> Dict[str, Optional[str]]:
    lower = {c.lower(): c for c in columns}

    def by_name(names):
        return next((lower[n] for n in names if n in lower), None)

    audio = next((c for c in columns if _is_audio_cell(first.get(c))), None) or by_name(AUDIO_NAMES)
    return {"audio": audio, "label": by_name(LABEL_NAMES), "speaker": by_name(SPEAKER_NAMES),
            "id": by_name(ID_NAMES)}


def _short(v, n=80):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_short(x, 30)}" for k, x in v.items()) + "}"
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, (list, np.ndarray)) and len(v) > 8:
        return f"<sequence of {len(v)}>"
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def _scalar_columns(schema) -> List[str]:
    import pyarrow.types as pt
    return [f.name for f in schema if pt.is_string(f.type) or pt.is_large_string(f.type) or pt.is_integer(f.type)
            or pt.is_boolean(f.type) or pt.is_dictionary(f.type)]


def inspect_parquet(root: str, n_rows: int = 3) -> str:
    """Schema, a few rows and value counts of the small columns; audio is never loaded in bulk."""
    pq = _pq()
    files = parquet_files(root)
    meta = [pq.ParquetFile(f).metadata for f in files]
    schema = pq.ParquetFile(files[0]).schema_arrow
    rows = _first_rows(files[0], n_rows)
    out = [f"{len(files)} parquet file(s), {sum(m.num_rows for m in meta)} rows",
           "files: " + ", ".join(os.path.basename(f) for f in files[:10]) + (" ..." if len(files) > 10 else ""),
           "", "columns:"]
    for f in schema:
        out.append(f"  {f.name:24s} {str(f.type)[:60]:60s} e.g. {_short(rows[0].get(f.name)) if rows else ''}")
    out += ["", f"first {len(rows)} rows:"]
    for r in rows:
        out.append("  " + " | ".join(f"{k}={_short(v, 40)}" for k, v in r.items()))
    small = _scalar_columns(schema)
    counts = {c: Counter() for c in small}
    for f in files:  # only the small columns are read
        t = pq.read_table(f, columns=small)
        for c in small:
            counts[c].update(t.column(c).to_pylist())
    out += ["", "low-cardinality columns (value counts over all files):"]
    for c, vc in counts.items():
        if 1 < len(vc) <= 30:
            out.append(f"  {c}: " + ", ".join(f"{k}={v}" for k, v in vc.most_common()))
    det = detect_columns(schema.names, rows[0] if rows else {})
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
        cell = bytes(cell)
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
    s = os.path.splitext(s)[0] if os.path.splitext(s)[1].lower() in (".wav", ".flac", ".mp3", ".ogg") else s
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def _first_audio(r: Dict, audio_cols: Sequence[str]):
    """First non-empty audio cell among the candidate columns -> (cell, column) or (None, None)."""
    for c in audio_cols:
        v = r.get(c)
        if v is not None and not (isinstance(v, dict) and v.get("bytes") is None and v.get("array") is None
                                  and not v.get("path")):
            return v, c
    return None, None


def _write_audio(cell, out_stem: str, base_dir: str) -> str:
    """Write one utterance to disk; returns the file path. Raw wav/flac/ogg bytes are copied as they are."""
    import soundfile as sf
    if isinstance(cell, dict) and cell.get("bytes") is not None:
        ext = os.path.splitext(cell.get("path") or "")[1].lower()
        if ext in COPY_FORMATS:
            path = out_stem + ext
            with open(path, "wb") as f:
                f.write(cell["bytes"])
            return path
    wav, sr = decode_audio(cell, base_dir)
    path = out_stem + ".wav"
    sf.write(path, wav, sr)
    return path


def build_parquet_manifest(root: str, wav_dir: str, audio_col: Optional[str] = None,
                           label_col: Optional[str] = None, speaker_col: Optional[str] = None,
                           id_col: Optional[str] = None, label_map: Optional[Dict[str, str]] = None,
                           classes: Sequence[str] = IEMOCAP_CLASSES, prefix: str = "",
                           overwrite: bool = False, log=print, batch_size: int = 64) -> List[Dict]:
    """Write audio to <wav_dir>/<utt_id>.<ext> and return manifest rows (fold = 1: test-only corpus).
    `audio_col` may list several columns ("mic_con,mic_clip"): the first non-empty one is used.
    Files are streamed in small record batches, so memory stays low for large corpora."""
    pq = _pq()
    files = parquet_files(root)
    schema = pq.ParquetFile(files[0]).schema_arrow
    det = detect_columns(schema.names, (_first_rows(files[0], 1) or [{}])[0])
    audio_col, label_col = audio_col or det["audio"], label_col or det["label"]
    speaker_col, id_col = speaker_col or det["speaker"], id_col or det["id"]
    if audio_col is None or label_col is None:
        raise SystemExit(f"could not detect the audio / label columns (detected {det}); "
                         "pass --audio_col / --label_col (see scripts/inspect_parquet.py)")
    audio_cols = [c.strip() for c in audio_col.split(",")]
    missing = [c for c in audio_cols + [label_col, speaker_col, id_col] if c and c not in schema.names]
    if missing:
        raise SystemExit(f"columns not in the Parquet schema: {missing}; available: {schema.names}")
    log(f"columns: audio={audio_cols} label={label_col} speaker={speaker_col} id={id_col}")
    label_map = label_map if label_map is not None else dict(DEFAULT_LABEL_MAP)
    os.makedirs(wav_dir, exist_ok=True)
    base_dir = root if os.path.isdir(root) else os.path.dirname(root)
    cols = audio_cols + [c for c in (label_col, speaker_col, id_col) if c]
    used_mic, no_audio = Counter(), 0

    rows, dropped, seen = [], Counter(), set()
    for fi, f in enumerate(files):
        pf, i = pq.ParquetFile(f), 0
        fstem = os.path.splitext(os.path.basename(f))[0]
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            for r in batch.to_pylist():
                i += 1
                key = str(r[label_col]).strip().lower()
                if key not in label_map:
                    dropped[key] += 1
                    continue
                label = label_map[key]
                if label not in classes:
                    raise SystemExit(f"label map sends {key!r} to {label!r}, which is not one of {list(classes)}")
                uid = _safe_id(prefix + (str(r[id_col]) if id_col else f"{fstem}_{i - 1}"))
                if uid in seen:
                    uid = _safe_id(f"{uid}_{fstem}_{i - 1}")
                cell, col = _first_audio(r, audio_cols)
                if cell is None:
                    no_audio += 1
                    continue
                seen.add(uid)
                used_mic[col] += 1
                stem = os.path.abspath(os.path.join(wav_dir, uid))
                done = [stem + e for e in COPY_FORMATS if os.path.exists(stem + e)]
                path = done[0] if done and not overwrite else _write_audio(cell, stem, base_dir)
                rows.append({"utt_id": uid, "path": path, "label": label, "label_id": list(classes).index(label),
                             "fold": 1, "speaker": str(r[speaker_col]) if speaker_col else ""})
        log(f"  {fi + 1}/{len(files)} {os.path.basename(f)}: {len(rows)} utterances kept so far")
    log("audio column used: " + ", ".join(f"{k}={v}" for k, v in used_mic.most_common())
        + (f"; {no_audio} rows without audio skipped" if no_audio else ""))
    if dropped:
        log("dropped (no target class): " + ", ".join(f"{k}={v}" for k, v in dropped.most_common()))
    if not rows:
        raise SystemExit("no utterance mapped to a target class; check --label_col / --label_map")
    return rows
