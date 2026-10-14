import io
import json

import numpy as np
import pytest
import soundfile as sf

from afea.data import read_manifest, write_manifest
from afea.data.parquet_ser import build_parquet_manifest, detect_columns, inspect_parquet, parse_label_map
from afea.trainer import run_cv
from test_train_smoke import _cfg, _synthetic


def _wav_bytes(sr=16000, sec=0.3, seed=0):
    buf = io.BytesIO()
    sf.write(buf, np.random.default_rng(seed).normal(scale=0.1, size=int(sr * sec)).astype(np.float32), sr,
             format="WAV")
    return buf.getvalue()


def _parquet(tmp_path):
    pa = pytest.importorskip("pyarrow", reason="pyarrow is only needed for Parquet corpora")
    import pyarrow.parquet as pq
    d = tmp_path / "pq"
    d.mkdir()
    labels = ["Neutral", "Angry", "Happy", "Sad", "Frustrated", "neutral"]
    t = pa.table({"audio": [{"bytes": _wav_bytes(seed=i), "path": f"f{i}.wav"} for i in range(6)],
                  "emotion": labels, "speaker_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                  "id": [f"utt{i}" for i in range(6)]})
    pq.write_table(t.slice(0, 3), d / "train-00000.parquet")
    pq.write_table(t.slice(3), d / "train-00001.parquet")
    return d


def test_parquet_manifest_maps_labels_and_decodes_audio(tmp_path):
    root = _parquet(tmp_path)
    first = {"audio": {"bytes": b"x", "path": "a.wav"}, "emotion": "Sad"}
    assert detect_columns(["id", "audio", "emotion", "speaker_id"], first) == \
        {"audio": "audio", "label": "emotion", "speaker": "speaker_id", "id": "id"}
    report = inspect_parquet(str(root))
    assert "2 parquet file(s), 6 rows" in report and "speaker_id: s1=2, s2=2, s3=2" in report
    assert "auto-detected: audio=audio, label=emotion" in report
    rows = build_parquet_manifest(str(root), str(tmp_path / "wav"), prefix="thaiser_", log=lambda m: None)
    assert [r["label"] for r in rows] == ["neu", "ang", "hap", "sad", "neu"]     # frustrated dropped
    assert [r["label_id"] for r in rows] == [3, 2, 1, 0, 3]                     # IEMOCAP class order
    wav, sr = sf.read(rows[0]["path"])
    assert sr == 16000 and len(wav) == 4800
    assert parse_label_map("0=neu, 1=ang") == {"0": "neu", "1": "ang"}


def test_cross_corpus_evaluation(tmp_path):
    (tmp_path / "src").mkdir()
    rows = _synthetic(tmp_path / "src", n=24)
    xroot = tmp_path / "tgt"
    xroot.mkdir()
    xrows = _synthetic(xroot, n=12)
    for r in xrows:
        r.update(path="", label="", speaker="", fold=1)
    write_manifest(xrows, str(tmp_path / "x.csv"))
    cfg = _cfg(tmp_path / "src", xc_manifest=str(tmp_path / "x.csv"), xc_feat_root=str(xroot),
               xc_monitor=True)
    s = run_cv(rows, cfg, seeds=[0, 1], device="cpu", out_dir=str(tmp_path / "run"))
    assert set(s["xc_fold_mean"]) == set(s["xc_vote"]) == {"WA", "UAR", "P", "F1"}
    assert np.array(s["xc_confusion"]).sum() == 12 * 2 * 2                      # 2 folds x 2 seeds
    res = json.load(open(tmp_path / "run" / "fold1_seed0.json"))
    assert len(res["xc"]["y_pred"]) == len(read_manifest(str(tmp_path / "x.csv")))
    assert "mon_xc_UAR" in res["history"][0]


def test_thaiser_layout_multiple_mics_and_majority_label(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    from afea.data.parquet_ser import PRESETS

    def flac(seed):
        buf = io.BytesIO()
        sf.write(buf, np.zeros(1600, dtype=np.float32) + 0.01 * seed, 16000, format="FLAC")
        return {"bytes": buf.getvalue(), "path": f"m{seed}.flac"}

    d = tmp_path / "thai"
    d.mkdir()
    pq.write_table(pa.table({
        "audio_id": ["s001_con_a_1.flac", "s001_con_a_2.flac", "s001_con_a_3.flac", "s002_con_b_1.flac"],
        "majority_emo": ["Neutral", "None", "Frustrated", "Sad"],
        "actor_id": ["001", "001", "002", "002"],
        "mic_con": [flac(1), flac(2), flac(3), None],           # last row: condenser mic missing
        "mic_clip": [flac(4), flac(5), flac(6), flac(7)],
    }), d / "train-00000-of-00001.parquet")
    p = PRESETS["thaiser"]
    with pytest.raises(SystemExit):          # preset lists mic_middle / mic_zoom, which this file lacks
        build_parquet_manifest(str(d), str(tmp_path / "w"), p["audio_col"], p["label_col"], p["speaker_col"],
                               p["id_col"], log=lambda m: None)
    logs = []
    rows = build_parquet_manifest(str(d), str(tmp_path / "w"), "mic_con,mic_clip", p["label_col"], p["speaker_col"],
                                  p["id_col"], prefix="thaiser_", log=logs.append)
    assert [r["utt_id"] for r in rows] == ["thaiser_s001_con_a_1", "thaiser_s002_con_b_1"]   # None / Frustrated dropped
    assert [r["label"] for r in rows] == ["neu", "sad"] and [r["speaker"] for r in rows] == ["001", "002"]
    assert rows[0]["path"].endswith(".flac") and sf.read(rows[1]["path"])[0][0] == pytest.approx(0.07, abs=1e-3)
    assert any("mic_con=1, mic_clip=1" in m for m in logs)
