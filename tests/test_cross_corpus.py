import io
import json

import numpy as np
import pandas as pd
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
    d = tmp_path / "pq"
    d.mkdir()
    labels = ["Neutral", "Angry", "Happy", "Sad", "Frustrated", "neutral"]
    df = pd.DataFrame({"audio": [{"bytes": _wav_bytes(seed=i), "path": f"f{i}.wav"} for i in range(6)],
                       "emotion": labels, "speaker_id": ["s1", "s1", "s2", "s2", "s3", "s3"],
                       "id": [f"utt{i}" for i in range(6)]})
    df.iloc[:3].to_parquet(d / "train-00000.parquet")
    df.iloc[3:].to_parquet(d / "train-00001.parquet")
    return d


def test_parquet_manifest_maps_labels_and_decodes_audio(tmp_path):
    root = _parquet(tmp_path)
    assert detect_columns(pd.read_parquet(root / "train-00000.parquet")) == \
        {"audio": "audio", "label": "emotion", "speaker": "speaker_id", "id": "id"}
    assert "auto-detected" in inspect_parquet(str(root))
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
