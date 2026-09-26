import numpy as np

from afea.config import TRAIN_DEFAULTS
from afea.trainer import run_cv


def _synthetic(tmp_path, n=40, folds=2):
    rng = np.random.default_rng(0)
    for s in ("wavlm", "fbank"):
        (tmp_path / s).mkdir()
    rows = []
    for i in range(n):
        y = i % 4
        m, k = rng.integers(10, 30), rng.integers(20, 60)   # unaligned lengths
        np.save(tmp_path / "wavlm" / f"u{i}.npy", (rng.normal(size=(m, 1024)) + y).astype(np.float16))
        np.save(tmp_path / "fbank" / f"u{i}.npy", (rng.normal(size=(k, 40)) + y).astype(np.float16))
        rows.append({"utt_id": f"u{i}", "label_id": y, "fold": i % folds + 1})
    return rows


def _cfg(tmp_path, **kw):
    cfg = dict(TRAIN_DEFAULTS, feat_root=str(tmp_path), model="afea", num_classes=4, margin=1.0,
               use_align=True, use_con=True, con_weights=[0.8, 0.5, 0.2], epochs=2, batch_size=8,
               lstm_hidden=8, val_ratio=0.2)
    cfg.update(kw)
    return cfg


def test_full_model_trains(tmp_path):
    rows = _synthetic(tmp_path)
    s = run_cv(rows, _cfg(tmp_path), device="cpu", out_dir=str(tmp_path / "run"))
    assert set(s["fold_mean"]) == {"WA", "UAR", "P", "F1"}
    assert (tmp_path / "run" / "summary.json").exists()


def test_ablations_train(tmp_path):
    rows = _synthetic(tmp_path)
    for kw in ({"model": "wavlm"}, {"model": "fbank"}, {"afea_layers": 0, "use_con": False},
               {"use_align": False}, {"afea_layers": 1, "use_con": False}, {"select": "last", "val_ratio": 0}):
        run_cv(rows, _cfg(tmp_path, **kw), folds=[1], device="cpu")
