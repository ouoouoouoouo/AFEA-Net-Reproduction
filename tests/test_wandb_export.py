import importlib.util
import os

spec = importlib.util.spec_from_file_location(
    "wandb_export", os.path.join(os.path.dirname(__file__), "..", "scripts", "wandb_export.py"))
we = importlib.util.module_from_spec(spec)
spec.loader.exec_module(we)


class FakeRun:
    def __init__(self, name, summary, history=None, state="finished"):
        self.name, self.summary, self.state, self._h = name, summary, state, history or {}

    def history(self, keys, samples, pandas):
        return list(self._h.get(keys[0].split("/")[0], []))


def _summ(wa):
    return {"fold_mean/WA": wa, "fold_mean/UAR": wa + 0.01, "fold_mean/F1": wa - 0.01}


def test_collect_and_tables():
    hist = {f"fold{k}_s42": [{f"fold{k}_s42/mon_test_UAR": u, f"fold{k}_s42/mon_test_WA": u - 0.02}
                             for u in (0.70, 0.75, 0.72)] for k in range(1, 6)}
    runs = [
        FakeRun("iemocap_afea_net_s42", _summ(0.71)),
        FakeRun("iemocap_afea_net_s1", _summ(0.72)),
        FakeRun("iemocap_t_afea_net_s42", _summ(0.74)),
        FakeRun("iemocap_t_afea_net_s42", {}, state="crashed"),       # failed duplicate is ignored
        FakeRun("iemocap_p_afea_net_s42", _summ(0.70), hist),
        FakeRun("unrelated-run", _summ(0.5)),
    ]
    data = we.collect(runs, lambda cfg: cfg.startswith(("p_", "pn_")))
    assert set(data["iemocap"]) == {"afea_net", "t_afea_net", "p_afea_net"}
    assert data["iemocap"]["p_afea_net"][42]["oracle"] == {"WA": 0.73, "UAR": 0.75}
    text = "\n".join(we.tables(data))
    assert "| t_afea_net | 1 | 74.0 |" in text and "71.0 (afea_net) | +3.0 | 75.1 / 75.3" in text
    assert "| p_afea_net | 1 | 70.0 |" in text and "| 73.0 | 75.0 |" in text
    assert "| afea_net | 2 | 71.5 ± 0.5 |" in text


def test_paper_table_markdown_and_latex():
    from afea.report import paper_table, stats_from_runs
    runs = {f"tw_{n}": {42: {"WA": wa, "UAR": wa + 0.01, "P": wa + 0.02, "F1": wa - 0.01},
                        1: {"WA": wa + 0.002, "UAR": wa + 0.012, "P": wa + 0.022, "F1": wa - 0.008}}
            for n, wa in (("fbank", 0.56), ("wavlm", 0.73), ("wo_align", 0.736), ("afea_net", 0.73))}
    md, tex = paper_table(stats_from_runs(runs, "tw_"), "iemocap", "tw")
    text = "\n".join(md)
    assert "| WavLM | 73.1 ± 0.1 |" in text            # mean of 73.0 / 73.2
    assert "| w/o L_ali | **73.7** ± 0.1 |" in text      # best reproduction WA is bold
    assert "| AFEA-1 | – |" in text                     # missing rows are shown, not dropped
    assert "**75.1**" in text and "2 seeds" in text and "best WA on the held-out fold" in text
    assert tex.count(r"\midrule") == 5 and r"\bottomrule" in tex and r"\textbf{73.7}{\scriptsize$\pm$0.1}" in tex
    assert tex.count("{") == tex.count("}")


def test_select_test_by_wa(tmp_path):
    import json
    import numpy as np
    from afea.config import TRAIN_DEFAULTS
    from afea.trainer import run_cv
    rng = np.random.default_rng(0)
    for s in ("wavlm", "fbank"):
        (tmp_path / s).mkdir()
    rows = []
    for i in range(40):
        y = i % 4
        np.save(tmp_path / "wavlm" / f"u{i}.npy", (rng.normal(size=(rng.integers(10, 30), 1024)) + y).astype(np.float16))
        np.save(tmp_path / "fbank" / f"u{i}.npy", (rng.normal(size=(rng.integers(20, 60), 40)) + y).astype(np.float16))
        rows.append({"utt_id": f"u{i}", "label_id": y, "fold": i % 2 + 1})
    cfg = dict(TRAIN_DEFAULTS, feat_root=str(tmp_path), model="afea", num_classes=4, margin=1.0, use_align=True,
               use_con=True, con_weights=[0.8, 0.5, 0.2], epochs=3, batch_size=8, lstm_hidden=8,
               select="test", val_ratio=0, test_select_metric="wa", diag=False)
    run_cv(rows, cfg, folds=[1], device="cpu", out_dir=str(tmp_path / "r"))
    res = json.load(open(tmp_path / "r" / "fold1_seed42.json"))
    best = max(res["history"], key=lambda h: h["mon_test_WA"])
    assert res["selected_epoch"] == best["epoch"] and abs(res["test"]["WA"] - best["mon_test_WA"]) < 1e-9
