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
