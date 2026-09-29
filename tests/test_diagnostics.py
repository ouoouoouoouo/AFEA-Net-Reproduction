import numpy as np
import torch

from afea.config import TRAIN_DEFAULTS
from afea.dataset import DualStreamFeatureDataset, collate_dual
from afea.losses import seal_loss, seal_stats
from afea.model import AFEANet, BiLSTMEncoder, LayerMix, SingleStreamNet
from afea.trainer import run_cv


def test_dropout_post_pool_matches_pre_pool_in_eval():
    torch.manual_seed(0)
    a = BiLSTMEncoder(10, 4, 0.5, "pre_pool").eval()
    b = BiLSTMEncoder(10, 4, 0.5, "post_pool").eval()
    b.load_state_dict(a.state_dict())
    x, l = torch.randn(2, 7, 10), torch.tensor([7, 3])
    assert torch.allclose(a(x, l), b(x, l))


def test_pre_pool_dropout_makes_pooled_features_nonnegative_in_training():
    torch.manual_seed(0)
    enc = BiLSTMEncoder(10, 16, 0.5, "pre_pool").train()
    s = enc(torch.randn(8, 50, 10), torch.full((8,), 50))
    assert (s >= 0).all()      # the reason why SEAL margins >= sqrt(2) are indistinguishable


def test_seal_margins_above_sqrt2_give_identical_gradients_for_nonnegative_inputs():
    torch.manual_seed(0)
    y = torch.tensor([0, 1, 0, 1, 2])
    grads = []
    for m in (1.5, 2.0):
        sw = torch.rand(5, 8, requires_grad=True)
        with torch.no_grad():
            sw.copy_(torch.rand(5, 8, generator=torch.Generator().manual_seed(1)))
        sf = torch.rand(5, 8, generator=torch.Generator().manual_seed(2))
        seal_loss(sw, sf, y, m).backward()
        grads.append(sw.grad.clone())
    assert torch.allclose(grads[0], grads[1])
    # with batch centring the distances exceed sqrt(2) and the margins differ again
    sw = torch.rand(5, 8, generator=torch.Generator().manual_seed(1))
    sf = torch.rand(5, 8, generator=torch.Generator().manual_seed(2))
    assert seal_stats(sw, sf, y, 2.0, "l2c")["seal_neg_dist"] > 2 ** 0.5 - 0.3
    assert not torch.allclose(seal_loss(sw, sf, y, 1.5, "l2c") - seal_loss(sw, sf, y, 2.0, "l2c"),
                              seal_loss(sw, sf, y, 1.5, "l2") - seal_loss(sw, sf, y, 2.0, "l2"))


def test_layer_mix_and_models_accept_all_layer_lists():
    torch.manual_seed(0)
    xs = [torch.randn(5, n, 1024).half() for n in (12, 7)]
    mix = LayerMix(5)
    out = mix(xs)
    assert out.shape == (2, 12, 1024) and out[1, 7:].abs().sum() == 0
    out.sum().backward()
    assert mix.logits.grad is not None
    m = AFEANet(4, lstm_hidden=8, wavlm_num_layers=5).eval()
    o = m(xs, torch.tensor([12, 7]), torch.randn(2, 20, 40), torch.tensor([20, 15]))
    assert o["logits"].shape == (2, 4)
    s = SingleStreamNet(4, 1024, lstm_hidden=8, wavlm_num_layers=5).eval()
    assert s(xs, torch.tensor([12, 7]))["logits"].shape == (2, 4)


def _synthetic(tmp_path, n=40, all_layers=False):
    rng = np.random.default_rng(0)
    for s in ("wavlm", "fbank", "wavlm_all"):
        (tmp_path / s).mkdir()
    rows = []
    for i in range(n):
        y = i % 4
        m, k = rng.integers(10, 30), rng.integers(20, 60)
        w = (rng.normal(size=(m, 1024)) + y).astype(np.float16)
        np.save(tmp_path / "wavlm" / f"u{i}.npy", w)
        np.save(tmp_path / "wavlm_all" / f"u{i}.npy", np.stack([w * (j + 1) for j in range(3)]))
        np.save(tmp_path / "fbank" / f"u{i}.npy", (rng.normal(size=(k, 40)) + y).astype(np.float16))
        rows.append({"utt_id": f"u{i}", "label_id": y, "fold": i % 2 + 1})
    return rows


def _cfg(tmp_path, **kw):
    cfg = dict(TRAIN_DEFAULTS, feat_root=str(tmp_path), model="afea", num_classes=4, margin=1.0,
               use_align=True, use_con=True, con_weights=[0.8, 0.5, 0.2], epochs=2, batch_size=8,
               lstm_hidden=8, val_ratio=0.2)
    cfg.update(kw)
    return cfg


def test_all_layer_dataset_collate(tmp_path):
    rows = _synthetic(tmp_path)
    ds = DualStreamFeatureDataset(rows[:3], str(tmp_path), ("wavlm_all", "fbank"))
    b = collate_dual([ds[i] for i in range(3)])
    assert isinstance(b["wavlm"], list) and b["wavlm"][0].shape[0] == 3
    assert b["wavlm_len"].tolist() == [x.shape[1] for x in b["wavlm"]]


def test_diagnostics_do_not_change_training(tmp_path):
    rows = _synthetic(tmp_path)
    a = run_cv(rows, _cfg(tmp_path, diag=True), folds=[1], device="cpu")
    b = run_cv(rows, _cfg(tmp_path, diag=False), folds=[1], device="cpu")
    assert a["fold_mean"] == b["fold_mean"]


def test_new_variants_train_and_log_diagnostics(tmp_path):
    rows = _synthetic(tmp_path)
    for kw in ({"seal_norm": "l2c"}, {"seal_norm": "none"}, {"dropout_pos": "post_pool"},
               {"wavlm_layers": "all"}, {"model": "wavlm", "wavlm_layers": "all"}):
        out = tmp_path / ("run_" + "_".join(f"{k}{v}" for k, v in kw.items()))
        run_cv(rows, _cfg(tmp_path, **kw), folds=[1], device="cpu", out_dir=str(out))
    import json
    res = json.load(open(tmp_path / "run_wavlm_layersall" / "fold1_seed42.json"))
    assert len(res["layer_weights"]) == 3 and "WA_shuf_fil" in res["test_probe"]
    assert "diag/ise_a_wav_L3" in res["history"][0] and "val_WA_shuf_wav" in res["history"][0]
    assert "diag/grad_enc_fil" in res["history"][0] and "diag/layer_w02" in res["history"][0]
