import torch

from afea.model import AFEANet, InterSpeechAggregation, InterSpeechExcitation, masked_max_pool


def test_masked_max_pool_ignores_padding():
    x = torch.tensor([[[1.0], [5.0], [100.0]], [[-3.0], [-1.0], [-2.0]]])
    out = masked_max_pool(x, torch.tensor([2, 3]))
    assert out.squeeze(-1).tolist() == [5.0, -1.0]


def test_ise_softmax_over_streams_and_cross_excitation():
    torch.manual_seed(0)
    ise = InterSpeechExcitation(8)
    sw, sf = torch.randn(4, 8), torch.randn(4, 8)
    i_wav, i_fil, a = ise(sw, sf)
    assert a.shape == (4, 2)
    assert torch.allclose(a.sum(-1), torch.ones(4))           # A_wav + A_fil = 1
    # cross-stream: I_wav = S_wav + A_fil * S_fil ; I_fil = S_fil + A_wav * S_wav
    assert torch.allclose(i_wav, sw + a[:, 1:] * sf)
    assert torch.allclose(i_fil, sf + a[:, :1] * sw)


def test_isa_is_sigmoid_not_softmax():
    torch.manual_seed(0)
    isa = InterSpeechAggregation(8)
    iw, if_ = torch.randn(4, 8), torch.randn(4, 8)
    fused, w_wav, w_fil = isa(iw, if_)
    assert w_wav.shape == w_fil.shape == (4, 8)
    assert ((w_wav > 0) & (w_wav < 1)).all() and ((w_fil > 0) & (w_fil < 1)).all()
    assert not torch.allclose(w_wav + w_fil, torch.ones_like(w_wav))  # independent gates
    assert torch.allclose(fused, w_wav * iw + w_fil * if_)


def test_afeanet_unaligned_streams_and_layer_averaging():
    torch.manual_seed(0)
    model = AFEANet(num_classes=4, lstm_hidden=16, num_afea_layers=3).eval()
    B = 3
    wavlm_len = torch.tensor([50, 20, 33])      # ~50 frames/s
    fbank_len = torch.tensor([100, 41, 67])     # 100 frames/s -> different, unaligned lengths
    out = model(torch.randn(B, 50, 1024), wavlm_len, torch.randn(B, 100, 40), fbank_len)
    assert out["logits"].shape == (B, 4)
    assert out["s_wav"].shape == out["s_fil"].shape == (B, 32)
    assert len(out["f_wav"]) == len(out["f_fil"]) == 3
    # Eq. 14-15: F^l = (F_fusion^l + F^{l-1}) / 2, with F^0 = S~
    prev_w, prev_f = out["s_wav"], out["s_fil"]
    for l, extra in enumerate(out["afea"]):
        assert torch.allclose(out["f_wav"][l], (extra["fusion"] + prev_w) / 2)
        assert torch.allclose(out["f_fil"][l], (extra["fusion"] + prev_f) / 2)
        prev_w, prev_f = out["f_wav"][l], out["f_fil"][l]
    assert torch.equal(out["fusion"], out["afea"][-1]["fusion"])  # classifier sees last fusion


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    model = AFEANet(num_classes=4, lstm_hidden=8, num_afea_layers=2).eval()
    w, f = torch.randn(1, 30, 1024), torch.randn(1, 60, 40)
    a = model(w, torch.tensor([30]), f, torch.tensor([60]))["logits"]
    wp = torch.cat([w, torch.randn(1, 7, 1024)], 1)
    fp = torch.cat([f, torch.randn(1, 9, 40)], 1)
    b = model(wp, torch.tensor([30]), fp, torch.tensor([60]))["logits"]
    assert torch.allclose(a, b, atol=1e-6)


def test_no_afea_concat_baseline():
    model = AFEANet(num_classes=4, lstm_hidden=8, num_afea_layers=0).eval()
    out = model(torch.randn(2, 10, 1024), torch.tensor([10, 5]), torch.randn(2, 20, 40), torch.tensor([20, 9]))
    assert out["fusion"].shape == (2, 32) and out["f_wav"] == []


def test_inter_continuity_term_is_independent_of_afea_parameters():
    """Eq. 14-15 imply F^l_wav - F^l_fil = (S~_wav - S~_fil) / 2^l, so the inter-speech
    continuity loss (Eq. 24) only sees the pooled BiLSTM features, never the AFEA modules."""
    torch.manual_seed(0)
    model = AFEANet(num_classes=4, lstm_hidden=8, num_afea_layers=4).eval()
    out = model(torch.randn(3, 20, 1024), torch.tensor([20, 11, 7]), torch.randn(3, 40, 40), torch.tensor([40, 30, 9]))
    for l in range(4):
        expected = (out["s_wav"] - out["s_fil"]) / 2 ** (l + 1)
        assert torch.allclose(out["f_wav"][l] - out["f_fil"][l], expected, atol=1e-6)


def test_isa_separate_mlps_are_independent_sigmoid_gates():
    torch.manual_seed(0)
    isa = InterSpeechAggregation(8, mlp="separate")
    iw, if_ = torch.randn(4, 8), torch.randn(4, 8)
    fused, w_wav, w_fil = isa(iw, if_)
    assert torch.allclose(w_wav, torch.sigmoid(isa.mlp_wav(torch.cat([iw, if_], -1))))
    assert torch.allclose(w_fil, torch.sigmoid(isa.mlp_fil(torch.cat([iw, if_], -1))))
    assert torch.allclose(fused, w_wav * iw + w_fil * if_)
    model = AFEANet(num_classes=4, lstm_hidden=8, isa_mlp="separate").eval()
    out = model(torch.randn(2, 10, 1024), torch.tensor([10, 6]), torch.randn(2, 20, 40), torch.tensor([20, 9]))
    assert out["logits"].shape == (2, 4)
