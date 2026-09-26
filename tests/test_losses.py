import torch
import torch.nn.functional as F

from afea.losses import continuity_loss, seal_loss


def test_seal_is_full_BxB_cross_stream():
    torch.manual_seed(0)
    B, D, margin = 5, 7, 1.0
    sw, sf = torch.randn(B, D), torch.randn(B, D)
    y = torch.tensor([0, 1, 0, 2, 1])
    zw, zf = F.normalize(sw, dim=-1), F.normalize(sf, dim=-1)
    total = 0.0
    for i in range(B):
        for j in range(B):                       # every WavLM_i against every Fbank_j
            d = torch.norm(zw[i] - zf[j])
            c = float(y[i] == y[j])
            total += c * d + (1 - c) * torch.relu(margin - d)
    expected = total / (B * B)
    assert torch.allclose(seal_loss(sw, sf, y, margin), expected, atol=1e-5)


def test_seal_is_not_diagonal_only():
    torch.manual_seed(1)
    sw, sf = torch.randn(4, 6), torch.randn(4, 6)
    y = torch.tensor([0, 0, 1, 1])
    zw, zf = F.normalize(sw, dim=-1), F.normalize(sf, dim=-1)
    diag_only = torch.norm(zw - zf, dim=-1).mean()
    assert not torch.allclose(seal_loss(sw, sf, y), diag_only)


def test_seal_gradient_is_finite_for_identical_vectors():
    x = torch.randn(3, 4, requires_grad=True)
    seal_loss(x, x.detach().clone(), torch.tensor([0, 0, 1])).backward()
    assert torch.isfinite(x.grad).all()


def test_continuity_loss_matches_equations():
    torch.manual_seed(0)
    sw, sf = torch.randn(4, 6), torch.randn(4, 6)
    fw = [torch.randn(4, 6) for _ in range(3)]
    ff = [torch.randn(4, 6) for _ in range(3)]
    w = (0.8, 0.5, 0.2)
    expected = sum(wl * (F.mse_loss(a, sw) + F.mse_loss(b, sf) + F.mse_loss(a, b))
                   for wl, a, b in zip(w, fw, ff))
    assert torch.allclose(continuity_loss(fw, ff, sw, sf, w), expected)
