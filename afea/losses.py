"""Auxiliary losses of AFEA-Net: SEAL (Eq. 5-6) and continuity learning (Eq. 19-24)."""

from typing import Sequence

import torch
import torch.nn.functional as F


def _seal_embed(x: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "l2":        # default reading of "normalize them"
        return F.normalize(x, dim=-1)
    if norm == "l2c":       # centre over the batch first, so cosines can be negative (distances up to 2)
        return F.normalize(x - x.mean(0, keepdim=True), dim=-1)
    if norm == "none":      # raw Euclidean distance
        return x
    raise ValueError(norm)


def _seal_dist(s_wav, s_fil, norm):
    zw, zf = _seal_embed(s_wav, norm), _seal_embed(s_fil, norm)
    # Explicit difference instead of torch.cdist: exact and has a stable gradient.
    diff = zw.unsqueeze(1) - zf.unsqueeze(0)                  # [B, B, D]
    return torch.sqrt((diff * diff).sum(-1) + 1e-12)          # [B, B]


def seal_loss(s_wav: torch.Tensor, s_fil: torch.Tensor, labels: torch.Tensor,
              margin: float = 1.0, norm: str = "l2") -> torch.Tensor:
    """Speech Emotion Alignment Learning loss (Eq. 6).

    Builds the full B x B cross-stream matrix: every WavLM embedding i is compared with
    every Fbank embedding j in the batch (not only i == j).
        D_ij = || norm(S_wav^i) - norm(S_fil^j) ||_2
        C_ij = 1 if y_i == y_j else 0
        L    = 1/(B*B) * sum_ij [ C_ij * D_ij + (1 - C_ij) * relu(margin - D_ij) ]
    `norm` is L2 normalisation by default (the paper says "normalize them"); see
    `_seal_embed` for the alternatives. As written in Eq. 6 the distance is not squared.
    """
    dist = _seal_dist(s_wav, s_fil, norm)
    same = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    per_pair = same * dist + (1.0 - same) * F.relu(margin - dist)
    return per_pair.mean()                                    # N = B*B pairs


def continuity_loss(f_wav: Sequence[torch.Tensor], f_fil: Sequence[torch.Tensor],
                    s_wav: torch.Tensor, s_fil: torch.Tensor,
                    weights: Sequence[float]) -> torch.Tensor:
    """Continuity learning loss (Eq. 19-24).

    For AFEA layer l:
        intra_l = MSE(F^l_wav, S_wav) + MSE(F^l_fil, S_fil)   (keep each stream's identity)
        inter_l = MSE(F^l_wav, F^l_fil)                        (complementarity across streams)
        L_con   = sum_l w_l * (intra_l + inter_l),  (w_1, w_2, w_3) = (alpha, beta, gamma)
    """
    if len(weights) != len(f_wav):
        raise ValueError(f"need one weight per AFEA layer: got {len(weights)} for {len(f_wav)} layers")
    total = s_wav.new_zeros(())
    for w, fw, ff in zip(weights, f_wav, f_fil):
        if w == 0:
            continue
        intra = F.mse_loss(fw, s_wav) + F.mse_loss(ff, s_fil)
        inter = F.mse_loss(fw, ff)
        total = total + w * (intra + inter)
    return total


@torch.no_grad()
def seal_stats(s_wav, s_fil, labels, margin: float = 1.0, norm: str = "l2"):
    """Diagnostics: mean positive / negative pair distance and the fraction of negative
    pairs still inside the margin (1.0 = the hinge is active for every negative pair)."""
    dist = _seal_dist(s_wav, s_fil, norm)
    same = labels.unsqueeze(1) == labels.unsqueeze(0)
    neg = dist[~same]
    return {"seal_pos_dist": float(dist[same].mean()),
            "seal_neg_dist": float(neg.mean()) if neg.numel() else 0.0,
            "seal_neg_in_margin": float((neg < margin).float().mean()) if neg.numel() else 0.0}
