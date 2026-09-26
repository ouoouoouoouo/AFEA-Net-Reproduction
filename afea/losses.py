"""Auxiliary losses of AFEA-Net: SEAL (Eq. 5-6) and continuity learning (Eq. 19-24)."""

from typing import Sequence

import torch
import torch.nn.functional as F


def seal_loss(s_wav: torch.Tensor, s_fil: torch.Tensor, labels: torch.Tensor,
              margin: float = 1.0) -> torch.Tensor:
    """Speech Emotion Alignment Learning loss (Eq. 6).

    Builds the full B x B cross-stream matrix: every WavLM embedding i is compared with
    every Fbank embedding j in the batch (not only i == j).
        D_ij = || norm(S_wav^i) - norm(S_fil^j) ||_2
        C_ij = 1 if y_i == y_j else 0
        L    = 1/(B*B) * sum_ij [ C_ij * D_ij + (1 - C_ij) * relu(margin - D_ij) ]
    `norm` is L2 normalisation (the paper says "normalize them"). As written in Eq. 6 the
    distance is not squared.
    """
    zw = F.normalize(s_wav, dim=-1)
    zf = F.normalize(s_fil, dim=-1)
    # Explicit difference instead of torch.cdist: exact and has a stable gradient.
    diff = zw.unsqueeze(1) - zf.unsqueeze(0)                  # [B, B, D]
    dist = torch.sqrt((diff * diff).sum(-1) + 1e-12)          # [B, B]
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
