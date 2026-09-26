"""AFEA-Net: BiLSTM encoders + SEAL + stacked AFEA (ISE + ISA) + FCN classifier.

Shapes (paper Table 1 uses [B, 1, D]; we drop the singleton axis and use [B, D]):
    X_wav [B, M, 1024] --BiLSTM--> S_wav [B, M, D] --max over valid M--> S~_wav [B, D]
    X_fil [B, N,   40] --BiLSTM--> S_fil [B, N, D] --max over valid N--> S~_fil [B, D]
M and N are independent (different frame rates, no alignment between streams).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_max_pool(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Temporal max pooling over the valid (un-padded) frames only. x: [B, T, D] -> [B, D]."""
    t = torch.arange(x.shape[1], device=x.device)
    mask = t.unsqueeze(0) < lengths.to(x.device).unsqueeze(1)          # [B, T]
    x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return x.max(dim=1).values


class BiLSTMEncoder(nn.Module):
    """One bidirectional LSTM layer + dropout + temporal max pooling (Eq. 1-4).

    `hidden_per_direction` is the LSTM hidden size of each direction; the output
    dimension is D = 2 * hidden_per_direction (forward/backward states concatenated).
    """

    def __init__(self, input_dim: int, hidden_per_direction: int = 256, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_per_direction, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = 2 * hidden_per_direction

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return masked_max_pool(self.dropout(out), lengths)


class InterSpeechExcitation(nn.Module):
    """ISE (Eq. 7-11): stream-level softmax weighting + cross-stream excitation.

    G_wav = Linear_wav([S_wav; S_fil]) in R^{B x 1},  G_fil = Linear_fil([S_wav; S_fil])
    (A_wav, A_fil) = softmax over the two streams  ->  A_wav + A_fil = 1 per sample
    E_wav = A_wav * S_wav,  E_fil = A_fil * S_fil
    I_wav = S_wav + E_fil,  I_fil = S_fil + E_wav   (each stream is excited by the OTHER one)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.map_wav = nn.Linear(2 * dim, 1)
        self.map_fil = nn.Linear(2 * dim, 1)

    def forward(self, s_wav: torch.Tensor, s_fil: torch.Tensor):
        s_cat = torch.cat([s_wav, s_fil], dim=-1)                                   # [B, 2D]
        g = torch.cat([self.map_wav(s_cat), self.map_fil(s_cat)], dim=-1)           # [B, 2]
        a = torch.softmax(g, dim=-1)                                                # [B, 2]
        a_wav, a_fil = a[:, :1], a[:, 1:]
        e_wav, e_fil = a_wav * s_wav, a_fil * s_fil
        return s_wav + e_fil, s_fil + e_wav, a


class InterSpeechAggregation(nn.Module):
    """ISA (Eq. 12-13): sigmoid feature-wise gating (NOT a softmax; W_wav and W_fil are
    independent values in (0, 1) per channel and need not sum to 1).

    [W_wav; W_fil] = sigmoid(MLP([I_wav; I_fil])),  MLP = Linear(2D, H) -> ReLU -> Linear(H, 2D)
    F_fusion = W_wav * I_wav + W_fil * I_fil
    """

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = hidden or dim
        self.mlp = nn.Sequential(nn.Linear(2 * dim, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, 2 * dim))

    def forward(self, i_wav: torch.Tensor, i_fil: torch.Tensor):
        w = torch.sigmoid(self.mlp(torch.cat([i_wav, i_fil], dim=-1)))              # [B, 2D]
        w_wav, w_fil = w.chunk(2, dim=-1)
        return w_wav * i_wav + w_fil * i_fil, w_wav, w_fil


class AFEALayer(nn.Module):
    """One AFEA module: ISE -> ISA -> averaging with the layer inputs (Eq. 14-15)."""

    def __init__(self, dim: int, isa_hidden: Optional[int] = None):
        super().__init__()
        self.ise = InterSpeechExcitation(dim)
        self.isa = InterSpeechAggregation(dim, isa_hidden)

    def forward(self, f_wav: torch.Tensor, f_fil: torch.Tensor):
        i_wav, i_fil, a = self.ise(f_wav, f_fil)
        fusion, w_wav, w_fil = self.isa(i_wav, i_fil)
        return {"fusion": fusion, "f_wav": (fusion + f_wav) / 2, "f_fil": (fusion + f_fil) / 2,
                "ise_weights": a, "isa_w_wav": w_wav, "isa_w_fil": w_fil}


def make_classifier(in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.0) -> nn.Module:
    """FCN head (Eq. 17): Linear(in, 512) -> ReLU -> [Dropout] -> Linear(512, C)."""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
                         nn.Dropout(dropout), nn.Linear(hidden, num_classes))


class AFEANet(nn.Module):
    """Dual-stream AFEA-Net.

    num_afea_layers = 0 reproduces the "w/o AFEA" ablation: S~_wav and S~_fil are
    concatenated and classified directly.
    """

    def __init__(self, num_classes: int, wavlm_dim: int = 1024, fbank_dim: int = 40,
                 lstm_hidden: int = 256, lstm_dropout: float = 0.5, num_afea_layers: int = 3,
                 isa_hidden: Optional[int] = None, fcn_hidden: int = 512, fcn_dropout: float = 0.0):
        super().__init__()
        self.enc_wav = BiLSTMEncoder(wavlm_dim, lstm_hidden, lstm_dropout)
        self.enc_fil = BiLSTMEncoder(fbank_dim, lstm_hidden, lstm_dropout)
        dim = self.enc_wav.out_dim
        self.afea = nn.ModuleList([AFEALayer(dim, isa_hidden) for _ in range(num_afea_layers)])
        cls_in = dim if num_afea_layers > 0 else 2 * dim
        self.classifier = make_classifier(cls_in, num_classes, fcn_hidden, fcn_dropout)

    def forward(self, wavlm, wavlm_len, fbank, fbank_len) -> Dict[str, object]:
        s_wav = self.enc_wav(wavlm, wavlm_len)
        s_fil = self.enc_fil(fbank, fbank_len)
        f_wav, f_fil = s_wav, s_fil
        f_wav_list: List[torch.Tensor] = []
        f_fil_list: List[torch.Tensor] = []
        extras = []
        if len(self.afea) == 0:
            fused = torch.cat([s_wav, s_fil], dim=-1)
        else:
            for layer in self.afea:
                out = layer(f_wav, f_fil)
                f_wav, f_fil, fused = out["f_wav"], out["f_fil"], out["fusion"]
                f_wav_list.append(f_wav)
                f_fil_list.append(f_fil)
                extras.append(out)
        return {"logits": self.classifier(fused), "s_wav": s_wav, "s_fil": s_fil,
                "f_wav": f_wav_list, "f_fil": f_fil_list, "fusion": fused, "afea": extras}


class SingleStreamNet(nn.Module):
    """Single-stream baseline of Table 4: BiLSTM -> max pooling -> FCN."""

    def __init__(self, num_classes: int, input_dim: int, lstm_hidden: int = 256,
                 lstm_dropout: float = 0.5, fcn_hidden: int = 512, fcn_dropout: float = 0.0):
        super().__init__()
        self.enc = BiLSTMEncoder(input_dim, lstm_hidden, lstm_dropout)
        self.classifier = make_classifier(self.enc.out_dim, num_classes, fcn_hidden, fcn_dropout)

    def forward(self, x, x_len) -> Dict[str, object]:
        s = self.enc(x, x_len)
        return {"logits": self.classifier(s), "fusion": s}
