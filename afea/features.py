"""Frame-level feature extraction: WavLM (1024-d) and Kaldi-style Fbank (40-d).

The two streams are extracted independently and keep their own frame rates
(WavLM: 20 ms hop -> ~50 frames/s; Fbank: 10 ms hop -> 100 frames/s).
They are NOT aligned or resampled to a common length; the model encodes each
stream with its own BiLSTM and pools over time separately.
"""

from typing import Optional

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load a mono float32 waveform [T] at 16 kHz (IEMOCAP is 16 kHz, RAVDESS 48 kHz)."""
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(axis=1))
    if sr != target_sr:
        import torchaudio.functional as AF
        wav = AF.resample(wav, sr, target_sr)
    return wav


def compute_fbank(wav: torch.Tensor, num_mel_bins: int = 40, frame_length_ms: float = 25.0,
                  frame_shift_ms: float = 10.0, cmvn: bool = True) -> torch.Tensor:
    """40-d log Mel filterbank, 25 ms window / 10 ms shift -> [N, 40].

    Uses torchaudio's Kaldi-compatible implementation (Hamming window, pre-emphasis 0.97,
    no dither for determinism). With `cmvn=True` each utterance is mean/variance normalised
    per dimension (assumption: the paper does not specify normalisation).
    """
    import torchaudio.compliance.kaldi as kaldi
    # Kaldi expects int16-range amplitudes; scaling only shifts log-energies by a constant.
    feats = kaldi.fbank(wav.unsqueeze(0) * 32768.0, num_mel_bins=num_mel_bins,
                        frame_length=frame_length_ms, frame_shift=frame_shift_ms,
                        dither=0.0, energy_floor=0.0, sample_frequency=SAMPLE_RATE,
                        window_type="hamming")
    if cmvn:
        feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-5)
    return feats


class WavLMExtractor:
    """Frozen WavLM feature extractor returning frame-level hidden states [M, 1024].

    Default checkpoint: `microsoft/wavlm-large` (hidden size 1024, the dimension reported
    in the paper). `layer=None` returns the final transformer output (`last_hidden_state`);
    an integer selects `hidden_states[layer]` (0 = CNN/projection output, 24 = last layer).

    `input_norm` controls waveform normalisation before the model:
      "auto" - whatever the checkpoint's feature extractor config says (`do_normalize`)
      "on"   - per-utterance zero-mean / unit-variance, i.e. `F.layer_norm(wav, wav.shape)` as in the
               official microsoft/unilm WavLM example when `cfg.normalize` is True (WavLM Large)
      "off"  - raw waveform
    """

    def __init__(self, checkpoint: str = "microsoft/wavlm-large", layer: Optional[int] = None,
                 device: str = "cpu", model=None, feature_extractor=None, input_norm: str = "auto"):
        from transformers import AutoFeatureExtractor, WavLMModel
        if input_norm not in ("auto", "on", "off"):
            raise ValueError(input_norm)
        self.device = device
        self.layer = layer
        self.model = (model if model is not None else WavLMModel.from_pretrained(checkpoint)).to(device).eval()
        self.feature_extractor = (feature_extractor if feature_extractor is not None
                                  else AutoFeatureExtractor.from_pretrained(checkpoint))
        self.config_do_normalize = bool(getattr(self.feature_extractor, "do_normalize", False))
        self.normalize = self.config_do_normalize if input_norm == "auto" else input_norm == "on"
        self.feature_extractor.do_normalize = False          # we apply it ourselves (below)

    def _inputs(self, wav: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        inputs = self.feature_extractor(wav.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return inputs["input_values"].to(self.device)

    @torch.no_grad()
    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        x = self._inputs(wav)
        out = self.model(x, output_hidden_states=self.layer is not None)
        h = out.last_hidden_state if self.layer is None else out.hidden_states[self.layer]
        return h[0].float().cpu()

    @torch.no_grad()
    def all_layers(self, wav: torch.Tensor) -> torch.Tensor:
        """All hidden states stacked: [L, M, D] (L = 25 for WavLM-Large: CNN/projection + 24 layers)."""
        out = self.model(self._inputs(wav), output_hidden_states=True)
        return torch.stack([h[0] for h in out.hidden_states]).float().cpu()


def save_feature(path: str, feat: torch.Tensor, fp16: bool = True) -> None:
    arr = feat.numpy().astype(np.float16 if fp16 else np.float32)
    np.save(path, arr)
