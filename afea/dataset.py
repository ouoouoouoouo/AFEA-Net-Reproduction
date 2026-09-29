"""Dataset over pre-extracted features and a collate function that pads each
stream independently (the WavLM and Fbank sequences keep their own lengths)."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# Process-wide feature cache (keyed by file path) so that all folds / seeds run in one
# process read each .npy from disk only once.
_CACHE: Dict[str, np.ndarray] = {}


class DualStreamFeatureDataset(Dataset):
    def __init__(self, rows: List[Dict], feat_root: str, streams=("wavlm", "fbank"),
                 max_frames: Optional[Dict[str, int]] = None, cache: bool = False):
        self.rows = rows
        self.feat_root = feat_root
        self.streams = tuple(streams)
        self.max_frames = max_frames or {}
        self.cache = cache

    def __len__(self):
        return len(self.rows)

    def _load(self, stream: str, utt_id: str) -> torch.Tensor:
        path = os.path.join(self.feat_root, stream, utt_id + ".npy")
        if stream == "wavlm_all":
            # [L, M, 1024] float16, ~25x larger than one layer: memory-map instead of a per-process
            # cache so that concurrent jobs share the OS page cache. Kept in float16; LayerMix
            # mixes the layers on the GPU.
            x = np.load(path, mmap_mode="r")
            limit = self.max_frames.get("wavlm")
            if limit:
                x = x[:, :limit]
            return torch.from_numpy(np.array(x))           # copy out of the (read-only) mmap
        x = _CACHE.get(path) if self.cache else None
        if x is None:
            x = np.load(path)                      # stored as float16 -> half the RAM of float32
            if self.cache:
                _CACHE[path] = x
        limit = self.max_frames.get(stream)
        if limit:
            x = x[:limit]
        return torch.from_numpy(x.astype(np.float32))

    def __getitem__(self, idx):
        row = self.rows[idx]
        # "wavlm_all" (all layers) is exposed under the same "wavlm" key as the single-layer feature
        item = {("wavlm" if s.startswith("wavlm") else s): self._load(s, row["utt_id"]) for s in self.streams}
        item["label"] = row["label_id"]
        return item


def pad_stream(seqs: List[torch.Tensor]):
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    out = seqs[0].new_zeros(len(seqs), int(lengths.max()), seqs[0].shape[1])
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out, lengths


def collate_dual(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    out = {"label": torch.tensor([b["label"] for b in batch], dtype=torch.long)}
    for stream in ("wavlm", "fbank"):
        if stream not in batch[0]:
            continue
        if batch[0][stream].dim() == 3:  # all-layer WavLM [L, M, D]: keep unpadded, LayerMix pads
            out[stream] = [b[stream] for b in batch]
            out[stream + "_len"] = torch.tensor([b[stream].shape[1] for b in batch], dtype=torch.long)
        else:
            out[stream], out[stream + "_len"] = pad_stream([b[stream] for b in batch])
    return out
