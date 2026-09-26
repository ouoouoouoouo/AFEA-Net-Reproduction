"""Dataset over pre-extracted features and a collate function that pads each
stream independently (the WavLM and Fbank sequences keep their own lengths)."""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class DualStreamFeatureDataset(Dataset):
    def __init__(self, rows: List[Dict], feat_root: str, streams=("wavlm", "fbank"),
                 max_frames: Optional[Dict[str, int]] = None, cache: bool = False):
        self.rows = rows
        self.feat_root = feat_root
        self.streams = tuple(streams)
        self.max_frames = max_frames or {}
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.rows)

    def _load(self, stream: str, utt_id: str) -> torch.Tensor:
        key = (stream, utt_id)
        if self._cache is not None and key in self._cache:
            return self._cache[key]
        x = torch.from_numpy(np.load(os.path.join(self.feat_root, stream, utt_id + ".npy")).astype(np.float32))
        limit = self.max_frames.get(stream)
        if limit:
            x = x[:limit]
        if self._cache is not None:
            self._cache[key] = x
        return x

    def __getitem__(self, idx):
        row = self.rows[idx]
        item = {s: self._load(s, row["utt_id"]) for s in self.streams}
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
        if stream in batch[0]:
            out[stream], out[stream + "_len"] = pad_stream([b[stream] for b in batch])
    return out
