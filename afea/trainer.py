"""Cross-validation training loop for AFEA-Net and its ablations."""

import copy
import json
import os
import random
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import DualStreamFeatureDataset, collate_dual
from .losses import continuity_loss, seal_loss
from .metrics import compute_metrics, confusion
from .model import AFEANet, SingleStreamNet

FEAT_DIMS = {"wavlm": 1024, "fbank": 40}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_fold(rows: List[Dict], test_fold: int, val_ratio: float, seed: int):
    """Test = one fold (IEMOCAP session / RAVDESS actor group); train = the other folds.
    Optionally carve a stratified validation subset out of the training folds."""
    test = [r for r in rows if r["fold"] == test_fold]
    train = [r for r in rows if r["fold"] != test_fold]
    val: List[Dict] = []
    if val_ratio > 0:
        from sklearn.model_selection import train_test_split
        train, val = train_test_split(train, test_size=val_ratio, random_state=seed,
                                      stratify=[r["label_id"] for r in train])
    return train, val, test


def build_model(cfg: Dict) -> nn.Module:
    if cfg["model"] == "afea":
        return AFEANet(cfg["num_classes"], FEAT_DIMS["wavlm"], FEAT_DIMS["fbank"],
                       lstm_hidden=cfg["lstm_hidden"], lstm_dropout=cfg["lstm_dropout"],
                       num_afea_layers=cfg["afea_layers"], isa_hidden=cfg["isa_hidden"],
                       fcn_hidden=cfg["fcn_hidden"], fcn_dropout=cfg["fcn_dropout"])
    return SingleStreamNet(cfg["num_classes"], FEAT_DIMS[cfg["model"]], lstm_hidden=cfg["lstm_hidden"],
                           lstm_dropout=cfg["lstm_dropout"], fcn_hidden=cfg["fcn_hidden"],
                           fcn_dropout=cfg["fcn_dropout"])


def streams_for(cfg: Dict) -> Sequence[str]:
    return ("wavlm", "fbank") if cfg["model"] == "afea" else (cfg["model"],)


def forward_and_loss(model: nn.Module, batch: Dict, cfg: Dict, device) -> Dict[str, torch.Tensor]:
    y = batch["label"].to(device)
    if cfg["model"] == "afea":
        out = model(batch["wavlm"].to(device), batch["wavlm_len"],
                    batch["fbank"].to(device), batch["fbank_len"])
    else:
        s = cfg["model"]
        out = model(batch[s].to(device), batch[s + "_len"])
    losses = {"ce": F.cross_entropy(out["logits"], y)}
    if cfg["model"] == "afea":
        if cfg["use_align"]:
            losses["ali"] = seal_loss(out["s_wav"], out["s_fil"], y, cfg["margin"])
        if cfg["use_con"] and out["f_wav"]:
            losses["con"] = continuity_loss(out["f_wav"], out["f_fil"], out["s_wav"], out["s_fil"],
                                            cfg["con_weights"])
    losses["total"] = sum(losses.values())     # Eq. 16: L = L_ce + L_ali + L_con
    losses["logits"] = out["logits"]
    return losses


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        out = forward_and_loss(model, batch, cfg, device)
        ps.append(out["logits"].argmax(-1).cpu())
        ys.append(batch["label"])
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return compute_metrics(y, p, cfg["num_classes"]), y, p


def make_loader(rows, cfg, shuffle, seed=0):
    ds = DualStreamFeatureDataset(rows, cfg["feat_root"], streams_for(cfg),
                                  max_frames=cfg.get("max_frames"), cache=cfg.get("cache", False))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, collate_fn=collate_dual,
                      num_workers=cfg.get("num_workers", 0), generator=g,
                      drop_last=False)


def train_fold(rows: List[Dict], test_fold: int, cfg: Dict, seed: int, device, log=print) -> Dict:
    set_seed(seed)
    train_rows, val_rows, test_rows = split_fold(rows, test_fold, cfg["val_ratio"], seed)
    train_loader = make_loader(train_rows, cfg, shuffle=True, seed=seed)
    val_loader = make_loader(val_rows, cfg, shuffle=False) if val_rows else None
    test_loader = make_loader(test_rows, cfg, shuffle=False)

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_score, best_state, best_epoch = -1.0, None, -1
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, agg, n = time.time(), {}, 0
        for batch in train_loader:
            losses = forward_and_loss(model, batch, cfg, device)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            bs = batch["label"].shape[0]
            n += bs
            for k, v in losses.items():
                if k != "logits":
                    agg[k] = agg.get(k, 0.0) + float(v.detach()) * bs
        rec = {"epoch": epoch, **{k: v / n for k, v in agg.items()}, "time": time.time() - t0}
        if val_loader is not None:
            vm, _, _ = evaluate(model, val_loader, cfg, device)
            rec.update({"val_" + k: v for k, v in vm.items()})
            score = vm["UAR"] + vm["WA"]
            if cfg["select"] == "val" and score > best_score:
                best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        history.append(rec)
        log(f"  fold {test_fold} seed {seed} ep {epoch:3d} " +
            " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k not in ("epoch",)))
    if cfg["select"] == "val" and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = cfg["epochs"]
    metrics, y, p = evaluate(model, test_loader, cfg, device)
    log(f"  fold {test_fold} seed {seed} TEST (epoch {best_epoch}): " +
        " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return {"fold": test_fold, "seed": seed, "selected_epoch": best_epoch, "test": metrics,
            "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
            "confusion": confusion(y, p, cfg["num_classes"]).tolist(),
            "y_true": y.tolist(), "y_pred": p.tolist(), "history": history}


def run_cv(rows: List[Dict], cfg: Dict, folds: Optional[Sequence[int]] = None,
           seeds: Sequence[int] = (42,), device="cpu", out_dir: Optional[str] = None) -> Dict:
    folds = list(folds) if folds else sorted({r["fold"] for r in rows})
    results = []
    logf = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        logf = open(os.path.join(out_dir, "train.log"), "a")

    def log(msg):
        print(msg, flush=True)
        if logf:
            logf.write(msg + "\n")
            logf.flush()

    for seed in seeds:
        for fold in folds:
            res = train_fold(rows, fold, cfg, seed, device, log)
            results.append(res)
            if out_dir:
                with open(os.path.join(out_dir, f"fold{fold}_seed{seed}.json"), "w") as f:
                    json.dump(res, f)
    summary = summarize(results, cfg["num_classes"])
    log("SUMMARY (mean over folds): " + json.dumps(summary["fold_mean"]))
    if out_dir:
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logf.close()
    return summary


def summarize(results: List[Dict], num_classes: int) -> Dict:
    keys = ["WA", "UAR", "P", "F1"]
    per = {k: [r["test"][k] for r in results] for k in keys}
    y = [t for r in results for t in r["y_true"]]
    p = [t for r in results for t in r["y_pred"]]
    seeds = sorted({r["seed"] for r in results})
    pooled = compute_metrics(y, p, num_classes)
    if len(seeds) > 1:  # pooled over folds within each seed, then averaged over seeds
        per_seed = [compute_metrics([t for r in results if r["seed"] == s for t in r["y_true"]],
                                    [t for r in results if r["seed"] == s for t in r["y_pred"]],
                                    num_classes) for s in seeds]
        pooled = {k: float(np.mean([m[k] for m in per_seed])) for k in keys}
    return {
        "fold_mean": {k: float(np.mean(v)) for k, v in per.items()},
        "fold_std": {k: float(np.std(v)) for k, v in per.items()},
        "pooled": pooled,
        "per_run": [{"fold": r["fold"], "seed": r["seed"], "epoch": r["selected_epoch"], **r["test"]}
                    for r in results],
    }
