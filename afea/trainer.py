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
from .losses import continuity_loss, seal_loss, seal_stats
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


def wavlm_dir(cfg: Dict) -> str:
    return "wavlm_all" if cfg.get("wavlm_layers", "last") == "all" else "wavlm"


def build_model(cfg: Dict) -> nn.Module:
    n_layers = cfg.get("wavlm_num_layers", 0) if wavlm_dir(cfg) == "wavlm_all" else 0
    common = dict(lstm_hidden=cfg["lstm_hidden"], lstm_dropout=cfg["lstm_dropout"],
                  fcn_hidden=cfg["fcn_hidden"], fcn_dropout=cfg["fcn_dropout"],
                  dropout_pos=cfg.get("dropout_pos", "pre_pool"))
    if cfg["model"] == "afea":
        return AFEANet(cfg["num_classes"], FEAT_DIMS["wavlm"], FEAT_DIMS["fbank"],
                       num_afea_layers=cfg["afea_layers"], isa_hidden=cfg["isa_hidden"],
                       isa_mlp=cfg.get("isa_mlp", "shared"), wavlm_num_layers=n_layers, **common)
    return SingleStreamNet(cfg["num_classes"], FEAT_DIMS[cfg["model"]],
                           wavlm_num_layers=n_layers if cfg["model"] == "wavlm" else 0, **common)


def streams_for(cfg: Dict) -> Sequence[str]:
    if cfg["model"] == "afea":
        return (wavlm_dir(cfg), "fbank")
    return (wavlm_dir(cfg),) if cfg["model"] == "wavlm" else ("fbank",)


def _to(x, device):
    return [t.to(device, non_blocking=True) for t in x] if isinstance(x, list) else x.to(device)


def forward_and_loss(model: nn.Module, batch: Dict, cfg: Dict, device, shuffle=None,
                     generator=None) -> Dict[str, torch.Tensor]:
    y = batch["label"].to(device)
    if cfg["model"] == "afea":
        out = model(_to(batch["wavlm"], device), batch["wavlm_len"],
                    batch["fbank"].to(device), batch["fbank_len"], shuffle=shuffle, generator=generator)
    else:
        s = cfg["model"]
        out = model(_to(batch[s], device), batch[s + "_len"])
    losses = {"ce": F.cross_entropy(out["logits"], y)}
    if cfg["model"] == "afea":
        if cfg["use_align"]:
            losses["ali"] = seal_loss(out["s_wav"], out["s_fil"], y, cfg["margin"],
                                      cfg.get("seal_norm", "l2"))
        if cfg["use_con"] and out["f_wav"]:
            losses["con"] = continuity_loss(out["f_wav"], out["f_fil"], out["s_wav"], out["s_fil"],
                                            cfg["con_weights"])
    losses["total"] = sum(losses.values())     # Eq. 16: L = L_ce + L_ali + L_con
    losses["logits"] = out["logits"]
    losses["_out"] = out
    return losses


# ----------------------------------------------------------------------------------------------
# Diagnostics. None of these consume the global RNG, so enabling them does not change training.

@torch.no_grad()
def batch_diagnostics(out: Dict, y: torch.Tensor, cfg: Dict) -> Dict[str, float]:
    d = {"train_acc": float((out["logits"].argmax(-1) == y).float().mean())}
    if cfg["model"] != "afea":
        return d
    sw, sf = out["s_wav"], out["s_fil"]
    d.update({"s_wav_norm": float(sw.norm(dim=-1).mean()), "s_fil_norm": float(sf.norm(dim=-1).mean()),
              "s_nonneg_frac": float(((sw >= 0).float().mean() + (sf >= 0).float().mean()) / 2)})
    d.update(seal_stats(sw, sf, y, cfg["margin"], cfg.get("seal_norm", "l2")))
    for l, ex in enumerate(out["afea"], 1):
        d[f"ise_a_wav_L{l}"] = float(ex["ise_weights"][:, 0].mean())      # softmax share of WavLM
        d[f"isa_w_wav_L{l}"] = float(ex["isa_w_wav"].mean())              # sigmoid gates
        d[f"isa_w_fil_L{l}"] = float(ex["isa_w_fil"].mean())
    for l, (fw, ff) in enumerate(zip(out["f_wav"], out["f_fil"]), 1):
        d[f"con_intra_L{l}"] = float(F.mse_loss(fw, sw) + F.mse_loss(ff, sf))
        d[f"con_inter_L{l}"] = float(F.mse_loss(fw, ff))
    return d


def grad_norms(model: nn.Module) -> Dict[str, float]:
    groups = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = name.split(".")[0]
            groups[g] = groups.get(g, 0.0) + float(p.grad.detach().pow(2).sum())
    return {f"grad_{g}": v ** 0.5 for g, v in groups.items()}


@torch.no_grad()
def evaluate(model, loader, cfg, device, probes: bool = False):
    """Metrics (+ mean CE). With `probes`, also accuracy when one stream's pooled vector is
    shuffled across the batch: a small drop means the model barely uses that stream."""
    model.eval()
    ys, ps, ce, n = [], [], 0.0, 0
    do_probe = probes and cfg["model"] == "afea"
    probe_preds = {"wav": [], "fil": []}
    gen = torch.Generator().manual_seed(1234)
    for batch in loader:
        out = forward_and_loss(model, batch, cfg, device)
        ps.append(out["logits"].argmax(-1).cpu())
        ys.append(batch["label"])
        ce += float(out["ce"]) * len(batch["label"])
        n += len(batch["label"])
        if do_probe:
            for stream in ("wav", "fil"):
                o = forward_and_loss(model, batch, cfg, device, shuffle=stream, generator=gen)
                probe_preds[stream].append(o["logits"].argmax(-1).cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    m = compute_metrics(y, p, cfg["num_classes"])
    m["CE"] = ce / max(n, 1)
    if do_probe:
        for stream in ("wav", "fil"):
            pm = compute_metrics(y, torch.cat(probe_preds[stream]).numpy(), cfg["num_classes"])
            m[f"WA_shuf_{stream}"] = pm["WA"]
    return m, y, p


def make_loader(rows, cfg, shuffle, seed=0):
    ds = DualStreamFeatureDataset(rows, cfg["feat_root"], streams_for(cfg),
                                  max_frames=cfg.get("max_frames"), cache=cfg.get("cache", False))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, collate_fn=collate_dual,
                      num_workers=cfg.get("num_workers", 0), generator=g,
                      drop_last=False)


def train_fold(rows: List[Dict], test_fold: int, cfg: Dict, seed: int, device, log=print,
               wandb_run=None) -> Dict:
    set_seed(seed)
    prefix = f"fold{test_fold}_s{seed}"
    if wandb_run is not None:
        wandb_run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/epoch")
    train_rows, val_rows, test_rows = split_fold(rows, test_fold, cfg["val_ratio"], seed)
    train_loader = make_loader(train_rows, cfg, shuffle=True, seed=seed)
    val_loader = make_loader(val_rows, cfg, shuffle=False) if val_rows else None
    test_loader = make_loader(test_rows, cfg, shuffle=False)

    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    best_score, best_state, best_epoch = -1.0, None, -1
    history = []
    diag = cfg.get("diag", True)
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, agg, n = time.time(), {}, 0
        for batch in train_loader:
            losses = forward_and_loss(model, batch, cfg, device)
            opt.zero_grad()
            losses["total"].backward()
            bs = batch["label"].shape[0]
            if diag:
                extra = batch_diagnostics(losses["_out"], batch["label"].to(device), cfg)
                extra.update(grad_norms(model))
                for k, v in extra.items():
                    agg["diag/" + k] = agg.get("diag/" + k, 0.0) + v * bs
            opt.step()
            n += bs
            for k, v in losses.items():
                if k not in ("logits", "_out"):
                    agg[k] = agg.get(k, 0.0) + float(v.detach()) * bs
        rec = {"epoch": epoch, **{k: v / n for k, v in agg.items()}, "time": time.time() - t0}
        if diag and getattr(model, "layer_mix", None) is not None:
            for i, w in enumerate(model.layer_mix.weights().tolist()):
                rec[f"diag/layer_w{i:02d}"] = w
        if val_loader is not None:
            vm, _, _ = evaluate(model, val_loader, cfg, device, probes=diag)
            rec.update({"val_" + k: v for k, v in vm.items()})
            score = vm["UAR"] + vm["WA"]
            if cfg["select"] == "val" and score > best_score:
                best_score, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch
        history.append(rec)
        if wandb_run is not None:
            wandb_run.log({f"{prefix}/{k}": v for k, v in rec.items()})
        log(f"  fold {test_fold} seed {seed} ep {epoch:3d} " +
            " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k != "epoch" and not k.startswith("diag/")))
    if cfg["select"] == "val" and best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = cfg["epochs"]
    full, y, p = evaluate(model, test_loader, cfg, device, probes=diag)
    metrics = {k: full[k] for k in ("WA", "UAR", "P", "F1")}
    probes = {k: v for k, v in full.items() if k not in metrics}
    layer_w = (model.layer_mix.weights().tolist() if getattr(model, "layer_mix", None) is not None else None)
    log(f"  fold {test_fold} seed {seed} TEST (epoch {best_epoch}): " +
        " ".join(f"{k}={v:.4f}" for k, v in full.items()))
    if wandb_run is not None:
        for k, v in full.items():
            wandb_run.summary[f"{prefix}/test_{k}"] = v
        wandb_run.summary[f"{prefix}/selected_epoch"] = best_epoch
        if layer_w is not None:
            wandb_run.summary[f"{prefix}/layer_weights"] = layer_w
    return {"fold": test_fold, "seed": seed, "selected_epoch": best_epoch, "test": metrics,
            "test_probe": probes, "layer_weights": layer_w,
            "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
            "confusion": confusion(y, p, cfg["num_classes"]).tolist(),
            "y_true": y.tolist(), "y_pred": p.tolist(), "history": history}


def run_cv(rows: List[Dict], cfg: Dict, folds: Optional[Sequence[int]] = None,
           seeds: Sequence[int] = (42,), device="cpu", out_dir: Optional[str] = None) -> Dict:
    folds = list(folds) if folds else sorted({r["fold"] for r in rows})
    results = []
    if wavlm_dir(cfg) == "wavlm_all" and cfg["model"] in ("afea", "wavlm"):
        probe = np.load(os.path.join(cfg["feat_root"], "wavlm_all", rows[0]["utt_id"] + ".npy"), mmap_mode="r")
        cfg["wavlm_num_layers"] = int(probe.shape[0])
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

    wandb_run = None
    if cfg.get("wandb"):
        try:
            import wandb
            wandb_run = wandb.init(project=cfg["wandb"], name=cfg.get("wandb_name"),
                                   group=cfg.get("wandb_group"), config=cfg)
        except Exception as e:  # never let logging kill a long unattended run
            log(f"WARNING: wandb disabled ({type(e).__name__}: {e})")
            wandb_run = None

    for seed in seeds:
        for fold in folds:
            fold_path = os.path.join(out_dir, f"fold{fold}_seed{seed}.json") if out_dir else None
            if fold_path and os.path.exists(fold_path) and cfg.get("resume", True):
                with open(fold_path) as f:
                    res = json.load(f)
                log(f"  fold {fold} seed {seed}: found {fold_path}, skipping (resume)")
            else:
                res = train_fold(rows, fold, cfg, seed, device, log, wandb_run)
                if fold_path:  # atomic write: a crash never leaves a half-written "done" file
                    with open(fold_path + ".tmp", "w") as f:
                        json.dump(res, f)
                    os.replace(fold_path + ".tmp", fold_path)
            results.append(res)
    summary = summarize(results, cfg["num_classes"])
    log("SUMMARY (mean over folds): " + json.dumps(summary["fold_mean"]))
    if wandb_run is not None:
        for part in ("fold_mean", "fold_std", "pooled"):
            for k, v in summary[part].items():
                wandb_run.summary[f"{part}/{k}"] = v
        cols = ["fold", "seed", "epoch", "WA", "UAR", "P", "F1"]
        import wandb
        wandb_run.log({"per_run": wandb.Table(
            columns=cols, data=[[r[c] for c in cols] for r in summary["per_run"]])})
        wandb_run.finish()
    if out_dir:
        with open(os.path.join(out_dir, "summary.json.tmp"), "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(os.path.join(out_dir, "summary.json.tmp"), os.path.join(out_dir, "summary.json"))
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
