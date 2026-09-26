"""Collect runs/<dataset>/<config>/seed*/summary.json into Markdown tables.

    python scripts/aggregate.py            # prints and writes runs/results.md

For each config: mean ± std over seeds of the 5-fold mean, next to the paper's number.
"""
import glob
import json
import os

import _path  # noqa: F401
import numpy as np

from afea.suite import PAPER, PAPER_ROW_NAMES, ablation_configs, sweep_configs, variant_configs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS = ("WA", "UAR", "P", "F1")


def load(ds, name):
    runs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "runs", ds, name, "seed*", "summary.json"))):
        seed = os.path.basename(os.path.dirname(p))[4:]
        runs[seed] = json.load(open(p))["fold_mean"]
    return runs


def fmt(runs, k):
    v = np.array([r[k] for r in runs.values()]) * 100
    return f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def table(ds, configs, with_paper):
    lines = []
    head = "| config | seeds | " + " | ".join(KEYS) + (" | paper WA / UAR / P / F1 |" if with_paper else " |")
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for name, args in configs:
        runs = load(ds, name)
        label = PAPER_ROW_NAMES.get(name, name) if with_paper else f"{name} (`{' '.join(args)}`)"
        if not runs:
            lines.append(f"| {label} | 0 | " + " | ".join("–" for _ in KEYS) + (" | |" if with_paper else " |"))
            continue
        row = f"| {label} | {len(runs)} | " + " | ".join(fmt(runs, k) for k in KEYS)
        if with_paper:
            p = PAPER[ds].get(name)
            row += " | " + (" / ".join(f"{x * 100:.1f}" for x in p) if p else "") + " |"
        else:
            row += " |"
        lines.append(row)
    return lines


def main():
    out = ["# Reproduction results", "",
           "Values are % (mean ± std over seeds of the 5-fold mean). Protocol: see ASSUMPTIONS.md.", ""]
    for ds in ("iemocap", "ravdess"):
        if not glob.glob(os.path.join(ROOT, "runs", ds, "*", "seed*", "summary.json")):
            continue
        out += [f"## {ds.upper()}", "", "### Ablations (Table {})".format(4 if ds == "iemocap" else 6), ""]
        out += table(ds, ablation_configs(ds), True) + [""]
        out += ["### Assumption variants of the full model", ""] + table(ds, variant_configs(ds), False) + [""]
        out += ["### Hyper-parameter sweeps (Fig. 4-5)", ""] + table(ds, sweep_configs(ds), False) + [""]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(ROOT, "runs", "results.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
