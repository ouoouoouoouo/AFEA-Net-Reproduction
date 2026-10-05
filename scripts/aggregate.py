"""Collect runs/<dataset>/<config>/seed*/summary.json into Markdown tables.

    python scripts/aggregate.py            # prints and writes runs/results.md

For each config: mean ± std over seeds of the 5-fold mean, next to the paper's number.
"""
import glob
import json
import os

import _path  # noqa: F401
import numpy as np

from afea.suite import (PAPER, PAPER_ROW_NAMES, ablation_configs, diagnosis_configs, protocol_configs,
                        sweep_configs, testsel_configs, variant_configs)

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


def diagnosis_table(ds):
    lines = ["| config | args | seeds | WA | UAR | F1 | reference | ref WA (same seeds) | ΔWA |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, args, ref in diagnosis_configs(ds):
        runs = load(ds, name)
        if not runs:
            lines.append(f"| {name} | `{' '.join(args)}` | 0 | – | – | – | {ref} | – | – |")
            continue
        ref_runs = {s: v for s, v in load(ds, ref).items() if s in runs}
        if ref_runs:
            rw = np.mean([v["WA"] for v in ref_runs.values()]) * 100
            ow = np.mean([runs[s]["WA"] for s in ref_runs]) * 100
            ref_txt, delta = f"{rw:.1f}", f"{ow - rw:+.1f}"
        else:
            ref_txt, delta = "–", "–"
        lines.append(f"| {name} | `{' '.join(args)}` | {len(runs)} | {fmt(runs, 'WA')} | {fmt(runs, 'UAR')} | "
                     f"{fmt(runs, 'F1')} | {ref} | {ref_txt} | {delta} |")
    return lines


def protocol_table(ds):
    lines = ["| config | seeds | WA | UAR | F1 | oracle WA | oracle UAR | paper WA / UAR |",
             "|---|---|---|---|---|---|---|---|"]
    for name, _, ref in protocol_configs(ds):
        paths = sorted(glob.glob(os.path.join(ROOT, "runs", ds, name, "seed*", "summary.json")))
        if not paths:
            lines.append(f"| {name} | 0 | – | – | – | – | – | |")
            continue
        sums = [json.load(open(p)) for p in paths]
        runs = {str(i): s["fold_mean"] for i, s in enumerate(sums)}
        okey = "oracle_fold_mean (test-selected epoch, optimistic)"
        orc = [s[okey] for s in sums if okey in s]
        o_wa = f"{np.mean([o['WA'] for o in orc]) * 100:.1f}" if orc else "–"
        o_uar = f"{np.mean([o['UAR'] for o in orc]) * 100:.1f}" if orc else "–"
        base = name.split("_", 1)[1]
        p = PAPER.get(ds, {}).get(base)
        lines.append(f"| {name} | {len(runs)} | {fmt(runs, 'WA')} | {fmt(runs, 'UAR')} | {fmt(runs, 'F1')} | "
                     f"{o_wa} | {o_uar} | {f'{p[0]*100:.1f} / {p[1]*100:.1f}' if p else ''} |")
    return lines


def testsel_table(ds):
    lines = ["| config | seeds | WA | UAR | F1 | same config, unbiased selection (WA, same seeds) | gain from test selection |",
             "|---|---|---|---|---|---|---|"]
    for name, args, ref in testsel_configs(ds):
        runs = load(ds, name)
        if not runs:
            continue
        ref_runs = {s: v for s, v in load(ds, ref).items() if s in runs}
        if ref_runs:
            rw = np.mean([v["WA"] for v in ref_runs.values()]) * 100
            ow = np.mean([runs[s]["WA"] for s in ref_runs]) * 100
            ref_txt, delta = f"{rw:.1f} ({ref})", f"{ow - rw:+.1f}"
        else:
            ref_txt, delta = f"– ({ref})", "–"
        lines.append(f"| {name} | {len(runs)} | {fmt(runs, 'WA')} | {fmt(runs, 'UAR')} | {fmt(runs, 'F1')} | "
                     f"{ref_txt} | {delta} |")
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
        if any(load(ds, n) for n, _, _ in diagnosis_configs(ds)):
            out += ["### Diagnosis experiments (Δ = WA change vs the reference config, same seeds)", ""]
            out += diagnosis_table(ds) + [""]
        out += ["### Protocol checks (final WavLM layer, epoch selected by validation UAR)", "",
                "`p_*` = original features, `pn_*` = official waveform layer-norm before WavLM. "
                "Oracle = epoch chosen on the test fold (optimistic, for reference only).", ""]
        out += protocol_table(ds) + [""]
        if any(load(ds, n) for n, _, _ in testsel_configs(ds)):
            out += ["### 'valid = test' protocol (train on all training folds, epoch with best UAR on the "
                    "held-out fold; optimistic, common in IEMOCAP papers)", ""]
            out += testsel_table(ds) + [""]
    text = "\n".join(out)
    print(text)
    with open(os.path.join(ROOT, "runs", "results.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
