"""Rebuild result tables from Weights & Biases when the training server is unreachable.

    pip install wandb numpy
    wandb login
    python scripts/wandb_export.py --project <entity>/afea-net            # -> wandb_results.md
    python scripts/wandb_export.py --project <entity>/afea-net --runs      # also list every run

Run names follow train.py / run_suite.py: <dataset>_<config>_s<seed>. For each config the
script reports mean ± std over seeds of the 5-fold mean (summary key fold_mean/*). It also reports:
  * oracle  = for every fold, the epoch with the best test UAR (history key
              fold<k>_s<seed>/mon_test_UAR), averaged over folds; only for runs that logged it
  * the "valid = test" (t_*) runs next to the same config with unbiased selection.
Crashed or duplicated runs (same name) are ignored: the finished run with fold_mean wins.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from afea.suite import (PAPER, ablation_configs, diagnosis_configs, protocol_configs,  # noqa: E402
                        testsel_configs, variant_configs)

NAME_RE = re.compile(r"^(iemocap|ravdess)_(.+)_s(\d+)$")
KEYS = ("WA", "UAR", "F1")


def oracle_from_history(run, folds=range(1, 6)):
    """Mean over folds of (WA, UAR) at the epoch with the best monitored test UAR."""
    seed = NAME_RE.match(run.name).group(3)
    was, uars = [], []
    for k in folds:
        prefix = f"fold{k}_s{seed}"
        keys = [f"{prefix}/mon_test_UAR", f"{prefix}/mon_test_WA"]
        # one sampled request per fold (50 epochs << samples), much faster than scan_history
        rows = [r for r in run.history(keys=keys, samples=10000, pandas=False) if all(k in r for k in keys)]
        if not rows:
            return None
        best = max(rows, key=lambda r: r[f"{prefix}/mon_test_UAR"])
        uars.append(best[f"{prefix}/mon_test_UAR"])
        was.append(best[f"{prefix}/mon_test_WA"])
    return {"WA": float(np.mean(was)), "UAR": float(np.mean(uars))}


def collect(runs, want_oracle):
    """-> {dataset: {config: {seed: {"WA":..,"UAR":..,"F1":.., "oracle": {...}|None}}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    todo = [r for r in runs if NAME_RE.match(r.name or "") and want_oracle(NAME_RE.match(r.name).group(2))]
    done = 0
    for run in runs:
        m = NAME_RE.match(run.name or "")
        if not m:
            continue
        summ = {k: run.summary[k] for k in run.summary.keys()}
        if "fold_mean/WA" not in summ:  # crashed / unfinished run
            continue
        ds, cfg, seed = m.group(1), m.group(2), int(m.group(3))
        rec = {k: float(summ[f"fold_mean/{k}"]) for k in KEYS if f"fold_mean/{k}" in summ}
        rec["oracle"] = None
        if want_oracle(cfg):
            done += 1
            print(f"  oracle {done}/{len(todo)}: {run.name}", flush=True)
            rec["oracle"] = oracle_from_history(run)
        out[ds][cfg][seed] = rec
    return out


def fmt(vals):
    v = np.array(vals) * 100
    return f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def row(cfg, seeds, extra=""):
    return (f"| {cfg} | {len(seeds)} | " + " | ".join(fmt([s[k] for s in seeds.values() if k in s]) for k in KEYS)
            + extra + " |")


def tables(data):
    lines = []
    for ds in ("iemocap", "ravdess"):
        if ds not in data:
            continue
        cfgs = data[ds]
        lines += [f"## {ds.upper()}", ""]

        groups = [
            ("Ablations (Table 4/6), unbiased selection", [n for n, _ in ablation_configs(ds)], True),
            ("Assumption variants", [n for n, _ in variant_configs(ds)], False),
            ("Diagnosis experiments", [n for n, _, _ in diagnosis_configs(ds)], False),
        ]
        for title, names, paper in groups:
            present = [n for n in names if n in cfgs]
            if not present:
                continue
            head = "| config | seeds | WA | UAR | F1 |" + (" paper WA / UAR |" if paper else "")
            lines += [f"### {title}", "", head, "|" + "---|" * (head.count("|") - 1)]
            for n in present:
                p = PAPER.get(ds, {}).get(n)
                extra = f" | {p[0]*100:.1f} / {p[1]*100:.1f}" if paper and p else (" | " if paper else "")
                lines.append(row(n, cfgs[n], extra))
            lines.append("")

        present = [n for n, _, _ in protocol_configs(ds) if n in cfgs]
        if present:
            lines += ["### Protocol checks (final WavLM layer, epoch picked by validation UAR; "
                      "p_ = original features, pn_ = official waveform layer-norm)", "",
                      "| config | seeds | WA | UAR | F1 | oracle WA | oracle UAR |", "|---|---|---|---|---|---|---|"]
            for n in present:
                orc = [s["oracle"] for s in cfgs[n].values() if s.get("oracle")]
                o = (f" | {np.mean([x['WA'] for x in orc])*100:.1f} | {np.mean([x['UAR'] for x in orc])*100:.1f}"
                     if orc else " | – | –")
                lines.append(row(n, cfgs[n], o))
            lines.append("")

        present = [(n, ref) for n, _, ref in testsel_configs(ds) if n in cfgs]
        if present:
            lines += ["### 'valid = test' (train on all training folds, epoch with best test UAR; optimistic)", "",
                      "| config | seeds | WA | UAR | F1 | unbiased same config (WA, same seeds) | gain | paper WA / UAR |",
                      "|---|---|---|---|---|---|---|---|"]
            for n, ref in present:
                runs = cfgs[n]
                ref_runs = {s: v for s, v in cfgs.get(ref, {}).items() if s in runs}
                if ref_runs:
                    rw = np.mean([v["WA"] for v in ref_runs.values()]) * 100
                    ow = np.mean([runs[s]["WA"] for s in ref_runs]) * 100
                    ref_txt, gain = f"{rw:.1f} ({ref})", f"{ow - rw:+.1f}"
                else:
                    ref_txt, gain = f"– ({ref})", "–"
                p = PAPER.get(ds, {}).get(ref)
                ptxt = f"{p[0]*100:.1f} / {p[1]*100:.1f}" if p else ""
                lines.append(row(n, runs, f" | {ref_txt} | {gain} | {ptxt}"))
            lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="<entity>/<project>, e.g. myteam/afea-net")
    ap.add_argument("--out", default="wandb_results.md")
    ap.add_argument("--runs", action="store_true", help="also list every run name and state")
    ap.add_argument("--no_oracle", action="store_true", help="skip history scans (faster)")
    args = ap.parse_args()

    import wandb
    runs = list(wandb.Api(timeout=60).runs(args.project, per_page=200))
    print(f"{len(runs)} runs in {args.project}")
    if args.runs:
        by_state = defaultdict(list)
        for r in runs:
            by_state[r.state].append(r.name)
        for state, names in sorted(by_state.items()):
            print(f"\n[{state}] {len(names)} runs")
            for n in sorted(names):
                print("  ", n)

    want_oracle = (lambda cfg: False) if args.no_oracle else (lambda cfg: cfg.startswith(("p_", "pn_")))
    data = collect(runs, want_oracle)
    text = "\n".join(["# Results exported from W&B", f"Project: `{args.project}`", ""] + tables(data))
    with open(args.out, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
