"""Summarise the diagnostic metrics of every run that logged them -> runs/diagnostics.md.

    python scripts/diagnose.py

Values are averaged over folds and seeds, taken at the epoch that was selected for testing.
  Stream usage   test WA drop when the WavLM / Fbank pooled vector is shuffled across the batch
                 (≈0 means the model ignores that stream)
  Fitting        train accuracy vs validation WA at the selected epoch, selected epoch
  SEAL           mean positive / negative pair distance, share of negatives inside the margin,
                 share of non-negative pooled features
  AFEA           ISE softmax share of WavLM per layer, ISA sigmoid gates of the last layer
  Gradients      gradient norm per module (enc_wav, enc_fil, afea, classifier, layer_mix)
  Layers         top WavLM layers by learned weight (weighted-sum runs)
"""
import glob
import json
import os
from collections import defaultdict

import _path  # noqa: F401
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect(cfg_dir):
    recs = []
    for p in sorted(glob.glob(os.path.join(cfg_dir, "seed*", "fold*_seed*.json"))):
        r = json.load(open(p))
        if "test_probe" not in r:
            continue
        sel = next((h for h in r["history"] if h["epoch"] == r["selected_epoch"]), r["history"][-1])
        recs.append((r, sel))
    return recs


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def f(v, pct=False, signed=False):
    if v is None:
        return "–"
    v = v * 100 if pct else v
    return f"{v:+.1f}" if signed else (f"{v:.1f}" if pct else f"{v:.3f}")


def main():
    out = ["# Diagnostics", "", __doc__.split("\n\n", 1)[1].strip().replace("\n", "  \n"), ""]
    for ds in ("iemocap", "ravdess"):
        rows = {}
        for cfg_dir in sorted(glob.glob(os.path.join(ROOT, "runs", ds, "*"))):
            recs = collect(cfg_dir)
            if recs:
                rows[os.path.basename(cfg_dir)] = recs
        if not rows:
            continue
        out += [f"## {ds.upper()}", ""]

        out += ["### Stream usage and fitting", "",
                "| config | runs | test WA | Δ shuffle WavLM | Δ shuffle Fbank | train acc | val WA | sel. epoch |",
                "|---|---|---|---|---|---|---|---|"]
        for name, recs in rows.items():
            wa = mean([r["test"]["WA"] for r, _ in recs])
            sw = mean([r["test_probe"].get("WA_shuf_wav") for r, _ in recs])
            sf = mean([r["test_probe"].get("WA_shuf_fil") for r, _ in recs])
            out.append(f"| {name} | {len(recs)} | {f(wa, True)} | {f(sw - wa if sw is not None else None, True, True)} | "
                       f"{f(sf - wa if sf is not None else None, True, True)} | "
                       f"{f(mean([s.get('diag/train_acc') for _, s in recs]), True)} | "
                       f"{f(mean([s.get('val_WA') for _, s in recs]), True)} | "
                       f"{mean([r['selected_epoch'] for r, _ in recs]):.1f} |")

        out += ["", "### SEAL and pooled features (dual-stream runs)", "",
                "| config | seal pos dist | seal neg dist | neg in margin | ali loss | pooled ≥ 0 | ‖S_wav‖ | ‖S_fil‖ |",
                "|---|---|---|---|---|---|---|---|"]
        for name, recs in rows.items():
            if not any("diag/seal_pos_dist" in s for _, s in recs):
                continue
            g = lambda k: mean([s.get(k) for _, s in recs])  # noqa: E731
            out.append(f"| {name} | {f(g('diag/seal_pos_dist'))} | {f(g('diag/seal_neg_dist'))} | "
                       f"{f(g('diag/seal_neg_in_margin'))} | {f(g('ali'))} | {f(g('diag/s_nonneg_frac'))} | "
                       f"{f(g('diag/s_wav_norm'))} | {f(g('diag/s_fil_norm'))} |")

        out += ["", "### AFEA internals and gradients", "",
                "| config | ISE A_wav L1/L2/L3 | ISA W_wav / W_fil (last) | con intra / inter (last) | "
                "grad enc_wav | grad enc_fil | grad afea | grad classifier |",
                "|---|---|---|---|---|---|---|---|"]
        for name, recs in rows.items():
            g = lambda k: mean([s.get(k) for _, s in recs])  # noqa: E731
            layers = sorted({int(k.split("_L")[1]) for _, s in recs for k in s if k.startswith("diag/ise_a_wav_L")})
            ise = " / ".join(f(g(f"diag/ise_a_wav_L{l}")) for l in layers) or "–"
            last = layers[-1] if layers else None
            isa = f"{f(g(f'diag/isa_w_wav_L{last}'))} / {f(g(f'diag/isa_w_fil_L{last}'))}" if last else "–"
            con = f"{f(g(f'diag/con_intra_L{last}'))} / {f(g(f'diag/con_inter_L{last}'))}" if last else "–"
            enc_w = g("diag/grad_enc_wav") if g("diag/grad_enc_wav") is not None else g("diag/grad_enc")
            out.append(f"| {name} | {ise} | {isa} | {con} | {f(enc_w)} | {f(g('diag/grad_enc_fil'))} | "
                       f"{f(g('diag/grad_afea'))} | {f(g('diag/grad_classifier'))} |")

        lw_rows = [(n, [r["layer_weights"] for r, _ in recs if r.get("layer_weights")]) for n, recs in rows.items()]
        lw_rows = [(n, w) for n, w in lw_rows if w]
        if lw_rows:
            out += ["", "### Learned WavLM layer weights (0 = CNN output, 24 = last layer)", "",
                    "| config | top-5 layers (weight) | weight on last layer | uniform |", "|---|---|---|---|"]
            for name, ws in lw_rows:
                w = np.mean(np.array(ws), axis=0)
                top = np.argsort(-w)[:5]
                out.append(f"| {name} | {', '.join(f'{i} ({w[i]:.3f})' for i in top)} | {w[-1]:.3f} | {1 / len(w):.3f} |")
        out.append("")

    text = "\n".join(out)
    print(text)
    os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "diagnostics.md"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
