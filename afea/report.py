"""Paper-style "reproduction vs. paper" ablation tables (Markdown and LaTeX).

`stats` maps a config name to {"WA": (mean, std), "UAR": ..., "P": ..., "F1": ..., "n": seeds}, in %.
"""

from typing import Dict, List, Tuple

from .suite import PAPER

METRICS = ("WA", "UAR", "P", "F1")

# (config, Markdown label, LaTeX label), grouped like Table 4 / Table 6 of the paper
PAPER_ROWS = [
    [("fbank", "Fbank", "Fbank"), ("wavlm", "WavLM", "WavLM")],
    [("wo_align", "w/o L_ali", r"w/o $L_{ali}$")],
    [("wo_afea", "w/o AFEA", "w/o AFEA")] + [(f"afea{i}", f"AFEA-{i}", f"AFEA-{i}") for i in range(1, 5)],
    [("wo_con", "w/o L_con", r"w/o $L_{con}$"), ("con_layer1", "w/o L_con1", r"w/o $L_{con1}$"),
     ("con_layers12", "w/o L_con2", r"w/o $L_{con2}$"), ("afea_net", "w/o L_con3", r"w/o $L_{con3}$")],
    [("afea_net", "AFEA-Net", "AFEA-Net")],
]

# protocol name -> (config prefix, description used in the table note)
PROTOCOLS = {
    "unbiased": ("", "epoch selected on a validation split of the training sessions"),
    "t": ("t_", "epoch with the best UAR on the held-out fold (valid = test)"),
    "tw": ("tw_", "epoch with the best WA on the held-out fold (valid = test)"),
}


def paper_table(stats: Dict[str, Dict], dataset: str, protocol: str) -> Tuple[List[str], str]:
    """Return (markdown lines, LaTeX string). Missing rows are shown as '–'."""
    if not stats:
        return [], ""
    _, desc = PROTOCOLS[protocol]
    paper = PAPER.get(dataset, {})
    best_r = {k: max(v[k][0] for v in stats.values()) for k in METRICS}
    best_p = {k: max(paper[n][i] * 100 for g in PAPER_ROWS for n, _, _ in g if n in paper)
              for i, k in enumerate(METRICS)} if paper else {}
    seeds = sorted({v["n"] for v in stats.values()})
    seed_txt = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}-{seeds[-1]}"
    ds_name = "IEMOCAP" if dataset == "iemocap" else "RAVDESS"
    split = "leave-one-session-out" if dataset == "iemocap" else "actor-disjoint"
    note = (f"Reproduction: mean±std over {seed_txt} seeds, {split} 5-fold CV, final WavLM layer, {desc}. "
            "Bold: best in each column.")

    md = [f"### Ablation experiments on {ds_name}: reproduction vs. paper (%), protocol `{protocol}`", "",
          "| | Repro WA | Repro UAR | Repro P | Repro F1 | Paper WA | Paper UAR | Paper P | Paper F1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Ablation experiments on the " + ds_name + r" dataset: reproduction vs.\ paper (\%).}",
           r"\begin{tabular}{lcccccccc}", r"\toprule",
           r" & \multicolumn{4}{c}{\textbf{Reproduction}} & \multicolumn{4}{c}{\textbf{Paper}} \\",
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
           r" & \textbf{WA} & \textbf{UAR} & \textbf{P} & \textbf{F1}"
           r" & \textbf{WA} & \textbf{UAR} & \textbf{P} & \textbf{F1} \\",
           r"\midrule"]
    for gi, group in enumerate(PAPER_ROWS):
        for name, md_label, tex_label in group:
            st, pp = stats.get(name), paper.get(name)
            md_cells, tex_cells = [], []
            for k in METRICS:
                if st:
                    m, sd = st[k]
                    bold = abs(m - best_r[k]) < 1e-9
                    md_cells.append(f"**{m:.1f}** ± {sd:.1f}" if bold else f"{m:.1f} ± {sd:.1f}")
                    val = r"\textbf{" + f"{m:.1f}" + "}" if bold else f"{m:.1f}"
                    tex_cells.append(val + r"{\scriptsize$\pm$" + f"{sd:.1f}" + "}")
                else:
                    md_cells.append("–")
                    tex_cells.append("--")
            for i, k in enumerate(METRICS):
                if pp:
                    v = pp[i] * 100
                    bold = abs(v - best_p[k]) < 1e-9
                    md_cells.append(f"**{v:.1f}**" if bold else f"{v:.1f}")
                    tex_cells.append(r"\textbf{" + f"{v:.1f}" + "}" if bold else f"{v:.1f}")
                else:
                    md_cells.append("–")
                    tex_cells.append("--")
            md.append(f"| {md_label} | " + " | ".join(md_cells) + " |")
            tex.append(tex_label + " & " + " & ".join(tex_cells) + r" \\")
        tex.append(r"\midrule" if gi < len(PAPER_ROWS) - 1 else r"\bottomrule")
    tex += [r"\end{tabular}", "", r"\footnotesize{" + note.replace("±", r"$\pm$") + "}", r"\end{table}"]
    md += ["", note, ""]
    return md, "\n".join(tex)


def stats_from_runs(runs_by_cfg: Dict[str, Dict[int, Dict[str, float]]], prefix: str) -> Dict[str, Dict]:
    """runs_by_cfg: {config: {seed: {"WA": fraction, ...}}} -> stats for paper_table (in %)."""
    import numpy as np
    out = {}
    for group in PAPER_ROWS:
        for name, _, _ in group:
            runs = runs_by_cfg.get(prefix + name, {})
            if not runs:
                continue
            st = {}
            for k in METRICS:
                vals = [r[k] * 100 for r in runs.values() if k in r]
                st[k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)
            st["n"] = len(runs)
            out[name] = st
    return out
