"""Experiment definitions shared by scripts/run_suite.py and scripts/aggregate.py."""

from typing import Dict, List, Tuple

from .config import DATASET_PRESETS


def ablation_configs(dataset: str) -> List[Tuple[str, List[str]]]:
    """Rows of Table 4 (IEMOCAP) / Table 6 (RAVDESS). See ASSUMPTIONS.md section 7."""
    a, b, _ = DATASET_PRESETS[dataset]["con_weights"]
    return [
        ("fbank", ["--model", "fbank"]),
        ("wavlm", ["--model", "wavlm"]),
        ("wo_align", ["--no_align"]),
        ("wo_afea", ["--afea_layers", "0"]),
        ("afea1", ["--afea_layers", "1", "--no_con"]),
        ("afea2", ["--afea_layers", "2", "--no_con"]),
        ("afea3", ["--afea_layers", "3", "--no_con"]),
        ("afea4", ["--afea_layers", "4", "--no_con"]),
        ("wo_con", ["--no_con"]),
        ("con_layer1", ["--con_weights", str(a), "0", "0"]),
        ("con_layers12", ["--con_weights", str(a), str(b), "0"]),
        ("afea_net", []),
    ]


def variant_configs(dataset: str) -> List[Tuple[str, List[str]]]:
    """Robustness checks of the full model against the most influential assumptions."""
    return [
        ("afea_net_D1024", ["--lstm_hidden", "512"]),              # "512 hidden units" = per direction
        ("afea_net_lastepoch", ["--select", "last", "--val_ratio", "0"]),  # no validation split
    ]


def sweep_configs(dataset: str) -> List[Tuple[str, List[str]]]:
    """Hyper-parameter sweeps of Fig. 4 (margin) and Fig. 5 (alpha, beta, gamma, one at a time)."""
    p = DATASET_PRESETS[dataset]
    out = [(f"sweep_margin{m}", ["--margin", str(m)]) for m in (0.5, 1.0, 1.5, 2.0) if m != p["margin"]]
    grid = [round(0.1 * k, 1) for k in range(1, 11)]
    for idx, name in enumerate(("alpha", "beta", "gamma")):
        for v in grid:
            if v == p["con_weights"][idx]:
                continue
            w = list(p["con_weights"])
            w[idx] = v
            out.append((f"sweep_{name}{v}", ["--con_weights"] + [str(x) for x in w]))
    return out


def diagnosis_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """Follow-up experiments that locate the gap to the paper (RESULTS.md). Each entry is
    (name, train.py args, reference config it should be compared with). Every run logs the
    diagnostics in afea/trainer.py (stream-shuffle probes, ISE/ISA weights, SEAL distances,
    gradient norms, WavLM layer weights)."""
    return [
        # 0. baselines re-run with diagnostics (training is identical to the earlier runs)
        ("dx_afea_net", [], "afea_net"),
        ("dx_afea3_nocon", ["--afea_layers", "3", "--no_con"], "afea3"),
        ("dx_wo_afea", ["--afea_layers", "0"], "wo_afea"),
        # 1. SEAL normalisation (margin >= sqrt(2) issue)
        ("seal_l2c", ["--seal_norm", "l2c"], "afea_net"),
        ("seal_l2c_m1.5", ["--seal_norm", "l2c", "--margin", "1.5"], "afea_net"),
        ("seal_none", ["--seal_norm", "none"], "afea_net"),
        # 2. dropout after max pooling
        ("post_wavlm", ["--model", "wavlm", "--dropout_pos", "post_pool"], "wavlm"),
        ("post_wo_afea", ["--afea_layers", "0", "--dropout_pos", "post_pool"], "wo_afea"),
        ("post_afea3_nocon", ["--afea_layers", "3", "--no_con", "--dropout_pos", "post_pool"], "afea3"),
        ("post_afea_net", ["--dropout_pos", "post_pool"], "afea_net"),
        # 3. learnable weighted sum of all WavLM layers
        ("wsum_wavlm", ["--model", "wavlm", "--wavlm_layers", "all"], "wavlm"),
        ("wsum_wo_afea", ["--afea_layers", "0", "--wavlm_layers", "all"], "wo_afea"),
        ("wsum_afea_net", ["--wavlm_layers", "all"], "afea_net"),
        # 4. all fixes combined
        ("fix_wavlm", ["--model", "wavlm", "--dropout_pos", "post_pool", "--wavlm_layers", "all"], "wavlm"),
        ("fix_wo_afea", ["--afea_layers", "0", "--dropout_pos", "post_pool", "--wavlm_layers", "all"], "wo_afea"),
        ("fix_afea_net", ["--dropout_pos", "post_pool", "--wavlm_layers", "all"], "afea_net"),
        ("fix_afea_net_l2c", ["--dropout_pos", "post_pool", "--wavlm_layers", "all", "--seal_norm", "l2c"],
         "afea_net"),
        # 5. ISA with two independent MLPs instead of one shared MLP (Eq. 12 is ambiguous)
        ("isa_sep_afea3_nocon", ["--afea_layers", "3", "--no_con", "--isa_mlp", "separate"], "afea3"),
        ("isa_sep_afea_net", ["--isa_mlp", "separate"], "afea_net"),
        ("isa_sep_fix_afea_net", ["--dropout_pos", "post_pool", "--wavlm_layers", "all", "--isa_mlp", "separate"],
         "fix_afea_net"),
    ]


def protocol_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """Paper-faithful protocol checks: final WavLM layer (official unilm default), epoch picked
    by validation UAR only; each on the original features and on features extracted with
    the official waveform layer-norm (features/<ds>_wnorm, skipped if absent). Every run also
    logs the optimistic test-selected ("oracle") number to bound the effect of selection."""
    base = [("wavlm", ["--model", "wavlm"]), ("wo_afea", ["--afea_layers", "0"]),
            ("wo_align", ["--no_align"]), ("afea_net", [])]
    out = []
    for name, args in base:
        out.append((f"p_{name}", args + ["--select_metric", "uar"], name))
        out.append((f"pn_{name}", args + ["--select_metric", "uar", "--feat_root", f"features/{dataset}_wnorm"],
                    f"p_{name}"))
    return out


def testsel_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """'valid = test' protocol: train on all training folds (no validation split) and report the
    epoch with the best UAR on the held-out fold. Widespread in IEMOCAP papers but optimistically
    biased; results are reported separately from the unbiased runs."""
    t = ["--select", "test", "--val_ratio", "0"]
    fix = ["--dropout_pos", "post_pool", "--wavlm_layers", "all"]
    return [
        ("t_fbank", ["--model", "fbank"] + t, "fbank"),
        ("t_wavlm", ["--model", "wavlm"] + t, "wavlm"),
        ("t_wo_align", ["--no_align"] + t, "wo_align"),
        ("t_wo_afea", ["--afea_layers", "0"] + t, "wo_afea"),
        ("t_afea3_nocon", ["--afea_layers", "3", "--no_con"] + t, "afea3"),
        ("t_afea_net", t, "afea_net"),
        ("t_fix_wavlm", ["--model", "wavlm"] + fix + t, "fix_wavlm"),
        ("t_fix_wo_afea", ["--afea_layers", "0"] + fix + t, "fix_wo_afea"),
        ("t_fix_afea_net", fix + t, "fix_afea_net"),
    ]


def featnorm_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """Feature / representation normalisation that the paper might have applied but does not report.
    fn_ = dataset-level z-score of WavLM and Fbank inputs; ln_ = LayerNorm on the pooled vectors
    before SEAL / AFEA; fnln_ = both. Unbiased (validation-based) selection; reference in column 3."""
    fn, ln = ["--feat_norm", "global"], ["--pooled_norm", "layernorm"]
    return [
        ("fn_wavlm", ["--model", "wavlm"] + fn, "wavlm"),
        ("fn_fbank", ["--model", "fbank"] + fn, "fbank"),
        ("fn_wo_afea", ["--afea_layers", "0"] + fn, "wo_afea"),
        ("fn_wo_align", ["--no_align"] + fn, "wo_align"),
        ("fn_afea_net", fn, "afea_net"),
        ("ln_wo_afea", ["--afea_layers", "0"] + ln, "wo_afea"),
        ("ln_afea3_nocon", ["--afea_layers", "3", "--no_con"] + ln, "afea3"),
        ("ln_wo_align", ["--no_align"] + ln, "wo_align"),
        ("ln_afea_net", ln, "afea_net"),
        ("fnln_wo_align", ["--no_align"] + fn + ln, "wo_align"),
        ("fnln_afea_net", fn + ln, "afea_net"),
    ]


def testsel_wa_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """All Table 4/6 ablation rows under 'valid = test' with the epoch picked by the best test WA
    (tw_<name>); the reference is the same row with unbiased validation-based selection."""
    t = ["--select", "test", "--val_ratio", "0", "--test_select_metric", "wa"]
    return [(f"tw_{n}", a + t, n) for n, a in ablation_configs(dataset)]


def testsel_wa_ln_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """Same as testsel_wa_configs plus LayerNorm on the pooled vectors (twln_<name>), the only setting in
    which SEAL and AFEA both helped under unbiased selection (round 4); the reference is the tw_ row."""
    t = ["--select", "test", "--val_ratio", "0", "--test_select_metric", "wa", "--pooled_norm", "layernorm"]
    return [(f"twln_{n}", a + t, f"tw_{n}") for n, a in ablation_configs(dataset)]



# Names of the LayerNorm runs that already exist in featnorm_configs (same arguments), reused here.
LN_EXISTING = {"wo_align": "ln_wo_align", "wo_afea": "ln_wo_afea", "afea3": "ln_afea3_nocon",
               "wo_con": "ln_afea3_nocon", "afea_net": "ln_afea_net"}


def ln_ablation_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """Every dual-stream Table 4/6 row with LayerNorm on the pooled vectors, unbiased protocol (4 training
    sessions, 10 % of them for epoch selection, held-out session for testing only). Fbank / WavLM are
    single-stream, so LayerNorm does not apply and the plain rows are used. "wo_con" has the same
    arguments as "afea3" and maps to the same run. Reference = the same row without LayerNorm."""
    ln = ["--pooled_norm", "layernorm"]
    out, seen = [], set()
    for n, a in ablation_configs(dataset):
        if n in ("fbank", "wavlm"):
            continue
        name = LN_EXISTING.get(n, f"ln_{n}")
        if name not in seen:
            seen.add(name)
            out.append((name, a + ln, n))
    return out


def afea_dx_configs(dataset: str) -> List[Tuple[str, List[str], str]]:
    """Where does AFEA go wrong? Every AFEA depth without L_con (AFEA-1..4) and the full model, with
      r16_   the ISA MLP built like the FilterLayer of [51] (the block the paper cites as AFEA's
             inspiration): one sigmoid MLP per stream with a reduction-16 bottleneck, hidden = D/16 = 32
             (D = 512), instead of one shared MLP with hidden = D (about 10x fewer AFEA parameters);
      sag_   the original SA-Gate block of [51] in place of AFEA (sigmoid MLP first, then a softmax gate
             that mixes the block inputs convexly, ReLU outputs), i.e. a fusion whose scale cannot grow;
      r16ln_ / sagln_  the same on top of LayerNorm on the pooled vectors (full model only).
    Unbiased protocol. Reference (column 3) = the same row with the paper's AFEA."""
    rows = dict(ablation_configs(dataset))
    r16 = ["--isa_mlp", "separate", "--isa_hidden", "32"]
    sag = ["--afea_block", "sagate"]
    ln = ["--pooled_norm", "layernorm"]
    depths = ("afea1", "afea2", "afea3", "afea4", "afea_net")
    out = [(f"r16_{n}", rows[n] + r16, n) for n in depths]
    out += [(f"sag_{n}", rows[n] + sag, n) for n in depths]
    out += [("r16ln_afea_net", r16 + ln, "ln_afea_net"), ("sagln_afea_net", sag + ln, "ln_afea_net")]
    return out

# Tables 4 and 6 of the paper: (WA, UAR, P, F1)
PAPER = {
    "iemocap": {
        "fbank": (.562, .563, .589, .564), "wavlm": (.728, .728, .736, .730),
        "wo_align": (.740, .739, .747, .738), "wo_afea": (.739, .737, .744, .735),
        "afea1": (.743, .745, .757, .742), "afea2": (.745, .746, .758, .745),
        "afea3": (.746, .749, .755, .748), "afea4": (.744, .746, .754, .743),
        "wo_con": (.746, .749, .755, .748), "con_layer1": (.747, .746, .758, .749),
        "con_layers12": (.748, .748, .762, .751), "afea_net": (.751, .753, .760, .754),
    },
    "ravdess": {
        "fbank": (.484, .465, .501, .459), "wavlm": (.771, .770, .788, .761),
        "wo_align": (.784, .784, .791, .780), "wo_afea": (.782, .780, .789, .778),
        "afea1": (.785, .787, .800, .785), "afea2": (.791, .786, .801, .790),
        "afea3": (.795, .796, .803, .792), "afea4": (.790, .788, .797, .791),
        "wo_con": (.795, .796, .803, .792), "con_layer1": (.797, .796, .809, .797),
        "con_layers12": (.798, .797, .812, .799), "afea_net": (.803, .806, .808, .804),
    },
}

PAPER_ROW_NAMES: Dict[str, str] = {
    "fbank": "Fbank", "wavlm": "WavLM", "wo_align": "w/o L_ali", "wo_afea": "w/o AFEA",
    "afea1": "AFEA-1", "afea2": "AFEA-2", "afea3": "AFEA-3", "afea4": "AFEA-4",
    "wo_con": "w/o L_con", "con_layer1": "w/o L_con1", "con_layers12": "w/o L_con2",
    "afea_net": "AFEA-Net (ours)",
}
