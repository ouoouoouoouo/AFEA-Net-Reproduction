#!/usr/bin/env bash
# Reproduce the rows of Table 4 (IEMOCAP) / Table 6 (RAVDESS).
# Usage: bash scripts/run_ablations.sh iemocap manifests/iemocap.csv features/iemocap [extra train.py args]
set -euo pipefail
DS=$1; MAN=$2; FEAT=$3; shift 3
if [ "$DS" = iemocap ]; then W1="0.8"; W2="0.8 0.5"; else W1="0.3"; W2="0.3 0.2"; fi
run() { name=$1; shift; python scripts/train.py --dataset "$DS" --manifest "$MAN" --feat_root "$FEAT" --out "runs/$DS/$name" "$@"; }

run fbank          --model fbank "$@"
run wavlm          --model wavlm "$@"
run wo_align       --no_align "$@"
run wo_afea        --afea_layers 0 "$@"
for L in 1 2 3 4; do run afea$L --afea_layers $L --no_con "$@"; done
run wo_con         --no_con "$@"
# continuity learning on the first k AFEA layers only (see ASSUMPTIONS.md, A-ABL-2)
run con_layer1     --con_weights $W1 0 0 "$@"
run con_layers12   --con_weights $W2 0 "$@"
run afea_net       "$@"

python - "$DS" <<'PY'
import json, glob, os, sys
ds = sys.argv[1]
print(f"{'setting':14s}   WA     UAR    P      F1")
for d in sorted(glob.glob(f"runs/{ds}/*/summary.json")):
    m = json.load(open(d))["fold_mean"]
    print(f"{os.path.basename(os.path.dirname(d)):14s} " + " ".join(f"{m[k]:.4f}" for k in ("WA", "UAR", "P", "F1")))
PY
