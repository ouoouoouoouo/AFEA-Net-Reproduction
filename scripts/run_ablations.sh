#!/usr/bin/env bash
# Reproduce the rows of Table 4 (IEMOCAP) / Table 6 (RAVDESS).
# Usage: bash scripts/run_ablations.sh iemocap manifests/iemocap.csv features/iemocap [extra train.py args]
#   GPUS="0 1 2 3" bash scripts/run_ablations.sh ...   # spread the 12 configs over 4 GPUs (one process per GPU)
#   extra args are passed to every run, e.g. --cache --wandb afea-net --wandb_group iemocap_ablations
set -euo pipefail
DS=$1; MAN=$2; FEAT=$3; shift 3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ "$DS" = iemocap ]; then W1="0.8"; W2="0.8 0.5"; else W1="0.3"; W2="0.3 0.2"; fi

# name | train.py args
CONFIGS=(
  "fbank|--model fbank"
  "wavlm|--model wavlm"
  "wo_align|--no_align"
  "wo_afea|--afea_layers 0"
  "afea1|--afea_layers 1 --no_con"
  "afea2|--afea_layers 2 --no_con"
  "afea3|--afea_layers 3 --no_con"
  "afea4|--afea_layers 4 --no_con"
  "wo_con|--no_con"
  "con_layer1|--con_weights $W1 0 0"     # continuity on first k layers (ASSUMPTIONS.md, A-ABL-2)
  "con_layers12|--con_weights $W2 0"
  "afea_net|"
)

run() {  # $1 = "name|args"
  local name=${1%%|*} cfg_args=${1#*|}
  mkdir -p "runs/$DS/$name"
  # shellcheck disable=SC2086
  python "$SCRIPT_DIR/train.py" --dataset "$DS" --manifest "$MAN" --feat_root "$FEAT" \
      --out "runs/$DS/$name" $cfg_args "${EXTRA[@]}" > "runs/$DS/$name/stdout.log" 2>&1 \
    && echo "[done] $name" || echo "[FAIL] $name (see runs/$DS/$name/stdout.log)"
}
EXTRA=("$@")

read -r -a GPU_LIST <<< "${GPUS:-0}"
N=${#GPU_LIST[@]}
for ((w = 0; w < N; w++)); do
  (
    export CUDA_VISIBLE_DEVICES=${GPU_LIST[$w]}
    for ((i = w; i < ${#CONFIGS[@]}; i += N)); do run "${CONFIGS[$i]}"; done
  ) &
done
wait

python - "$DS" <<'PY'
import json, glob, os, sys
ds = sys.argv[1]
print(f"{'setting':14s}   WA     UAR    P      F1")
for d in sorted(glob.glob(f"runs/{ds}/*/summary.json")):
    m = json.load(open(d))["fold_mean"]
    print(f"{os.path.basename(os.path.dirname(d)):14s} " + " ".join(f"{m[k]:.4f}" for k in ("WA", "UAR", "P", "F1")))
PY
