#!/usr/bin/env bash
# Reuse the existing unbundled QJL path; keep all majority outputs untouched.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/.hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be}"
GROUPS_PATH="${GROUPS_PATH:-results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/qjl_bundle}"
LAYERS="${LAYERS:-all}"
RANKS="${RANKS:-64 128 512}"
TASKS="${TASKS:-arc_easy,arc_challenge}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
RUN_EVAL="${RUN_EVAL:-1}"

if [[ "$LAYERS" == all ]]; then
  suffix=""
else
  suffix="_layers${LAYERS//,/_}"
fi

for rank in $RANKS; do
  output="$OUTPUT_ROOT/individual_d2048_r${rank}${suffix}"
  checkpoint="$output/bundle_qjl_state.pt"
  mkdir -p "$output"
  if [[ -f "$checkpoint" ]]; then
    "$PYTHON" - "$checkpoint" "$rank" "$LAYERS" <<'PY'
import sys, torch
path, rank, layers = sys.argv[1:]
payload = torch.load(path, map_location="meta", weights_only=False)
config = payload["config"]
expected_layers = list(range(24)) if layers == "all" else sorted(int(x) for x in layers.split(","))
assert (config["bundle_mode"] == "none" and config["residual_mode"] == "lowrank"
        and config["qjl_d"] == 2048 and config["rank"] == int(rank)
        and config["layers"] == expected_layers), f"existing checkpoint settings differ: {path}"
print(f"[individual] reused checkpoint: {path}", flush=True)
PY
  else
    "$PYTHON" scripts/QJL/run_bundle.py \
      --mode compress --model-path "$MODEL_PATH" --groups-path "$GROUPS_PATH" \
      --layers "$LAYERS" --qjl-d 2048 --rank "$rank" \
      --bundle-mode none --residual-mode lowrank \
      --projection-seed 0 --svd-seed 42 --svd-oversample 8 --svd-niter 2 \
      --backend triton --output-dir "$output" --skip-eval
  fi
  if [[ "$RUN_EVAL" == 1 ]]; then
    eval_dir="$output/eval_arc"
    if [[ -e "$eval_dir/metrics.json" ]]; then
      echo "[individual] existing evaluation preserved: $eval_dir/metrics.json"
    else
      "$PYTHON" scripts/QJL/run_bundle.py \
        --mode eval --model-path "$MODEL_PATH" --checkpoint "$checkpoint" \
        --backend triton --task "$TASKS" --eval-limit "$EVAL_LIMIT" \
        --eval-batch-size "$EVAL_BATCH_SIZE" --output-dir "$eval_dir"
    fi
  fi
done
