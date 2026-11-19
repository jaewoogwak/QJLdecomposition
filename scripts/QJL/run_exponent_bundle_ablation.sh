#!/usr/bin/env bash
# Smoke and full runs use separate cache directories by default.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/.hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be}"
GROUPS_PATH="${GROUPS_PATH:-results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt}"
LAYERS="${LAYERS:-12}"
MAX_EXPERTS="${MAX_EXPERTS:-0}"
QJL_D="${QJL_D:-2048}"
PROJECTION_SEED="${PROJECTION_SEED:-0}"
RANDOM_SEED="${RANDOM_SEED:-42}"

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  output="$OUTPUT_DIR"
else
  output="results/qjl_exponent_bundle_ablation"
  if [[ "$MAX_EXPERTS" != 0 ]]; then
    output="$output/smoke_layers${LAYERS//,/_}_max${MAX_EXPERTS}"
  elif [[ "$LAYERS" != 12 ]]; then
    output="$output/layers${LAYERS//,/_}"
  fi
fi

"$PYTHON" scripts/QJL/analyze_exponent_bundle_ablation.py \
  --model-path "$MODEL_PATH" --groups-path "$GROUPS_PATH" \
  --layers "$LAYERS" --max-experts "$MAX_EXPERTS" \
  --qjl-d "$QJL_D" --projection-seed "$PROJECTION_SEED" \
  --random-seed "$RANDOM_SEED" --output-dir "$output"
