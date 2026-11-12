#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
CALIB_DATASET="${CALIB_DATASET:-dclm}"
TRAIN_DATA="${TRAIN_DATA:-rescode/data/${CALIB_DATASET}-calib.jsonl}"
N_SENTENCES="${N_SENTENCES:-2400}"
CALIB_SEED="${CALIB_SEED:-42}"
CALIB_STATS_CACHE="${CALIB_STATS_CACHE:-results/qwen/calib_stats/${CALIB_DATASET}-n${N_SENTENCES}-seed${CALIB_SEED}.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/residual_norm_ablation}"
MODEL_PATH="${MODEL_PATH:-}"
GPU="${GPU:-0}"
LAYERS="${LAYERS:-all}"
NUM_EXPERTS="${NUM_EXPERTS:-45}"
SALIENCY_TOP_TAU="${SALIENCY_TOP_TAU:-15.0}"
DELTA="${DELTA:-1e-3}"
RANK="${RANK:-128}"
PYTHON="${PYTHON:-.venv/bin/python}"

export CUDA_VISIBLE_DEVICES="$GPU"
args=(
  --model "$MODEL"
  --calib-dataset "$CALIB_DATASET"
  --calibration-path "$TRAIN_DATA"
  --cache-path "$CALIB_STATS_CACHE"
  --output-dir "$OUTPUT_DIR"
  --layers "$LAYERS"
  --num-groups "$NUM_EXPERTS"
  --singleton-ratio "$("$PYTHON" -c 'import sys; print(float(sys.argv[1]) / 100)' "$SALIENCY_TOP_TAU")"
  --delta "$DELTA"
  --rank "$RANK"
  --n-sentences "$N_SENTENCES"
  --calib-seed "$CALIB_SEED"
)
if [[ -n "$MODEL_PATH" ]]; then
  args+=(--model-path "$MODEL_PATH")
fi
"$PYTHON" scripts/QJL/analyze_residual_norm_ablation.py "${args[@]}" "$@"
