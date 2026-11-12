#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-}"
BASELINE_CKPT="${BASELINE_CKPT:-}"
D_FACTORS="${D_FACTORS:-2,4,8,16}"
TASKS="${TASKS:-arc_easy,arc_challenge,mmlu,openbookqa,rte}"
RESIDUAL_MODE="${RESIDUAL_MODE:-qjl}"
if [[ "$RESIDUAL_MODE" == "fp32" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-results/base_qjl_sweep_fp32_residual}"
else
  OUTPUT_DIR="${OUTPUT_DIR:-results/base_qjl_sweep}"
fi
B_OUTPUT_DIR="${B_OUTPUT_DIR:-results/base_qjl_sweep}"
FP32_RESIDUAL_DEVICE="${FP32_RESIDUAL_DEVICE:-auto}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
LAYERS="${LAYERS:-all}"
BASE_SEED="${BASE_SEED:-1}"
RESIDUAL_SEED="${RESIDUAL_SEED:-0}"
BACKEND="${BACKEND:-triton}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="$GPU"
args=(--mode sweep --residual-mode "$RESIDUAL_MODE"
      --b-output-dir "$B_OUTPUT_DIR" --fp32-residual-device "$FP32_RESIDUAL_DEVICE"
      --d-factors "$D_FACTORS" --tasks "$TASKS"
      --output-dir "$OUTPUT_DIR" --eval-batch-size "$EVAL_BATCH_SIZE"
      --eval-limit "$EVAL_LIMIT" --layers "$LAYERS"
      --base-seed "$BASE_SEED" --residual-seed "$RESIDUAL_SEED"
      --backend "$BACKEND")
if [[ -n "$MODEL_PATH" ]]; then args+=(--model-path "$MODEL_PATH"); fi
if [[ -n "$BASELINE_CKPT" ]]; then args+=(--baseline-ckpt "$BASELINE_CKPT"); fi
"$PYTHON" scripts/QJL/run_base_qjl_sweep.py "${args[@]}" "$@"
