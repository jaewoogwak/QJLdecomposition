#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON="${PYTHON:-.venv/bin/python}"
MODE="${MODE:-sweep}"
MODEL_PATH="${MODEL_PATH:-/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat}"
OUTPUT_DIR="${OUTPUT_DIR:-results/activation_bundle/d2n_k2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoint}"
CACHE_DIR="${CACHE_DIR:-$OUTPUT_DIR/cache}"
QJL_D_FACTOR="${QJL_D_FACTOR:-2}"
GROUP_SIZE="${GROUP_SIZE:-2}"
PAIRING="${PAIRING:-similarity}"
LAYERS="${LAYERS:-all}"
CASES="${CASES:-A,B,C,D,E}"
TASKS="${TASKS:-arc_easy,arc_challenge,hellaswag,winogrande}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
BACKEND="${BACKEND:-triton}"

args=(--mode "$MODE" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR"
      --checkpoint-dir "$CHECKPOINT_DIR" --cache-dir "$CACHE_DIR"
      --qjl-d-factor "$QJL_D_FACTOR" --group-size "$GROUP_SIZE"
      --pairing "$PAIRING" --layers "$LAYERS"
      --cases "$CASES" --task "$TASKS" --eval-batch-size "$EVAL_BATCH_SIZE"
      --eval-limit "$EVAL_LIMIT" --backend "$BACKEND")
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
if [[ -n "${RANK:-}" ]]; then args+=(--rank "$RANK"); fi
if [[ -n "${RANKS:-}" ]]; then args+=(--ranks "$RANKS"); fi
if [[ "${SKIP_EVAL:-0}" == 1 ]]; then args+=(--skip-eval); fi
if [[ -n "${CALIBRATION_JSONL:-}" ]]; then args+=(--calibration-jsonl "$CALIBRATION_JSONL"); fi
if [[ -n "${HELDOUT_JSONL:-}" ]]; then args+=(--heldout-jsonl "$HELDOUT_JSONL"); fi
if [[ -n "${CALIBRATION_SIZE:-}" ]]; then args+=(--calibration-size "$CALIBRATION_SIZE"); fi
if [[ -n "${HELDOUT_SIZE:-}" ]]; then args+=(--heldout-size "$HELDOUT_SIZE"); fi
if [[ -n "${CALIBRATION_BATCH_SIZE:-}" ]]; then args+=(--calibration-batch-size "$CALIBRATION_BATCH_SIZE"); fi
if [[ -n "${REFINE_SWEEPS:-}" ]]; then args+=(--refine-sweeps "$REFINE_SWEEPS"); fi
if [[ -n "${REFINE_BLOCK:-}" ]]; then args+=(--refine-block "$REFINE_BLOCK"); fi
if [[ "${REFINE_ALL_BITS:-0}" == 1 ]]; then args+=(--refine-all-bits); fi

exec "$PYTHON" scripts/QJL/run_activation_bundle.py "${args[@]}" "$@"
