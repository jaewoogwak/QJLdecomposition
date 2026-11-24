#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat}"
QJL_D_FACTOR="${QJL_D_FACTOR:-2}"
RANK="${RANK:-512}"
TASKS="${TASKS:-arc_easy,arc_challenge}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
LAYERS="${LAYERS:-all}"
CASES="${CASES:-original,A,B}"
OUTPUT_DIR="${OUTPUT_DIR:-results/base_bundling/pairwise_d2n_r512}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoint}"
BACKEND="${BACKEND:-triton}"
PROJECTION_SEED="${PROJECTION_SEED:-0}"
PAIRING_SEED="${PAIRING_SEED:-0}"
BUNDLE_SEED="${BUNDLE_SEED:-0}"

mkdir -p "$OUTPUT_DIR"
"$PYTHON" scripts/QJL/run_base_bundling.py --mode compress --resume \
    --model-path "$MODEL_PATH" --layers "$LAYERS" \
    --qjl-d-factor "$QJL_D_FACTOR" --rank "$RANK" \
    --projection-seed "$PROJECTION_SEED" --pairing-seed "$PAIRING_SEED" \
    --bundle-seed "$BUNDLE_SEED" --checkpoint-dir "$CHECKPOINT_DIR" \
    --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/compress.log"

for case in ${CASES//,/ }; do
    if [[ "$case" != original && "$case" != A && "$case" != B ]]; then
        echo "invalid case: $case" >&2
        exit 2
    fi
    mkdir -p "$OUTPUT_DIR/$case"
    "$PYTHON" scripts/QJL/run_base_bundling.py --mode eval --case "$case" \
        --model-path "$MODEL_PATH" --layers "$LAYERS" \
        --checkpoint-dir "$CHECKPOINT_DIR" --output-dir "$OUTPUT_DIR" \
        --backend "$BACKEND" --task "$TASKS" --eval-batch-size "$EVAL_BATCH_SIZE" \
        --eval-limit "$EVAL_LIMIT" 2>&1 | tee "$OUTPUT_DIR/$case/eval.log"
done
