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
EVAL_VARIANTS="${EVAL_VARIANTS:-${CASES:-original,base-only,base-plus-lowrank}}"
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

for variant in ${EVAL_VARIANTS//,/ }; do
    case "$variant" in
        original) eval_variant=original; case_dir=original ;;
        base-only|A) eval_variant=base-only; case_dir=A ;;
        base-plus-lowrank|B) eval_variant=base-plus-lowrank; case_dir=B ;;
        *) echo "invalid evaluation variant: $variant" >&2; exit 2 ;;
    esac
    mkdir -p "$OUTPUT_DIR/$case_dir"
    "$PYTHON" scripts/QJL/run_base_bundling.py --mode eval --eval-variant "$eval_variant" \
        --model-path "$MODEL_PATH" --layers "$LAYERS" \
        --checkpoint-dir "$CHECKPOINT_DIR" --output-dir "$OUTPUT_DIR" \
        --backend "$BACKEND" --task "$TASKS" --eval-batch-size "$EVAL_BATCH_SIZE" \
        --eval-limit "$EVAL_LIMIT" 2>&1 | tee "$OUTPUT_DIR/$case_dir/eval.log"
done
