#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qjl_residual_bundle/sweep}"
LAYERS="${LAYERS:-all}"
TASKS="${TASKS:-winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
PAIRING="${PAIRING:-similarity}"
BACKEND="${BACKEND:-triton}"
PROJECTION_SEED="${PROJECTION_SEED:-0}"
PAIRING_SEED="${PAIRING_SEED:-0}"
BUNDLE_SEED="${BUNDLE_SEED:-0}"
RUNS="${RUNS:-none:1 none:2 majority:2 majority:4 magnitude-aware:2 magnitude-aware:4}"

mkdir -p "$OUTPUT_DIR"
for spec in $RUNS; do
    method="${spec%%:*}"
    factor="${spec##*:}"
    tag="${method}_d${factor}n_${PAIRING}"
    dest="$OUTPUT_DIR/$tag"
    checkpoint="$dest/compact_qjl_state.pt"
    mkdir -p "$dest"
    if [[ ! -f "$checkpoint" ]]; then
        "$PYTHON" scripts/QJL/run_residual_bundle.py --mode compress \
            --model-path "$MODEL_PATH" --layers "$LAYERS" \
            --bundle-method "$method" --pairing "$PAIRING" \
            --qjl-d-factor "$factor" --projection-seed "$PROJECTION_SEED" \
            --pairing-seed "$PAIRING_SEED" --bundle-seed "$BUNDLE_SEED" \
            --backend "$BACKEND" --checkpoint "$checkpoint" --output-dir "$dest" \
            --skip-eval 2>&1 | tee "$dest/compress.log"
    fi
    if [[ ! -f "$dest/eval/metrics.json" ]]; then
        "$PYTHON" scripts/QJL/run_residual_bundle.py --mode eval \
            --model-path "$MODEL_PATH" --checkpoint "$checkpoint" \
            --backend "$BACKEND" --output-dir "$dest/eval" --task "$TASKS" \
            --eval-limit "$EVAL_LIMIT" --eval-batch-size "$EVAL_BATCH_SIZE" \
            2>&1 | tee "$dest/eval.log"
    fi
done
