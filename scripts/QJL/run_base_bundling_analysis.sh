#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat}"
CALIBRATION_JSONL="${CALIBRATION_JSONL:-rescode/data/wikipedia-analysis-calibration.jsonl}"
HELDOUT_JSONL="${HELDOUT_JSONL:-rescode/data/wikipedia-analysis-heldout.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/base_bundling_analysis}"
STAGE="${STAGE:-all}"
LAYERS="${LAYERS:-12}"
PROJECTIONS="${PROJECTIONS:-gate,up,down}"
CALIBRATION_SIZES="${CALIBRATION_SIZES:-128,256,512}"
HELDOUT_SIZE="${HELDOUT_SIZE:-128}"
INPUT_RANKS="${INPUT_RANKS:-32,64,128,256,512}"
RESIDUAL_RANKS="${RESIDUAL_RANKS:-0,32,64,128,256,352}"
QJL_D_FACTOR="${QJL_D_FACTOR:-2}"
SPECTRUM_SCOPE="${SPECTRUM_SCOPE:-representative}"
SPECTRUM_CASES="${SPECTRUM_CASES:-A,B,C2,C1,D,E,F,G,H}"

args=(--stage "$STAGE" --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR"
      --layers "$LAYERS" --projections "$PROJECTIONS"
      --calibration-sizes "$CALIBRATION_SIZES" --heldout-size "$HELDOUT_SIZE"
      --input-ranks "$INPUT_RANKS" --residual-ranks "$RESIDUAL_RANKS"
      --qjl-d-factor "$QJL_D_FACTOR" --spectrum-scope "$SPECTRUM_SCOPE"
      --spectrum-cases "$SPECTRUM_CASES" --resume)
if [[ -n "$CALIBRATION_JSONL" ]]; then args+=(--calibration-jsonl "$CALIBRATION_JSONL"); fi
if [[ -n "$HELDOUT_JSONL" ]]; then args+=(--heldout-jsonl "$HELDOUT_JSONL"); fi
mkdir -p "$OUTPUT_DIR"
"$PYTHON" scripts/QJL/analyze_base_bundling.py "${args[@]}" "$@" \
    2>&1 | tee "$OUTPUT_DIR/$STAGE.log"
