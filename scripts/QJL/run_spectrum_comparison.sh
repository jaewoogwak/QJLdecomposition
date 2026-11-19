#!/usr/bin/env bash
# Exact full spectra, resumed one layer at a time. Set LAYERS=12 for smoke.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/workspace/.hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be}"
GROUPS_PATH="${GROUPS_PATH:-results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt}"
MAJORITY_CHECKPOINT="${MAJORITY_CHECKPOINT:-results/qjl_bundle/full_majority_d2048_r128/bundle_qjl_state.pt}"
LAYERS="${LAYERS:-all}"
BUNDLE_MODE="${BUNDLE_MODE:-both}"
MAX_EXPERTS="${MAX_EXPERTS:-0}"
FIGURE_LAYER="${FIGURE_LAYER:-12}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qjl_bundle/spectrum_comparison}"

"$PYTHON" scripts/QJL/analyze_residual_spectrum.py \
  --model-path "$MODEL_PATH" --groups-path "$GROUPS_PATH" \
  --majority-checkpoint "$MAJORITY_CHECKPOINT" --layers "$LAYERS" \
  --qjl-d 2048 --projection-seed 0 --bundle-mode "$BUNDLE_MODE" \
  --figure-layer "$FIGURE_LAYER" --max-experts "$MAX_EXPERTS" \
  --output-dir "$OUTPUT_DIR"
