#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen1.5-MoE-A2.7B-Chat}"
GROUPS_PATH="${GROUPS_PATH:-results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qjl_bundle/sweep}"
D_VALUES="${D_VALUES:-2048 4096 8192}"
RANKS="${RANKS:-64 128 256 512}"
TASKS="${TASKS:-arc_easy,arc_challenge,hellaswag,winogrande,boolq,mmlu,openbookqa,rte}"
LAYERS="${LAYERS:-all}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
DIAGNOSTIC_SAMPLES="${DIAGNOSTIC_SAMPLES:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
mkdir -p "$OUTPUT_DIR"
failed=0
for d in $D_VALUES; do
  for rank in $RANKS; do
    for mode in none majority; do
      tag="${mode}_d${d}_r${rank}"
      dest="$OUTPUT_DIR/$tag"
      mkdir -p "$dest"
      checkpoint="$dest/bundle_qjl_state.pt"
      if [[ ! -f "$checkpoint" ]]; then
        if ! "$PYTHON" scripts/QJL/run_bundle.py --mode compress \
            --model-path "$MODEL_PATH" --groups-path "$GROUPS_PATH" \
            --layers "$LAYERS" --qjl-d "$d" --rank "$rank" \
            --bundle-mode "$mode" --residual-mode lowrank \
            --diagnostic-samples "$DIAGNOSTIC_SAMPLES" \
            --checkpoint "$checkpoint" --output-dir "$dest" --skip-eval \
            >"$dest/compress.log" 2>&1; then
          printf '%s\tcompress\tfailed\n' "$tag" >>"$OUTPUT_DIR/failures.tsv"
          failed=1
          continue
        fi
      fi
      if [[ "$RUN_EVAL" == 1 && ! -f "$dest/eval/metrics.json" ]]; then
        if ! "$PYTHON" scripts/QJL/run_bundle.py --mode eval \
            --model-path "$MODEL_PATH" --checkpoint "$checkpoint" \
            --output-dir "$dest/eval" --task "$TASKS" \
            --eval-limit "$EVAL_LIMIT" --eval-batch-size "$EVAL_BATCH_SIZE" \
            >"$dest/eval.log" 2>&1; then
          printf '%s\teval\tfailed\n' "$tag" >>"$OUTPUT_DIR/failures.tsv"
          failed=1
        fi
      fi
    done
  done
done
exit "$failed"
