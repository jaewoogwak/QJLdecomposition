#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
CASES="${CASES:-A,B,C,D,E}"
D_FACTORS="${D_FACTORS:-1,2,4}"
LAYERS="${LAYERS:-all}"
TASKS="${TASKS:-arc_easy,arc_challenge,mmlu,openbookqa,rte}"
OUTPUT_DIR="${OUTPUT_DIR:-results/residual_ablation}"
MODEL_PATH="${MODEL_PATH:-}"
MAX_EXPERTS="${MAX_EXPERTS:-}"
SKIP_EVAL="${SKIP_EVAL:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GPTQ_CALIBRATION_PATH="${GPTQ_CALIBRATION_PATH:-rescode/data/dclm-calib.jsonl}"
DIAGNOSTIC_PATH="${DIAGNOSTIC_PATH:-rescode/data/wikipedia-calib.jsonl}"
GROUPING_CACHE="${GROUPING_CACHE:-results/qwen/calib_stats/dclm-n2400-seed42.pt}"
SAMPLE_CACHE="${SAMPLE_CACHE:-$OUTPUT_DIR/routed_samples.pt}"
GPTQ_CALIBRATION_SAMPLES="${GPTQ_CALIBRATION_SAMPLES:-128}"
DIAGNOSTIC_SAMPLES="${DIAGNOSTIC_SAMPLES:-16}"
N_SENTENCES="${N_SENTENCES:-2400}"
BACKEND="${BACKEND:-triton}"
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

args=(--mode sweep --cases "$CASES" --d-factors "$D_FACTORS" --layers "$LAYERS"
      --tasks "$TASKS" --output-dir "$OUTPUT_DIR" --eval-limit "$EVAL_LIMIT"
      --eval-batch-size "$EVAL_BATCH_SIZE" --backend "$BACKEND"
      --gptq-calibration-path "$GPTQ_CALIBRATION_PATH"
      --diagnostic-path "$DIAGNOSTIC_PATH" --grouping-cache "$GROUPING_CACHE"
      --sample-cache "$SAMPLE_CACHE" --n-sentences "$N_SENTENCES"
      --gptq-calibration-samples "$GPTQ_CALIBRATION_SAMPLES"
      --diagnostic-samples "$DIAGNOSTIC_SAMPLES")
[[ -z "$MODEL_PATH" ]] || args+=(--model-path "$MODEL_PATH")
[[ -z "$MAX_EXPERTS" ]] || args+=(--max-experts "$MAX_EXPERTS")
[[ "$SKIP_EVAL" != 1 ]] || args+=(--skip-eval)

"$PYTHON" scripts/QJL/run_residual_ablation.py "${args[@]}" "$@"

# Auto-finalize only a complete full-model, full-dataset sweep. Override with
# AUTO_PUSH_AND_STOP=0. Smoke/partial runs always leave the instance running.
complete=1
[[ "$LAYERS" == all && -z "$MAX_EXPERTS" && "$SKIP_EVAL" == 0 && "$EVAL_LIMIT" == 0 ]] || complete=0
for arg in "$@"; do
  case "$arg" in
    --layers|--layers=*|--max-experts|--max-experts=*|--skip-eval|--eval-limit|--eval-limit=*) complete=0 ;;
  esac
done
if [[ "${AUTO_PUSH_AND_STOP:-$complete}" == 1 && "$complete" == 1 ]]; then
  "$PYTHON" - "$OUTPUT_DIR" <<'PY'
import csv, sys
from pathlib import Path
root = Path(sys.argv[1])
with (root / 'summary.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
if not rows or any(not row['accuracy_mean'] for row in rows):
    raise SystemExit('refusing to stop: incomplete downstream accuracy in summary.csv')
for name in ('projection_errors.csv', 'memory_breakdown.csv', 'latency.csv', 'REPORT.md'):
    if not (root / name).is_file() or not (root / name).stat().st_size:
        raise SystemExit(f'refusing to stop: missing {name}')
PY
  bash scripts/QJL/push_results_and_shutdown.sh "$OUTPUT_DIR"
fi
