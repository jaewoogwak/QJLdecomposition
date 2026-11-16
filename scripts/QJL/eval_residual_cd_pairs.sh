#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-results/residual_ablation_96gb_CD}"
TASKS="${TASKS:-arc_easy,arc_challenge}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
BACKEND="${BACKEND:-triton}"

metric_complete() {
  "$PYTHON" - "$OUTPUT_DIR" "$1" "$TASKS" <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1]) / sys.argv[2]
path = folder / "eval_metrics.json"
try:
    metrics = json.loads(path.read_text())
    config = json.loads((folder / "config.json").read_text())
    metas = [json.loads(p.read_text()) for p in folder.glob("layer_*.meta.json")]
    valid = (metrics.get("tasks") == sys.argv[3]
             and metrics.get("eval_limit") == 0
             and metrics.get("config_digest") == config["digest"]
             and len(metas) == 24
             and sorted(metrics.get("source_sha256s", []))
             == sorted(meta["source_sha256"] for meta in metas))
except (OSError, ValueError, KeyError):
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

# The original sweep may be paused while its C_d2 evaluation continues.
# Wait for that evaluation, then terminate only the paused sweep scheduler.
if [[ -n "${WAIT_EVAL_PID:-}" && -n "${PAUSED_SWEEP_PID:-}" ]]; then
  scheduler_released=0
  release_scheduler_on_failure() {
    if [[ "$scheduler_released" == 0 ]]; then
      kill -CONT "$PAUSED_SWEEP_PID" 2>/dev/null || true
    fi
  }
  trap release_scheduler_on_failure EXIT
  echo "Waiting for C_d2 evaluation process $WAIT_EVAL_PID"
  until metric_complete C_d2; do
    state="$(ps -p "$WAIT_EVAL_PID" -o stat= 2>/dev/null || true)"
    if [[ -z "$state" || "$state" == Z* ]]; then
      echo "C_d2 evaluation exited without complete metrics" >&2
      exit 1
    fi
    sleep 15
  done
  while :; do
    state="$(ps -p "$WAIT_EVAL_PID" -o stat= 2>/dev/null || true)"
    [[ -z "$state" || "$state" == Z* ]] && break
    sleep 2
  done
  kill -KILL "$PAUSED_SWEEP_PID" 2>/dev/null || true
  scheduler_released=1
  trap - EXIT
  echo "C_d2 completed; stopped sweep scheduler $PAUSED_SWEEP_PID"
fi

for factor in 1 2 4; do
  for case in C D; do
    variant="${case}_d${factor}"
    if metric_complete "$variant"; then
      echo "[pairwise] reuse $variant evaluation"
      continue
    fi
    echo "[pairwise] evaluating $variant"
    "$PYTHON" scripts/QJL/run_residual_ablation.py \
      --mode eval --cases "$case" --d-factors "$factor" \
      --output-dir "$OUTPUT_DIR" --tasks "$TASKS" \
      --eval-limit 0 --eval-batch-size "$EVAL_BATCH_SIZE" --backend "$BACKEND"
    metric_complete "$variant"
  done
  "$PYTHON" - "$OUTPUT_DIR" "$factor" <<'PY'
import json
import sys
from pathlib import Path

root, factor = Path(sys.argv[1]), sys.argv[2]
c = json.loads((root / f"C_d{factor}/eval_metrics.json").read_text())
d = json.loads((root / f"D_d{factor}/eval_metrics.json").read_text())
ca, da = c["task_summary"]["avg"], d["task_summary"]["avg"]
print(f"[pairwise] d{factor}: C={ca:.4f} D={da:.4f} D-C={da-ca:+.4f}", flush=True)
PY
done

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, "scripts/QJL")
from run_residual_ablation import aggregate

selected = [(case, factor) for factor in (1, 2, 4) for case in ("C", "D")]
aggregate(Namespace(output_dir=Path(sys.argv[1]), max_experts=None), selected)
PY

if [[ "${FINALIZE:-0}" == 1 ]]; then
  roots=("$OUTPUT_DIR")
  [[ -z "${EXTRA_RESULTS_DIR:-}" ]] || roots+=("$EXTRA_RESULTS_DIR")
  find "${roots[@]}" -type f \
    \( -name '*.csv' -o -name '*.json' -o -name '*.md' -o -name '*.txt' \) \
    -size -100000000c -print0 | xargs -0 -r git add -f --
  git diff --cached --check
  if ! git diff --cached --quiet; then
    author="$(git log -1 --format=%an)"
    email="$(git log -1 --format=%ae)"
    git -c user.name="$author" -c user.email="$email" \
      commit -m "Add matched-budget QJL versus low-rank ARC results"
  fi
  git push origin ablation
  bash scripts/QJL/push_results_and_shutdown.sh "$OUTPUT_DIR"
fi
