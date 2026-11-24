#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat}"
CHECKPOINT="${CHECKPOINT:-results/qjl_exponent_mantissa_base_2d/rerun/compact_qjl_state.pt}"
ALPHAS="${ALPHAS:-0.0 0.25 0.50 0.75 1.0}"
TASKS="${TASKS:-arc_easy,arc_challenge}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/qjl_residual_alpha_sweep/d2n}"
BACKEND="${BACKEND:-triton}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "checkpoint not found: $CHECKPOINT" >&2
    exit 2
fi
mkdir -p "$OUTPUT_DIR"
alpha_list=()
for alpha in ${ALPHAS//,/ }; do
    if [[ ! "$alpha" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "invalid alpha: $alpha" >&2
        exit 2
    fi
    alpha_list+=("$alpha")
done
if ((${#alpha_list[@]} == 0)); then
    echo "ALPHAS is empty" >&2
    exit 2
fi

for alpha in "${alpha_list[@]}"; do
    dest="$OUTPUT_DIR/alpha_$alpha"
    mkdir -p "$dest"
    echo "[alpha sweep] evaluating alpha=$alpha from $CHECKPOINT"
    "$PYTHON" scripts/QJL/run_codebook_base.py --mode eval \
        --model-path "$MODEL_PATH" --checkpoint "$CHECKPOINT" \
        --backend "$BACKEND" --residual-scale "$alpha" \
        --task "$TASKS" --eval-batch-size "$EVAL_BATCH_SIZE" \
        --eval-limit "$EVAL_LIMIT" --output-dir "$dest" \
        2>&1 | tee "$dest/eval.log"
done

"$PYTHON" - "$OUTPUT_DIR" "$TASKS" "$CHECKPOINT" "${alpha_list[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
tasks = [task.strip() for task in sys.argv[2].split(",") if task.strip()]
checkpoint = sys.argv[3]
alphas = sys.argv[4:]
case_names = {"0.0": "A (Base only)", "0.25": "B", "0.50": "C",
              "0.75": "D", "1.0": "E (Individual baseline)"}
rows = []
for alpha in alphas:
    path = output / f"alpha_{alpha}" / "metrics.json"
    metrics = json.loads(path.read_text())
    summary = metrics["task_summary"]
    row = {"alpha": float(alpha), "case": case_names.get(alpha, ""),
           "checkpoint": metrics["checkpoint"],
           "evaluation_wall_seconds": metrics["evaluation_wall_seconds"],
           "mean_accuracy": summary["avg"], "mean_acc_norm": summary["avg_norm"]}
    task_metrics = {name: (acc, acc_norm) for name, acc, acc_norm in summary["tasks"]}
    for task in tasks:
        acc, norm = task_metrics.get(task, (None, None))
        row[f"{task}_acc"] = acc
        row[f"{task}_acc_norm"] = norm
    rows.append(row)
fields = ["alpha", "case", "checkpoint", "evaluation_wall_seconds",
          "mean_accuracy", "mean_acc_norm"]
for task in tasks:
    fields.extend((f"{task}_acc", f"{task}_acc_norm"))
with (output / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(output / "summary.json").write_text(json.dumps({
    "checkpoint": checkpoint, "tasks": tasks, "cases": rows,
    "acc_norm_fallback": "Use acc for tasks without acc_norm in mean_acc_norm"
}, indent=2, allow_nan=False) + "\n")
print(f"[alpha sweep] saved {output / 'summary.csv'} and summary.json")
PY
