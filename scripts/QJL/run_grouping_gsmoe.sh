#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${GS_MOE_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi

action="${1:?usage: run_grouping_gsmoe.sh test|smoke|analyze|compress|eval [options]}"
shift

case "$action" in
  test)
    CUDA_VISIBLE_DEVICES="" "$python_bin" -m unittest \
      tests/test_grouping_gsmoe.py tests/test_grouping_gsmoe_heldout.py -v "$@"
    ;;
  smoke)
    CUDA_VISIBLE_DEVICES="" "$python_bin" scripts/QJL/run_grouping_gsmoe.py --mode compress \
      --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
      --layers 12 --max-experts 2 --num-groups 1 --probe-rank 4 \
      --shared-ranks 4 --residual-ranks 0,2 --grouping-restarts 1 \
      --grouping-iterations 2 --calibration-blocks 4 \
      --calibration-seqlen 256 --calibration-batch-size 2 \
      --calibration-samples 8 --output-dir results/grouping_gsmoe/full_cov_smoke "$@"
    ;;
  analyze|compress|eval)
    "$python_bin" scripts/QJL/run_grouping_gsmoe.py --mode "$action" "$@"
    ;;
  *)
    echo "unknown action: $action" >&2
    exit 2
    ;;
esac
