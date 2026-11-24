#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

usage() {
    cat <<'EOF'
Usage:
  bash scripts/QJL/run_residual_bundle_sweep.sh \
    --mode eval --model-path MODEL --output-dir RUN_DIR \
    --bundle-method majority --pairing similarity --qjl-d-factor 2 \
    --checkpoint RUN_DIR/compact_qjl_state.pt [--eval-batch-size 1]

Use --mode compress, eval, or both. A single configuration requires explicit
--bundle-method, --qjl-d-factor, and --checkpoint. Use --all instead to run the
six standard configurations; checkpoint paths are then derived under OUTPUT_DIR.
EOF
}

python=.venv/bin/python
model_path=
output_dir=
checkpoint=
mode=both
method=
factor=
pairing=similarity
backend=triton
layers=all
task=winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte
eval_limit=0
eval_batch_size=1
projection_seed=0
pairing_seed=0
bundle_seed=0
row_chunk=128
all=false

while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --all) all=true; shift ;;
        --python|--model-path|--output-dir|--checkpoint|--mode|--bundle-method|\
        --qjl-d-factor|--pairing|--backend|--layers|--task|--eval-limit|\
        --eval-batch-size|--projection-seed|--pairing-seed|--bundle-seed|--row-chunk)
            option=$1
            if (($# < 2)); then echo "missing value for $option" >&2; exit 2; fi
            value=$2
            case "$option" in
                --python) python=$value ;;
                --model-path) model_path=$value ;;
                --output-dir) output_dir=$value ;;
                --checkpoint) checkpoint=$value ;;
                --mode) mode=$value ;;
                --bundle-method) method=$value ;;
                --qjl-d-factor) factor=$value ;;
                --pairing) pairing=$value ;;
                --backend) backend=$value ;;
                --layers) layers=$value ;;
                --task) task=$value ;;
                --eval-limit) eval_limit=$value ;;
                --eval-batch-size) eval_batch_size=$value ;;
                --projection-seed) projection_seed=$value ;;
                --pairing-seed) pairing_seed=$value ;;
                --bundle-seed) bundle_seed=$value ;;
                --row-chunk) row_chunk=$value ;;
            esac
            shift 2 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$model_path" || -z "$output_dir" ]]; then
    echo "--model-path and --output-dir are required" >&2
    exit 2
fi
if [[ "$mode" != compress && "$mode" != eval && "$mode" != both ]]; then
    echo "--mode must be compress, eval, or both" >&2
    exit 2
fi
if [[ "$pairing" != random && "$pairing" != similarity ]]; then
    echo "--pairing must be random or similarity" >&2
    exit 2
fi
if [[ "$backend" != triton && "$backend" != reference ]]; then
    echo "--backend must be triton or reference" >&2
    exit 2
fi
if [[ "$all" == true ]]; then
    if [[ -n "$method" || -n "$factor" || -n "$checkpoint" ]]; then
        echo "--all cannot be combined with --bundle-method, --qjl-d-factor, or --checkpoint" >&2
        exit 2
    fi
else
    if [[ -z "$method" || -z "$factor" || -z "$checkpoint" ]]; then
        echo "single runs require --bundle-method, --qjl-d-factor, and --checkpoint" >&2
        exit 2
    fi
    if [[ "$method" != none && "$method" != majority && "$method" != magnitude-aware ]]; then
        echo "--bundle-method must be none, majority, or magnitude-aware" >&2
        exit 2
    fi
fi

run_one() {
    local selected_method=$1 selected_factor=$2 selected_checkpoint=$3 dest=$4
    mkdir -p "$dest"
    if [[ "$mode" != eval && ! -f "$selected_checkpoint" ]]; then
        "$python" scripts/QJL/run_residual_bundle.py --mode compress \
            --model-path "$model_path" --layers "$layers" \
            --bundle-method "$selected_method" --pairing "$pairing" \
            --qjl-d-factor "$selected_factor" --projection-seed "$projection_seed" \
            --pairing-seed "$pairing_seed" --bundle-seed "$bundle_seed" \
            --row-chunk "$row_chunk" --backend "$backend" \
            --checkpoint "$selected_checkpoint" --output-dir "$dest" \
            --skip-eval 2>&1 | tee "$dest/compress.log"
    fi
    if [[ "$mode" != compress ]]; then
        if [[ ! -f "$selected_checkpoint" ]]; then
            echo "checkpoint missing: $selected_checkpoint" >&2
            return 1
        fi
        if [[ "$mode" == eval || ! -f "$dest/eval/metrics.json" ]]; then
            "$python" scripts/QJL/run_residual_bundle.py --mode eval \
                --model-path "$model_path" --checkpoint "$selected_checkpoint" \
                --bundle-method "$selected_method" --pairing "$pairing" \
                --qjl-d-factor "$selected_factor" --backend "$backend" \
                --output-dir "$dest/eval" --task "$task" \
                --eval-limit "$eval_limit" --eval-batch-size "$eval_batch_size" \
                2>&1 | tee "$dest/eval.log"
        fi
    fi
}

if [[ "$all" == true ]]; then
    for spec in none:1 none:2 majority:2 majority:4 magnitude-aware:2 magnitude-aware:4; do
        selected_method=${spec%%:*}
        selected_factor=${spec##*:}
        dest="$output_dir/${selected_method}_d${selected_factor}n_${pairing}"
        run_one "$selected_method" "$selected_factor" \
            "$dest/compact_qjl_state.pt" "$dest"
    done
else
    run_one "$method" "$factor" "$checkpoint" "$output_dir"
fi
