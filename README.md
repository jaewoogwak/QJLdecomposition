# QJLdecomposition

Standalone Git repository for QJL-based compression experiments on
`Qwen/Qwen1.5-MoE-A2.7B-Chat`. The main implementation supports a packed
2-bit codebook Base or a sign/exponent Base with a row-wise mantissa, plus a
packed QJL residual for every routed expert's Gate, Up, and Down projections.

This repository starts from the source tree of
[`jaewoogwak/ResCode`](https://github.com/jaewoogwak/ResCode),
`feature/codebook-base-qjl` at commit `2c0f706d4a3e76d32a62f561ad58b22b289c69ff`.
It has its own Git history and no ResCode Git remote or submodule. The Python
package remains named `rescode` so the experiment's imports continue to work.
The earlier ResCode project documentation is preserved in
[`README.rescode.md`](README.rescode.md).

## Environment

On Linux x86_64 with an NVIDIA GPU:

```bash
bash setup.sh
source .venv/bin/activate
```

The included `setup.sh` is identical to the one on ResCode's
`qjl-residual-scale` branch. It installs the pinned packages from
`requirements-rescode.lock` without cloning ResCode. Its optional environment
directory variable is still named `RESCODE_ENV_DIR` for compatibility.

Download the model revision expected by the scripts:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen1.5-MoE-A2.7B-Chat",
    revision="ec052fda178e241c7c443468d2fa1db6618996be",
)
PY
```

If the model is stored elsewhere, pass its snapshot directory with
`--model-path`.

## Compress and evaluate

For the exponent/mantissa Base with QJL dimension four times the input width:

```bash
.venv/bin/python scripts/QJL/run_codebook_base.py \
  --mode compress --base-method exponent-mantissa \
  --layers all --qjl-d-factor 4 \
  --output-dir results/local_4d/full --skip-eval

.venv/bin/python scripts/QJL/run_codebook_base.py \
  --mode eval \
  --checkpoint results/local_4d/full/compact_qjl_state.pt \
  --output-dir results/local_4d/full/eval_8tasks \
  --backend triton \
  --task winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte \
  --eval-limit 0 --eval-batch-size 16
```

Use `--base-method codebook` for the 2-bit codebook Base. See
`scripts/QJL/run_codebook_base.py --help` for other options. Generated `.pt`
checkpoints are ignored by Git; the committed result files contain diagnostics
and evaluation metrics.

## Pairwise residual bundling (`bundle` branch)

The new experiment retains the exponent-mantissa Base and row norm for each
routed expert while sharing one packed residual sketch between two experts in
the same layer and projection. Pairing can be seeded random or exact maximum
weight matching on binary Hamming agreement. Compression does not fit scales
or use activation data.

```bash
MODEL=/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat
for method in majority magnitude-aware; do
  out="results/qjl_residual_bundle/${method}_d2n_similarity"
  .venv/bin/python scripts/QJL/run_residual_bundle.py --mode compress \
    --model-path "$MODEL" --layers all --bundle-method "$method" \
    --pairing similarity --qjl-d-factor 2 --backend triton \
    --output-dir "$out" --skip-eval
  .venv/bin/python scripts/QJL/run_residual_bundle.py --mode eval \
    --model-path "$MODEL" --checkpoint "$out/compact_qjl_state.pt" \
    --output-dir "$out/eval" --backend triton --eval-limit 0
done
```

`--task arc_easy,arc_challenge` limits evaluation to named tasks; the default
is winogrande, arc_easy, arc_challenge, boolq, hellaswag, mmlu, openbookqa,
and rte. `--eval-limit 0` evaluates all questions. The script also supports
`--bundle-method none`, `--pairing random`, and independent projection,
pairing, and bundle seeds. To run the six standard bit-budget configurations,
use `MODEL_PATH="$MODEL" bash scripts/QJL/run_residual_bundle_sweep.sh`.
`RUNS='majority:2 magnitude-aware:2'` selects only the two D=2n bundles.
The new evaluator defaults to batch size 1 because full eight-task evaluation
at batch size 16 exhausted the 48 GB RTX PRO 5000 GPU during FP32 logits
allocation. Use `--eval-batch-size` or `EVAL_BATCH_SIZE` in the sweep to override.

The checkpoint stores `layers[layer].pairs[projection]`, expert-specific
`base` and `norms`, and `layers[layer].sketches[projection][pair_id]` once per
pair. Expert records refer to a sketch ID; loading places each pair's sketch
in one GPU allocation shared by both expert modules. The Gaussian projection
is regenerated from its seed. `storage.json` and `metrics.json` report actual
checkpoint bytes, unique sketch bytes, Base, norms, metadata, GPU resident
bytes, bits per routed weight, and compression against routed BF16 weights.
`accuracy.json` contains evaluation task results.

Independent diagnostics run separately:

```bash
.venv/bin/python scripts/QJL/analyze_residual_bundle.py \
  --model-path "$MODEL" \
  --checkpoint results/qjl_residual_bundle/majority_d2n_similarity/compact_qjl_state.pt \
  --output-dir results/qjl_residual_bundle/majority_d2n_similarity/analysis
```

This writes expert, pair, projection, and layer CSV files plus aggregate JSON.
It uses seeded synthetic inputs by default. `--activation-path` accepts a
torch file containing `layer -> projection -> expert -> routed input tensor`.
The analysis compares individual QJL, majority, and magnitude-aware on the
same inputs, and uses the checkpoint's actual pair map and shared sketches.

Run `uv pip install --python .venv/bin/python pytest` to install the test runner
into `.venv`. Validation on the remote
RTX PRO 5000 Blackwell: `.venv/bin/python -m pytest
tests/test_residual_bundle.py -q` yielded **8 passed** and the existing
`tests/test_exponent_mantissa_base_qjl.py` yielded **4 passed**. The local CPU
run yielded **7 passed, 1 skipped** (Triton requires CUDA). A real Qwen layer 0
smoke run completed compression, checkpoint reload, one `arc_easy` question,
and independent analysis of one saved pair. Full-model accuracy is reported
only after the separate evaluation jobs finish.

## Tests and attribution

```bash
PYTHONPATH=. .venv/bin/python tests/test_codebook_base_qjl.py
PYTHONPATH=. .venv/bin/python tests/test_exponent_mantissa_base_qjl.py
```

The source code includes work derived from ResCode and its dependencies.
Retain the included [MIT license](LICENSE) and any file-level notices when
redistributing it. The original documentation lists additional upstream
attributions.
