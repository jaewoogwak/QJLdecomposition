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

The existing D=2n checkpoints can be evaluated separately with an explicit
checkpoint path for each method:

```bash
bash scripts/QJL/run_residual_bundle_sweep.sh --mode eval \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --bundle-method majority --pairing similarity --qjl-d-factor 2 \
  --checkpoint results/qjl_residual_bundle/d2n_similarity/majority_d2n_similarity/compact_qjl_state.pt \
  --output-dir results/qjl_residual_bundle/d2n_similarity/majority_d2n_similarity \
  --eval-batch-size 1

bash scripts/QJL/run_residual_bundle_sweep.sh --mode eval \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --bundle-method magnitude-aware --pairing similarity --qjl-d-factor 2 \
  --checkpoint results/qjl_residual_bundle/d2n_similarity/magnitude-aware_d2n_similarity/compact_qjl_state.pt \
  --output-dir results/qjl_residual_bundle/d2n_similarity/magnitude-aware_d2n_similarity \
  --eval-batch-size 1
```

Use `--mode compress` for compression only or `--mode both` to compress and
evaluate one configuration. Single-configuration calls require an explicit
`--checkpoint`. For the complete six-configuration sweep:

```bash
bash scripts/QJL/run_residual_bundle_sweep.sh --all --mode both \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --output-dir results/qjl_residual_bundle/sweep
```

`--task arc_easy,arc_challenge` limits evaluation to named tasks; the default
is winogrande, arc_easy, arc_challenge, boolq, hellaswag, mmlu, openbookqa,
and rte. `--eval-limit 0` evaluates all questions. The script also supports
`--bundle-method none`, `--pairing random`, and independent projection,
pairing, and bundle seeds. To run the six standard bit-budget configurations,
use `--all` as shown above. `--mode eval` always uses the specified checkpoint
and checks that its saved method, pairing, and D factor match the options.
The new evaluator defaults to batch size 1 because full eight-task evaluation
at batch size 16 exhausted the 48 GB RTX PRO 5000 GPU during FP32 logits
allocation. Use `--eval-batch-size` to override.

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
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --checkpoint results/qjl_residual_bundle/d2n_similarity/majority_d2n_similarity/compact_qjl_state.pt \
  --output-dir results/qjl_residual_bundle/d2n_similarity/majority_d2n_similarity/analysis
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

## Individual QJL residual scale ablation

The existing Individual QJL checkpoint can be evaluated with a uniform
residual scale without recompression. The evaluator loads the checkpoint
afresh, multiplies the routed experts' Gate, Up, and Down residual row norms
by `--residual-scale`, and leaves the Exponent-Mantissa Base, sketches,
projection matrix, and router unchanged. A scale of `0` evaluates Base only;
`1` reproduces the original Individual QJL model.

```bash
.venv/bin/python scripts/QJL/run_codebook_base.py \
  --mode eval \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --checkpoint results/qjl_exponent_mantissa_base_2d/rerun/compact_qjl_state.pt \
  --output-dir results/qjl_residual_alpha_sweep/d2n/alpha_0.5 \
  --backend triton --task arc_easy,arc_challenge \
  --eval-batch-size 64 --eval-limit 0 --residual-scale 0.5
```

Run all five scales with a fresh checkpoint load for each evaluation:

```bash
bash scripts/QJL/run_residual_alpha_sweep.sh
```

The sweep defaults to `0.0,0.25,0.50,0.75,1.0`, D=2n Individual QJL,
zero-shot ARC Easy and Challenge, batch size 64, and full evaluation.
`MODEL_PATH`, `CHECKPOINT`, `ALPHAS`, `TASKS`, `EVAL_BATCH_SIZE`, `EVAL_LIMIT`,
and `OUTPUT_DIR` override the corresponding defaults. Set `TASKS` to
`winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte` for
the original eight-task set. Each `alpha_{value}` directory stores the
evaluator's accuracy and metrics plus an `eval.log`; `summary.csv` and
`summary.json` in the sweep output directory contain task accuracy,
normalized accuracy, mean accuracy, and evaluation time.

On the remote RTX PRO 5000, `pytest tests/test_residual_alpha.py
tests/test_exponent_mantissa_base_qjl.py -q` passed **7 tests**, including
Reference/Triton and fused Gate/Up checks. A smoke sweep with one question
per ARC task completed all five scales and wrote both summary files under
`results/qjl_residual_alpha_sweep/smoke_d2n/`. The original checkpoint's
SHA-256 remained unchanged. The full zero-shot ARC Easy and Challenge sweep
completed at batch size 64, with the same D=2n Individual checkpoint loaded
separately for each scale:

| α | ARC Easy acc | ARC Challenge acc | Mean acc | Mean acc_norm |
|---:|---:|---:|---:|---:|
| 0.00 | 0.7045 | 0.4036 | 0.5541 | 0.5485 |
| 0.25 | 0.6995 | 0.4027 | 0.5511 | 0.5499 |
| 0.50 | 0.6869 | 0.3831 | 0.5350 | 0.5381 |
| 0.75 | 0.6734 | 0.3695 | 0.5214 | 0.5252 |
| 1.00 | 0.6494 | 0.3635 | 0.5064 | 0.5102 |

Per-task normalized accuracy, evaluation time, and full-precision values
are in `results/qjl_residual_alpha_sweep/d2n/summary.csv` and `summary.json`.

## Pairwise QJL Base + low-rank residual (`base-bundling` branch)

This experiment pairs all 60 routed experts independently for Gate, Up, and
Down in each selected layer. Maximum-weight Hamming matching chooses 30
disjoint pairs per projection. A seeded majority vote produces one packed
sketch per pair, while FP32 row norms remain expert-specific. The effective
QJL Base uses the stored shared sketch; the FP32 SVD residual is computed
from the original weight minus that Base. Rank factors are saved as BF16.

Run all 24 layers and full zero-shot ARC Easy/Challenge evaluation:

```bash
bash scripts/QJL/run_base_bundling.sh
```

The script supports `MODEL_PATH`, `QJL_D_FACTOR`, `RANK`, `TASKS`,
`EVAL_BATCH_SIZE`, `EVAL_LIMIT`, `LAYERS`, `OUTPUT_DIR`, and `CASES`.
Defaults are D=2n, rank 512, batch 16, full ARC, and cases
`original,A,B`. Set `TASKS` to
`winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte`
for all eight benchmarks. Compression and evaluation can also be run
separately; `--resume` skips completed layer shards:

```bash
.venv/bin/python scripts/QJL/run_base_bundling.py --mode compress \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers all --qjl-d-factor 2 --rank 512 --resume \
  --output-dir results/base_bundling/pairwise_d2n_r512

.venv/bin/python scripts/QJL/run_base_bundling.py --mode eval --case A \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --output-dir results/base_bundling/pairwise_d2n_r512 \
  --backend triton --task arc_easy,arc_challenge \
  --eval-limit 0 --eval-batch-size 16
```

Use `--case original` or `--case B` for the other evaluations. Checkpoint
`base_layer_N.pt` stores pair maps, one packed sketch per pair, and individual
norms. `rank_layer_N.pt` stores only BF16 factor matrices and is loaded only
for B. This gives A and B identical Base tensors. Full singular spectra and
per-expert metrics are stored separately as `spectrum_layer_N.npz` and
`per_expert_layer_N.csv`; no exact residual matrix is checkpointed. The
output directory also contains aggregate `per_expert.csv`, plots,
`storage_A.json`, `storage_B.json`, `summary.csv`, `summary.json`,
`REPORT.md`, and each case's `eval_metrics.json` after evaluation.

Unit tests: `.venv/bin/python -m pytest tests/test_base_bundle_lowrank.py
tests/test_residual_bundle.py tests/test_residual_alpha.py -q` passed **14
tests** on the remote GPU, including Triton parity, unique sketch storage,
and compatibility with the existing experiments. A layer-12 smoke run
compressed all 180 routed matrices in that layer, wrote full spectra and
both plots, then reloaded Original/A/B
for one ARC Easy and one ARC Challenge question each. Both compressed
cases used the same 64,880,640 sketch bytes and 1,167,360 norm bytes;
only B loaded 637,009,920 BF16 factor bytes. Case A measured 1.019 BPW
and B 10.839 BPW for routed weights in the selected layer. Those two
questions are an execution check, not an accuracy estimate. Full 24-layer
compression and full ARC evaluation have not been run for this experiment.

## Tests and attribution

```bash
PYTHONPATH=. .venv/bin/python tests/test_codebook_base_qjl.py
PYTHONPATH=. .venv/bin/python tests/test_exponent_mantissa_base_qjl.py
```

The source code includes work derived from ResCode and its dependencies.
Retain the included [MIT license](LICENSE) and any file-level notices when
redistributing it. The original documentation lists additional upstream
attributions.
