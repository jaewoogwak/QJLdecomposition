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
`EVAL_BATCH_SIZE`, `EVAL_LIMIT`, `LAYERS`, `OUTPUT_DIR`, and `EVAL_VARIANTS`.
Defaults are D=2n, rank 512, batch 16, full ARC, and evaluation variants
`original,base-only,base-plus-lowrank`. Set `TASKS` to
`winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte`
for all eight benchmarks. Compression and evaluation can also be run
separately; `--resume` skips completed layer shards:

```bash
.venv/bin/python scripts/QJL/run_base_bundling.py --mode compress \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers all --qjl-d-factor 2 --rank 512 --resume \
  --output-dir results/base_bundling/pairwise_d2n_r512

.venv/bin/python scripts/QJL/run_base_bundling.py \
  --mode eval --eval-variant base-plus-lowrank \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --output-dir results/base_bundling/pairwise_d2n_r512 \
  --backend triton --task arc_easy,arc_challenge \
  --eval-limit 0 --eval-batch-size 16
```

Use `--eval-variant original` or `--eval-variant base-only` for the other
evaluations. The older `--case original/A/B` and `CASES` names remain accepted.
Result directories retain their original `original`, `A`, and `B` names.
Checkpoint `base_layer_N.pt` stores pair maps, one packed sketch per pair,
and individual norms. `rank_layer_N.pt` stores only BF16 factor matrices and
is loaded only for B. This gives A and B identical Base tensors. Full singular spectra and
per-expert metrics are stored separately as `spectrum_layer_N.npz` and
`per_expert_layer_N.csv`; no exact residual matrix is checkpointed. The
output directory also contains aggregate `per_expert.csv`, plots,
`storage_A.json`, `storage_B.json`, `summary.csv`, `summary.json`,
`REPORT.md`, and each case's `eval_metrics.json` after evaluation.

Unit tests: `.venv/bin/python -m pytest tests/test_base_bundle_lowrank.py
tests/test_residual_bundle.py tests/test_residual_alpha.py -q` passed **16
tests** on the remote GPU, including Triton parity, unique sketch storage,
and compatibility with the existing experiments. A layer-12 smoke run
compressed all 180 routed matrices in that layer, wrote full spectra and
both plots, then reloaded Original/A/B for one ARC Easy and one ARC
Challenge question each. Both compressed cases used the same 64,880,640
sketch bytes and 1,167,360 norm bytes;
only B loaded 637,009,920 BF16 factor bytes. Case A measured 1.019 BPW
and B 10.839 BPW for routed weights in the selected layer. Those two
questions are an execution check, not an accuracy estimate.

The full 24-layer, 4,320-matrix compression and zero-shot ARC Easy/Challenge
evaluation later completed with batch size 16:

| Case | ARC Easy acc | ARC Challenge acc | Mean acc | Mean acc_norm | Routed BPW |
|---|---:|---:|---:|---:|---:|
| Original BF16 | 0.7020 | 0.4002 | 0.5511 | 0.5465 | 16.000 |
| A: Bundled Base | 0.2673 | 0.1988 | 0.2330 | 0.2633 | 1.019 |
| B: Base + rank-512 | 0.4398 | 0.2679 | 0.3539 | 0.3521 | 10.839 |

Full results are in `results/base_bundling/pairwise_d2n_r512/`, including
the complete singular spectra, per-expert metrics, plots, and storage and
evaluation summaries. A/B checkpoint sizes count compressed routed layers;
the Original checkpoint size counts all model safetensors. Routed BPW uses
the same weight count for all three cases.

A separate full run kept D=2n and evaluated only B with rank 1024. ARC Easy
acc was **0.6881**, ARC Challenge acc **0.3899**, mean acc **0.5390**, and
mean acc_norm **0.5197**. Its routed storage was **20.657 BPW**, about 29.1%
larger than BF16 routed weights. Results are in
`results/base_bundling/pairwise_d2n_r1024/`; the original and Base-only
evaluations were not repeated for this rank.

## Activation-aware Base bundling analysis

`scripts/QJL/analyze_base_bundling.py` is an independent analysis of the
original model. It does not build a compressed model or run lm-eval. Input
files are disjoint JSONL files with one `{"text": "..."}` record per
sequence. The calibration pass uses every routed token except padding and
saves cumulative 128/256/512-sequence snapshots; the held-out pass is
separate. The router output, actual Gate input, and actual Down input are
hooked during the original forward. Gate and Up share one covariance.

On the analysis server, 640 Wikipedia documents have been prepared as
`rescode/data/wikipedia-analysis-calibration.jsonl` (first 512) and
`rescode/data/wikipedia-analysis-heldout.jsonl` (remaining 128). These are
the shell wrapper defaults, so the staged run is:

```bash
for stage in calibrate similarity base spectrum report; do
  STAGE="$stage" bash scripts/QJL/run_base_bundling_analysis.sh
done
```

To use another corpus, supply `CALIBRATION_JSONL` and `HELDOUT_JSONL` as
separate files. The repository's `scripts/RESCODE/prepare_calib_datasets.py`
can download a Wikipedia JSONL source; split its records without overlap
before running this analysis.

```bash
COMMON="--model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --calibration-jsonl /path/to/calibration.jsonl \
  --heldout-jsonl /path/to/heldout.jsonl \
  --output-dir results/base_bundling_analysis --layers 12 --resume"
.venv/bin/python scripts/QJL/analyze_base_bundling.py --stage calibrate $COMMON
.venv/bin/python scripts/QJL/analyze_base_bundling.py --stage similarity $COMMON
.venv/bin/python scripts/QJL/analyze_base_bundling.py --stage base $COMMON
.venv/bin/python scripts/QJL/analyze_base_bundling.py --stage spectrum $COMMON \
  --spectrum-scope representative --representative-count 5
.venv/bin/python scripts/QJL/analyze_base_bundling.py --stage report $COMMON
```

For the complete all-pair randomized spectrum analysis, run:

```bash
.venv/bin/python scripts/QJL/analyze_base_bundling.py --stage all \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --calibration-jsonl /path/to/calibration.jsonl \
  --heldout-jsonl /path/to/heldout.jsonl \
  --output-dir results/base_bundling_analysis_all \
  --layers 12 --projections gate,up,down \
  --calibration-sizes 128,256,512 --heldout-size 128 \
  --input-ranks 32,64,128,256,512 \
  --residual-ranks 0,32,64,128,256,352 \
  --qjl-d-factor 2 --spectrum-scope all --resume
```

The shell wrapper accepts the same options as environment variables, for
example `STAGE=all CALIBRATION_JSONL=/path/to/cal.jsonl
HELDOUT_JSONL=/path/to/held.jsonl bash scripts/QJL/run_base_bundling_analysis.sh`.
CLI options passed after the shell script override its defaults.

Cases A/B are dense sharing (arithmetic mean and regularized covariance
optimum), C2/C1 are individual binary D=2n/D=n, D binarizes the covariance
optimum, E is majority voting with original row norms, F optimizes the
majority row scales, G uses magnitude-aware voting and optimized scales, and
H uses `sign(W_i C_i G + W_j C_j G)` with optimized scales. The C1 projection
uses the first n columns of the same Gaussian matrix as C2. Covariance
errors use `Tr[(W−B)C(W−B)ᵀ] / Tr[WCWᵀ]`; row scales minimize the same
quadratic independently. Case B uses a positive trace-scaled ridge to make
the solve stable. Activation-aware low-rank factors use an inverse square
root with the configured ridge. BF16 factor errors are measured after
factor quantization. Randomized singular energy is labeled as an estimate;
`--exact-representative` adds exact SVD for selected pairs only.

`config.json` records seeds, data hashes, layers, ranks, and analysis
settings. `cache/` contains routed moments and per-stage CSV files; calibration
resumes from the most recent snapshot, and numeric stages resume after each
completed pair. The top-level outputs are `pairwise_metrics.csv`, `base_metrics.csv`,
`residual_metrics.csv`, `grouping_summary.csv`, `correlation.csv`,
`summary.json`, `REPORT.md`, `stage_runtime.json`, spectrum NPZ files, and
plots. In representative spectrum mode, joint-cost matching is explicitly
candidate-restricted and includes a seeded random perfect matching as a
feasibility control. Held-out statistics are used only after matching.

The Base stage caches each expert's `WC` and weight energy, then evaluates
error and cosine from shared quadratic terms. The spectrum stage computes
all rank-prefix covariance errors from low-rank Gram matrices; it does not
repeat a full weight-by-covariance product for every rank. Individual C1/C2
residual spectra are reused across pairs. Base and spectrum logs print
pair-level progress and an ETA. These optimizations preserve the FP32
covariance definitions and BF16 factor application used by the analysis.

Minimal verification on the server: `tests/test_base_bundle_analysis.py`
passed **5 synthetic tests**, including hook padding and snapshot checks,
covariance errors, binary ties, SVD and matching, and CSV/report generation.
The GPU path also checks seeded majority/magnitude-aware ties and the
complete synthetic Base A-H construction. Together with the existing
residual-bundle tests, **13 tests** passed after the GPU tie-device fix.
`analyze_base_bundling.py --help` and Python compilation also passed. The
actual Qwen calibration, all-pair analysis, SVD sweep, and model evaluation
were intentionally not run for this analysis task.

The later layer-12 representative-pair run is published under
`results/base_bundling_analysis/`. Its CSV, JSON, plots, and report are
versioned. The 22,992 individual spectrum NPZ files are stored in nine
`spectra_archives/*.tar.gz` archives (one per calibration size and projection);
their hashes and member counts are in `spectra_archives/MANIFEST.json`. To
restore the paths referenced by `residual_metrics.csv`, run:

```bash
mkdir -p results/base_bundling_analysis/spectra
for archive in results/base_bundling_analysis/spectra_archives/*.tar.gz; do
  tar -xzf "$archive" -C results/base_bundling_analysis/spectra
done
```

The 5.7 GB routed covariance cache and source Wikipedia JSONL files are not
versioned. The report and tabular metrics do not require them.

## Tests and attribution

```bash
PYTHONPATH=. .venv/bin/python tests/test_codebook_base_qjl.py
PYTHONPATH=. .venv/bin/python tests/test_exponent_mantissa_base_qjl.py
```

The source code includes work derived from ResCode and its dependencies.
Retain the included [MIT license](LICENSE) and any file-level notices when
redistributing it. The original documentation lists additional upstream
attributions.
