# GS-MoE grouping with full routed covariance

This independent Qwen1.5-MoE-A2.7B-Chat experiment uses GS-MoE reconstruction gain, balanced grouping, group shared projections, and expert residual SVD. It does not run QJL, pruning, or training. The new experiment root is `results/grouping_gsmoe/full_cov/`; earlier `results/grouping_gsmoe/` analyses and checkpoints remain separate.

## Calibration and decomposition

For every original-router routed token of expert `i`, calibration accumulates FP32 matrices. As in `/home/jaewoo/GS-MoE/src/calibration.py` and its Qwen2 adapter, each routed activation and Down output is multiplied by the square root of its normalized top-k routing probability before forming the Gram matrix. Each sum is divided by the **uncapped routed token count** `N_i`:

`C_x,i = Σ_t q_it x_t x_tᵀ / N_i` and `C_down,i = Σ_t q_it y_down,it y_down,itᵀ / N_i`.

The expert routing frequency remains `N_i`. `--calibration-samples` limits only the saved routed Gate/Up and Down inputs used for output diagnostics. Grouping gain, final `P_g`, and Gate/Up coefficients use the full covariances. Gate/Up weight Grams, balanced K-medoids, and the residual SVD retain the existing equations. The reconstruction is `A_i P_gᵀ + U_i V_i` for Gate/Up and `P_g B_i + U_i V_i` for Down. All checkpoint factors are BF16; residual SVD runs in FP32 on the residual after stored BF16 shared factors have been reconstructed.

Calibration processes one target layer at a time, reusing one loaded model and deterministic C4 calibration blocks, so only one layer's covariance accumulators reside on the GPU. This repeats the model forward for each uncached layer. The layer cache contains diagnostic samples, full FP32 covariance matrices, uncapped route counts, method name, dataset path and SHA-256, model config SHA-256, and calibration settings. A sidecar records the layer file size and settings. Incompatible old sample-only caches or changed settings are archived with a `.stale-N` suffix before recollection; grouping caches with changed calibration signatures are also archived before recomputation. Cache files can be large (roughly 2 GiB per layer for 60 experts and 2048-dimensional input/output covariance).

`compress` automatically performs the same diagnostics as `analyze`, while saving every checkpoint. Under `full_cov/analysis/`, the CSV, JSON, graphs, and `REPORT.md` include shared residual norms, FP32 residual SVD energy and tail norms, BF16 Gate/Up/Down routed output errors, full SwiGLU output errors, actual checkpoint bytes, effective expert-only bits/weight, and BF16 compression ratio. Summary rows distinguish expert means from pooled squared-norm ratios. The original router, shared expert, shared-expert gate, top-k routing, and SwiGLU remain in the HF model used for checkpoint evaluation.

## Checks and small smoke test

Run from `/home/jaewoo/QJLdecomposition`:

```bash
git switch grouping-gsmoe
bash scripts/QJL/run_grouping_gsmoe.sh test
bash scripts/QJL/run_grouping_gsmoe.sh smoke
```

The smoke command collects full routed covariance for Layer 12 using four short C4 calibration blocks, then compresses Experts 0–1 with shared rank 4 and residual ranks 0/2. It writes to `results/grouping_gsmoe/full_cov_smoke/`. Its small sample setting checks mechanics, not model quality.

On the current instance, the 10 toy/held-out tests passed. The real Layer 12 compress smoke completed: the 60 experts accumulated 24–116 routed tokens each while only eight diagnostic inputs per expert were saved; selected Experts 0/1 accumulated 61/62 tokens. Two BF16 checkpoints and 12 projection-diagnostic rows were written. A separate C4 shard `00001` smoke evaluation on the same saved factors reproduced the compress Train metrics with maximum absolute difference 0. The two-expert, rank-4 SwiGLU pooled Relative L2 was 0.64295 (Train) versus 0.99914 (Held-out) at residual rank 0, and 0.63953 versus 1.00000 at residual rank 2. These tiny ranks are for code validation only. The full 24-layer compress and full lm-eval were not run for this implementation.

## A. Full 24-layer compress with automatic diagnostics

This is a **manual, large run**. It collects new full-covariance caches into `full_cov/calibration/`, computes grouping/bases into `full_cov/grouping/`, saves 15 checkpoint combinations under `full_cov/checkpoints/`, and writes diagnostics to `full_cov/analysis/`.

Check available storage before starting. At 60 experts and hidden size 2048, the 24-layer FP32 covariance cache is about 45 GiB before metadata and diagnostic samples. A previous 15-combination checkpoint sweep occupied about 84 GiB. Plan for at least 130 GiB of free space plus a safety margin, or place `results/grouping_gsmoe/full_cov/` on a larger volume. Keep earlier results separate.

```bash
cd /home/jaewoo/QJLdecomposition
bash scripts/QJL/run_grouping_gsmoe.sh compress \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers all --num-groups 30 --probe-rank 64 \
  --shared-ranks 64,128,256 --residual-ranks 0,32,64,128,256 \
  --calibration-blocks 32 --calibration-seqlen 2048 \
  --calibration-batch-size 2 --calibration-samples 256 \
  --output-dir results/grouping_gsmoe/full_cov
```

Use `--mode analyze` through the same wrapper to generate diagnostics without checkpoint files. The full compress run already generates them.

## B. Separate held-out routed-output evaluation

After A, collect new C4 shard `00001` routed inputs once. The collection checks exact text-document overlap against training shard `00000` and saves activations outside Git. Held-out inputs are never used to fit grouping, shared projections, coefficients, or SVD. `evaluate` loads the saved BF16 checkpoints and compares the same factors on train and held-out inputs for Layers 0, 12, and 23.

```bash
cd /home/jaewoo/QJLdecomposition
.venv/bin/python scripts/QJL/evaluate_grouping_gsmoe_heldout.py \
  --mode collect --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers 0,12,23 --seed 31415 --calibration-blocks 32 \
  --calibration-seqlen 2048 --calibration-batch-size 2 \
  --calibration-samples 256 --min-heldout-samples 256

.venv/bin/python scripts/QJL/evaluate_grouping_gsmoe_heldout.py \
  --mode evaluate --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers 0,12,23 --shared-ranks 64,128,256 \
  --residual-ranks 0,32,64,128,256 --min-heldout-samples 256
```

Run `collect` only when the held-out cache is absent. `evaluate` is repeatable with the existing cache. Its output is `full_cov/heldout_layers_0_12_23/`, with `per_projection.csv`, `per_expert_swiglu.csv`, layer summaries, `train_vs_heldout.csv`, `summary.json`, factor/cache hashes, provenance, and `REPORT.md`. The full 24-layer compress must finish before the 15 rank combinations for these layers are available. Earlier held-out results under `results/grouping_gsmoe/heldout_layers_0_12_23/` describe the old 256-sample-fitted checkpoints, not this method.

## C. Full ARC-Easy and ARC-Challenge evaluation for every checkpoint

This loop is for manual execution after A and B. `--eval-limit 0` evaluates every example in each task. Each rank combination writes independently to `full_cov/eval/shared_{rank}/residual_{rank}/`.

```bash
cd /home/jaewoo/QJLdecomposition
for shared_rank in 64 128 256; do
  for residual_rank in 0 32 64 128 256; do
    bash scripts/QJL/run_grouping_gsmoe.sh eval \
      --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
      --checkpoint "results/grouping_gsmoe/full_cov/checkpoints/shared_${shared_rank}/residual_${residual_rank}/manifest.json" \
      --tasks arc_easy,arc_challenge --eval-limit 0 \
      --output-dir results/grouping_gsmoe/full_cov
  done
done
```

## D. Evaluate one selected checkpoint

Change the two rank variables to any pair produced by A. This command also uses the complete task sets; `--tasks` and `--eval-limit` can be changed for a smaller manual check.

```bash
cd /home/jaewoo/QJLdecomposition
shared_rank=256
residual_rank=64
bash scripts/QJL/run_grouping_gsmoe.sh eval \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --checkpoint "results/grouping_gsmoe/full_cov/checkpoints/shared_${shared_rank}/residual_${residual_rank}/manifest.json" \
  --tasks arc_easy,arc_challenge --eval-limit 0 \
  --output-dir results/grouping_gsmoe/full_cov
```

The held-out Train cache does not record individual source document IDs; the shard comparison proves exact source text separation but cannot establish the identity of each older cached document. The routed-output checks do not measure task accuracy. The 24-layer compress and full lm-eval commands above are supplied for the user and are not run as part of the small implementation smoke test.
