# GS-MoE gain grouping + shared projection + residual SVD

This experiment targets `Qwen/Qwen1.5-MoE-A2.7B-Chat` on the `grouping-gsmoe` branch. It adapts the reconstruction-gain objective and fixed-size balanced K-medoids from `/home/jaewoo/GS-MoE/src/compression/get_shared_experts.py`, and the shared projection/coefficient equations from its `compress.py`. It does not modify or import GS-MoE. It does not run QJL, neuron pruning, or training.

## Decomposition

For Gate/Up, each group stores one input basis `P_g` and each expert stores `A_i`:

`W_i ≈ A_i P_gᵀ + U_i V_i`.

For Down, each group stores one output basis `P_g` and each expert stores `B_i`:

`W_i ≈ P_g B_i + U_i V_i`.

The `P_g` bases are refit from group-pooled routed calibration statistics after grouping. Gate/Up use the leading eigenspace of `sqrt(C_x) (Σ W_iᵀW_i) sqrt(C_x)`, mapped back through `C_x^{-1/2}` and QR. Down uses the leading eigenspace of `Σ Y_down,iᵀY_down,i`. Per-expert Gate/Up coefficients solve the covariance-weighted least-squares problem; Down uses `P_gᵀW_i`.

The FP32 residual is computed **after reconstructing the stored BF16 shared basis and coefficient**. Its FP32 SVD is sliced at each residual rank. Residual rank 0 stores no `U_i,V_i` and is the shared projection only baseline. Every checkpoint factor is BF16. Routed output diagnostics use the checkpoint's BF16 factor matmuls and the original BF16 expert weights; FP32 SVD energy remains a separate measurement. Checkpoint installation retains the original router, top-k selection, shared expert, shared expert gate, and Qwen SwiGLU operation.

For singleton probe `P_j`, grouping uses the GS-MoE gain

`J_i(P_j) = f_i [J_up,i(P_j) + J_gate,i(P_j) + β J_down,i(P_j)]`,

where `f_i` is the routed token count. Input projection gain is `tr((Pᵀ C_i P)⁻¹ Pᵀ C_i W_iᵀ W_i C_i P)` and Down gain is `tr(Pᵀ A_i P)` with `A_i=Y_down,iᵀY_down,i/N_i`. The distance is `D_ij = 1 - clamp(J_i(P_j)/J_i(P_i), 0, 1)`. Fixed-size Hungarian assignment and medoid updates create balanced groups. The default 60 experts / 30 groups produces exactly two experts per group. `β=1` follows GS-MoE's default. A relative covariance ridge of `1e-4` stabilizes the empirical covariance, which can be rank-deficient with capped calibration samples; this is configurable.

## Setup and quick checks

Run from `/home/jaewoo/QJLdecomposition` with the repository `.venv` active. The script needs the existing repo dependencies plus `scipy` and `matplotlib`.

```bash
git switch grouping-gsmoe
bash scripts/QJL/run_grouping_gsmoe.sh test
bash scripts/QJL/run_grouping_gsmoe.sh smoke
```

The smoke command analyzes Layer 12, Expert 0–1, one group, shared rank 4, and residual ranks 0/2. It uses the existing routed activation cache at `results/shared_lowrank_qjl/full/analysis/grouping_cache.pt`; pass `--calibration-cache` with another file or directory if needed. The QJL cache has capped samples, so its `routing_frequency` is a sample-count proxy. A full study should use the fresh calibration collected by this script.

## Full analyze → compress → eval

**These commands are for manual execution.** `analyze` and `compress` run the full 24-layer/60-expert decomposition; `eval` runs full ARC-Easy and ARC-Challenge when `--eval-limit 0`.

```bash
cd /home/jaewoo/QJLdecomposition

bash scripts/QJL/run_grouping_gsmoe.sh analyze \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers all --num-groups 30 --probe-rank 64 \
  --shared-ranks 64,128,256 --residual-ranks 0,32,64,128,256 \
  --calibration-samples 256 --calibration-blocks 32 \
  --output-dir results/grouping_gsmoe

bash scripts/QJL/run_grouping_gsmoe.sh compress \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers all --num-groups 30 --probe-rank 64 \
  --shared-ranks 64,128,256 --residual-ranks 0,32,64,128,256 \
  --output-dir results/grouping_gsmoe

bash scripts/QJL/run_grouping_gsmoe.sh eval \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --checkpoint results/grouping_gsmoe/checkpoints/shared_128/residual_64/manifest.json \
  --tasks arc_easy,arc_challenge --eval-limit 0 \
  --output-dir results/grouping_gsmoe
```

Repeat `eval` with each `shared_{rank}/residual_{rank}` manifest. Its output is saved independently under `results/grouping_gsmoe/eval/shared_{rank}/residual_{rank}/`. `--tasks` accepts the lm-eval task names supported by `rescode.evaluation`; `--eval-limit 0` means no limit. No full evaluation runs automatically from `analyze` or `compress`.

`analyze` automatically downloads the standard C4 JSON shard if absent, runs one calibration pass through the original Qwen model, and saves routed Gate/Up inputs, Down intermediate activations, and **uncapped routing counts** in `results/grouping_gsmoe/calibration/layer_*.pt`. The calibration pass and decomposition operate layer-wise after collection. `compress` reuses these layer files and `results/grouping_gsmoe/grouping/layer_*.pt`, which contain the balanced labels and maximum-rank group bases. To use an existing routed-sample cache, pass `--calibration-cache PATH`; to reuse grouping from another output root, pass `--grouping-cache-dir PATH`. The latter checks the layer, expert selection, ranks, grouping settings, and seed before reuse. Keep the same calibration data when reusing it.

The full `compress` sweep writes 15 independent checkpoints and can consume substantial disk space. Each checkpoint includes the shared `P_g` once per group, expert coefficients, and residual factors. The original HF backbone, router, and shared expert are loaded separately for evaluation and are excluded from expert-only bpw.

## Outputs and metrics

`results/grouping_gsmoe/analysis/` contains `per_projection.csv`, `per_expert_swiglu.csv`, `groups.csv`, `routing_frequency.csv`, projection and SwiGLU summary CSVs, `summary.json`, `REPORT.md`, and graphs. `compress` adds `checkpoint_storage.csv` and manifests. Per-expert rows distinguish FP32 oracle SVD energy from stored BF16 factor error. Summary CSVs contain **expert means** and **pooled squared-norm** metrics separately. Gate/Up/Down outputs use each expert's routed inputs; full SwiGLU output runs the reconstructed Gate, Up, and Down paths together on routed hidden states.

Calibration-aware projection optimizes routed output reconstruction. Its unweighted Frobenius residual norm `||R||_F/||W||_F` can therefore exceed 1 even when routed output error improves.

## Checks run during implementation

- Toy tests: reconstruction gain matches the direct GS-MoE trace formula; balanced assignment has no duplicates; shared coefficient reconstruction and rank-0/full-rank residual boundaries; FP32 SVD energy and monotone error; BF16 checkpoint save/reload; router/shared expert identity and finite SwiGLU output.
- Real Qwen Layer 12 smoke: 2 experts, one group, shared rank 4, residual ranks 0/2. Both analyze and compress completed using the prior routed cache. Installing both checkpoints in the original Qwen model preserved router output exactly and produced finite `(1, 2, 2048)` block outputs.
- Fresh C4 calibration smoke: Layer 12, 4 blocks × 256 tokens, sample cap 8. The cache recorded 62 actual routed tokens for each selected expert while retaining 8 diagnostic samples, and analyze completed with `frequency_source=full_routing_count`.

The full 24-layer/60-expert analyze, compress sweep, and lm-eval are intentionally left for manual execution. The full sweep has not been timed or memory-profiled. The prior QJL cache is useful for smoke tests but has only capped routed samples and sample-count proxy frequencies; collect fresh calibration for reported full-scale results.
