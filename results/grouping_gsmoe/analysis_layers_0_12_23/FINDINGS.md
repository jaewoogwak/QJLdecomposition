# Layers 0, 12, 23: GS-MoE grouping + residual SVD

This is a snapshot of `results/grouping_gsmoe/analysis/` after the three-layer `analyze` run. It contains 60 experts per layer, 30 balanced groups of two per layer, Gate/Up/Down, probe rank 64, shared ranks 64/128/256, and residual ranks 0/32/64/128/256. Routed inputs came from the fresh C4 calibration cache with uncapped routing counts; each expert contributed 256 saved diagnostic samples.

## Full SwiGLU expert output

Pooled relative L2 error uses the sum of squared output errors divided by the sum of squared original output norms across the 180 experts. These are BF16 checkpoint-factor output diagnostics on routed activations, not lm-eval scores.

| Shared rank | Residual 0 | Residual 32 | Residual 64 | Residual 128 | Residual 256 |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.58441 | 0.56926 | 0.55591 | 0.53003 | 0.47697 |
| 128 | 0.43286 | 0.42223 | 0.41179 | 0.39130 | 0.34975 |
| 256 | 0.22185 | 0.21411 | 0.20739 | 0.19461 | **0.17057** |

At shared rank 256, adding residual rank 256 lowers pooled error from 0.22185 to 0.17057. Mean expert-output cosine rises from 0.96381 to 0.97901. The rank-0 column is the shared projection only baseline.

| Layer | Shared 256, residual 0 | Shared 256, residual 256 |
|---:|---:|---:|
| 0 | 0.17546 | 0.13062 |
| 12 | 0.31542 | 0.24355 |
| 23 | 0.21735 | 0.16706 |

Layer 12 has the largest pooled output error among the three at this setting.

## Projection diagnostics at shared rank 256

| Projection | Pooled shared residual `R/W` | Output L2, residual 0 | Output L2, residual 256 | Stored tail `E/W`, residual 256 |
|---|---:|---:|---:|---:|
| Gate | 1.48913 | 0.02610 | 0.00845 | 0.60966 |
| Up | 1.51989 | 0.02917 | 0.00905 | 0.62732 |
| Down | 0.86119 | 0.22087 | 0.17029 | 0.59637 |

Gate/Up `R/W` exceeds 1 because their covariance-aware shared projection targets routed output error, not unweighted Frobenius error. Their routed output error is nevertheless low at shared rank 256. Down remains the largest projection output error.

## Validation and limits

- `per_projection.csv`: 8,100 rows; `per_expert_swiglu.csv`: 2,700 rows; `groups.csv`: 90 groups; `routing_frequency.csv`: 180 expert entries.
- Every layer partitions experts 0–59 exactly once into 30 groups of two. All recorded numeric diagnostics are finite. For every expert/projection/shared-rank combination, FP32 oracle tail squared norm is nonincreasing over residual rank.
- Actual checkpoint bytes and compression ratio are unavailable from `analyze` mode. Full-model compression and lm-eval are separate runs; no ARC accuracy claim follows from these diagnostics.
