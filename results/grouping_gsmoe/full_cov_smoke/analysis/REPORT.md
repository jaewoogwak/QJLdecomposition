# GS-MoE gain grouping + shared projection + residual SVD

This experiment adapts GS-MoE's reconstruction gain and fixed-size balanced K-medoids to Qwen1.5-MoE. It does not run QJL, pruning, or training.

Layers: [12]; experts: [0, 1]; shared ranks: [4]; residual ranks: [0, 2].
Calibration: /home/jaewoo/QJLdecomposition/results/grouping_gsmoe/full_cov_smoke/calibration (routing frequency: full_routing_count).
Full covariance method: all-routed-tokens-route-probability-weighted-v1; dataset SHA-256: de0723fdb55d5b9a5e4e3f1cd7683bac68f8f235314ae423adb1f87ab5926d16; diagnostic sample cap per expert: 8. All routed tokens contribute to fitting covariance; saved samples are used only for output diagnostics.

Shared projection coefficients and bases are stored as BF16. The FP32 residual is defined after reconstructing those stored BF16 factors; the residual SVD oracle is measured before its factors are rounded to BF16. Pooled norm ratios use summed squared norms, while mean metrics average experts. Routed output errors run BF16 checkpoint arithmetic and are separate from Frobenius/SVD energy.

## Projection diagnostics

| Projection | Shared rank | Residual rank | Mean R/W | Pooled R/W | Pooled stored Tail/W | Pooled output L2 |
|---|---:|---:|---:|---:|---:|---:|
| down | 4 | 0 | 0.99761 | 0.99761 | 0.99761 | 0.62760 |
| down | 4 | 2 | 0.99761 | 0.99761 | 0.99080 | 0.62464 |
| gate | 4 | 0 | 0.99790 | 0.99790 | 0.99790 | 0.68955 |
| gate | 4 | 2 | 0.99790 | 0.99790 | 0.98939 | 0.67743 |
| up | 4 | 0 | 0.99956 | 0.99957 | 0.99957 | 0.75478 |
| up | 4 | 2 | 0.99956 | 0.99957 | 0.99390 | 0.73918 |

## Full SwiGLU expert output

| Shared rank | Residual rank | Mean relative L2 | Pooled relative L2 | Mean cosine |
|---:|---:|---:|---:|---:|
| 4 | 0 | 0.74324 | 0.64295 | 0.53850 |
| 4 | 2 | 0.74011 | 0.63953 | 0.53769 |

Group sizes: {2: 1}.

## Checkpoint storage

| Shared rank | Residual rank | Bytes | Effective bpw | BF16 ratio |
|---:|---:|---:|---:|---:|
| 4 | 0 | 120862 | 0.056 | 286.302x |
| 4 | 2 | 206825 | 0.096 | 167.306x |

The original HF backbone, router, and shared expert are external dependencies of these expert-only checkpoints. Full evaluation was not run as part of this report.
