# GS-MoE gain grouping + shared projection + residual SVD

This experiment adapts GS-MoE's reconstruction gain and fixed-size balanced K-medoids to Qwen1.5-MoE. It does not run QJL, pruning, or training.

Layers: [0, 12, 23]; experts: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]; shared ranks: [64, 128, 256]; residual ranks: [0, 32, 64, 128, 256].
Calibration: /home/jaewoo/QJLdecomposition/results/grouping_gsmoe/calibration (routing frequency: full_routing_count).

Shared projection coefficients and bases are stored as BF16. The FP32 residual is defined after reconstructing those stored BF16 factors; the residual SVD oracle is measured before its factors are rounded to BF16. Pooled norm ratios use summed squared norms, while mean metrics average experts. Routed output errors run BF16 checkpoint arithmetic and are separate from Frobenius/SVD energy.

## Projection diagnostics

| Projection | Shared rank | Residual rank | Mean R/W | Pooled R/W | Pooled stored Tail/W | Pooled output L2 |
|---|---:|---:|---:|---:|---:|---:|
| down | 64 | 0 | 0.95778 | 0.95846 | 0.95846 | 0.52871 |
| down | 64 | 32 | 0.95778 | 0.95846 | 0.90927 | 0.50652 |
| down | 64 | 64 | 0.95778 | 0.95846 | 0.86969 | 0.48722 |
| down | 64 | 128 | 0.95778 | 0.95846 | 0.79938 | 0.45206 |
| down | 64 | 256 | 0.95778 | 0.95846 | 0.67722 | 0.38909 |
| down | 128 | 0 | 0.92271 | 0.92371 | 0.92371 | 0.39506 |
| down | 128 | 32 | 0.92271 | 0.92371 | 0.87716 | 0.38072 |
| down | 128 | 64 | 0.92271 | 0.92371 | 0.83860 | 0.36764 |
| down | 128 | 128 | 0.92271 | 0.92371 | 0.76965 | 0.34306 |
| down | 128 | 256 | 0.92271 | 0.92371 | 0.64948 | 0.29794 |
| down | 256 | 0 | 0.85981 | 0.86119 | 0.86119 | 0.22087 |
| down | 256 | 32 | 0.85981 | 0.86119 | 0.81754 | 0.21381 |
| down | 256 | 64 | 0.85981 | 0.86119 | 0.78031 | 0.20715 |
| down | 256 | 128 | 0.85981 | 0.86119 | 0.71334 | 0.19438 |
| down | 256 | 256 | 0.85981 | 0.86119 | 0.59637 | 0.17029 |
| gate | 64 | 0 | 0.95862 | 0.96031 | 0.96031 | 0.40469 |
| gate | 64 | 32 | 0.95862 | 0.96031 | 0.89806 | 0.38388 |
| gate | 64 | 64 | 0.95862 | 0.96031 | 0.85702 | 0.36636 |
| gate | 64 | 128 | 0.95862 | 0.96031 | 0.78701 | 0.33649 |
| gate | 64 | 256 | 0.95862 | 0.96031 | 0.66753 | 0.28608 |
| gate | 128 | 0 | 0.97587 | 0.97774 | 0.97774 | 0.25786 |
| gate | 128 | 32 | 0.97587 | 0.97774 | 0.88107 | 0.24650 |
| gate | 128 | 64 | 0.97587 | 0.97774 | 0.83706 | 0.23598 |
| gate | 128 | 128 | 0.97587 | 0.97774 | 0.76608 | 0.21725 |
| gate | 128 | 256 | 0.97587 | 0.97774 | 0.64677 | 0.18500 |
| gate | 256 | 0 | 1.43711 | 1.48913 | 1.48913 | 0.02610 |
| gate | 256 | 32 | 1.43711 | 1.48913 | 0.90301 | 0.01269 |
| gate | 256 | 64 | 1.43711 | 1.48913 | 0.81189 | 0.01054 |
| gate | 256 | 128 | 1.43711 | 1.48913 | 0.73065 | 0.00909 |
| gate | 256 | 256 | 1.43711 | 1.48913 | 0.60966 | 0.00845 |
| up | 64 | 0 | 0.97384 | 0.97514 | 0.97514 | 0.45266 |
| up | 64 | 32 | 0.97384 | 0.97514 | 0.91819 | 0.43122 |
| up | 64 | 64 | 0.97384 | 0.97514 | 0.87837 | 0.41274 |
| up | 64 | 128 | 0.97384 | 0.97514 | 0.80831 | 0.38009 |
| up | 64 | 256 | 0.97384 | 0.97514 | 0.68676 | 0.32368 |
| up | 128 | 0 | 0.99591 | 0.99665 | 0.99665 | 0.28948 |
| up | 128 | 32 | 0.99591 | 0.99665 | 0.90075 | 0.27721 |
| up | 128 | 64 | 0.99591 | 0.99665 | 0.85795 | 0.26583 |
| up | 128 | 128 | 0.99591 | 0.99665 | 0.78684 | 0.24518 |
| up | 128 | 256 | 0.99591 | 0.99665 | 0.66535 | 0.20906 |
| up | 256 | 0 | 1.46301 | 1.51989 | 1.51989 | 0.02917 |
| up | 256 | 32 | 1.46301 | 1.51989 | 0.92433 | 0.01385 |
| up | 256 | 64 | 1.46301 | 1.51989 | 0.83221 | 0.01128 |
| up | 256 | 128 | 1.46301 | 1.51989 | 0.75064 | 0.00978 |
| up | 256 | 256 | 1.46301 | 1.51989 | 0.62732 | 0.00905 |

## Full SwiGLU expert output

| Shared rank | Residual rank | Mean relative L2 | Pooled relative L2 | Mean cosine |
|---:|---:|---:|---:|---:|
| 64 | 0 | 0.67813 | 0.58441 | 0.72609 |
| 64 | 32 | 0.65783 | 0.56926 | 0.74293 |
| 64 | 64 | 0.63942 | 0.55591 | 0.75837 |
| 64 | 128 | 0.60509 | 0.53003 | 0.78624 |
| 64 | 256 | 0.53732 | 0.47697 | 0.83534 |
| 128 | 0 | 0.49246 | 0.43286 | 0.86180 |
| 128 | 32 | 0.47828 | 0.42223 | 0.86948 |
| 128 | 64 | 0.46458 | 0.41179 | 0.87694 |
| 128 | 128 | 0.43843 | 0.39130 | 0.89081 |
| 128 | 256 | 0.38752 | 0.34975 | 0.91549 |
| 256 | 0 | 0.24762 | 0.22185 | 0.96381 |
| 256 | 32 | 0.23877 | 0.21411 | 0.96637 |
| 256 | 64 | 0.23097 | 0.20739 | 0.96855 |
| 256 | 128 | 0.21634 | 0.19461 | 0.97244 |
| 256 | 256 | 0.18899 | 0.17057 | 0.97901 |

Group sizes: {2: 90}.

## Checkpoint storage

Analyze mode does not write checkpoints; exact bytes are measured in compress mode.

The original HF backbone, router, and shared expert are external dependencies of these expert-only checkpoints. Full evaluation was not run as part of this report.
