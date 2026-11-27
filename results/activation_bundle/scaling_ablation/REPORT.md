# Activation-aware bundled Base: row scaling ablation

A0/A share majority-vote codes; B0/B share activation-aware codes.
Only the expert-specific row scale changes. A/B evaluation and
per-expert measurements are reused from `optimized_exact_k3_r256`.

All metrics below use 24 layers, group size 3, D=2n, rank 0,
zero-shot full ARC-Easy and ARC-Challenge. Error columns are
unweighted means over layer × projection × expert.

| Case | Row scale | Mean acc | Mean acc_norm | Weight rel. L2 | Cal. loss | Heldout loss | Cal. rel. L2 | Heldout rel. L2 | BPW |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | original_weight_row_l2 | 0.2397 | 0.2601 | 1.2384 | 1.2816 | 1.2927 | 1.1279 | 1.1328 | 0.6857 |
| A | activation_optimized | 0.2509 | 0.2546 | 0.9350 | 0.7553 | 0.7600 | 0.8672 | 0.8698 | 0.6857 |
| B0 | original_weight_row_l2 | 0.2351 | 0.2508 | 1.2466 | 1.3072 | 1.3192 | 1.1339 | 1.1391 | 0.6857 |
| B | activation_optimized | 0.2901 | 0.3002 | 0.9327 | 0.7511 | 0.7561 | 0.8635 | 0.8663 | 0.6857 |

Per-task accuracy and storage bytes are in `summary.csv`/`summary.json`.
`per_expert.csv` contains the underlying errors. New A0/B0 packed
checkpoints are in `checkpoint/`; covariance caches and the
original model are excluded from BPW.
