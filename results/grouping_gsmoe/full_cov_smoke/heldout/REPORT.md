# Held-out activation reconstruction: layers 12

The same saved BF16 checkpoint factors are evaluated on train and held-out routed inputs. No grouping, projection, coefficient, or SVD is fitted with held-out data. QJL and lm-eval are not run.

Train source: `/home/jaewoo/QJLdecomposition/rescode/data/c4-train.00000-of-01024.json`. Held-out source: `/home/jaewoo/QJLdecomposition/results/grouping_gsmoe/full_cov_smoke/heldout_data/c4-train.00001-of-01024.json`. Exact text-document overlap across the complete source shards: 0. Seed: 31415; held-out sample cap: 8.
Maximum absolute train-result reproduction differences: projection L2 0, SwiGLU L2 0.

Pooled relative L2 is sqrt(sum squared output error / sum squared original output norm). Each table entry uses the original BF16 expert outputs as reference and the saved BF16 checkpoint factors for reconstruction.

## Layer-wise full SwiGLU output: shared rank 4

| Layer | Residual rank | Train pooled L2 | Held-out pooled L2 | Train mean cosine | Held-out mean cosine |
|---:|---:|---:|---:|---:|---:|
| 12 | 0 | 0.64295 | 0.99914 | 0.53850 | 0.05424 |
| 12 | 2 | 0.63953 | 1.00000 | 0.53769 | 0.02033 |

## Gate/Up/Down output: shared rank 4, by layer

| Layer | Projection | Residual rank | Train pooled L2 | Held-out pooled L2 |
|---:|---|---:|---:|---:|
| 12 | gate | 0 | 0.68955 | 0.96729 |
| 12 | gate | 2 | 0.67743 | 0.92019 |
| 12 | up | 0 | 0.75478 | 0.97432 |
| 12 | up | 2 | 0.73918 | 0.95188 |
| 12 | down | 0 | 0.62760 | 0.99651 |
| 12 | down | 2 | 0.62464 | 0.98797 |

## Limits

The full-covariance train cache records a source SHA-256 matching shard 00000, and the grouping metadata matches that cache. Individual train document IDs were not recorded. We verified exact disjointness of the complete source-shard text sets. C4 may still contain similar or overlapping passages across different documents. The comparisons cover only routed experts and samples from the selected layers, not full-model language evaluation.
