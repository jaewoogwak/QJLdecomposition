# Held-out activation reconstruction: layers 0, 12, 23

The same saved BF16 checkpoint factors are evaluated on train and held-out routed inputs. No grouping, projection, coefficient, or SVD is fitted with held-out data. QJL and lm-eval are not run.

Train source: `/home/jaewoo/QJLdecomposition/rescode/data/c4-train.00000-of-01024.json`. Held-out source: `/home/jaewoo/QJLdecomposition/results/grouping_gsmoe/heldout_data/c4-train.00001-of-01024.json`. Exact text-document overlap across the complete source shards: 0. Seed: 31415; held-out sample cap: 256.
Maximum absolute train-result reproduction differences: projection L2 0, SwiGLU L2 0.

Pooled relative L2 is sqrt(sum squared output error / sum squared original output norm). Each table entry uses the original BF16 expert outputs as reference and the saved BF16 checkpoint factors for reconstruction.

## Layer-wise full SwiGLU output: shared rank 256

| Layer | Residual rank | Train pooled L2 | Held-out pooled L2 | Train mean cosine | Held-out mean cosine |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.17546 | 0.84394 | 0.98309 | 0.44808 |
| 0 | 64 | 0.16250 | 0.74108 | 0.98550 | 0.59975 |
| 0 | 256 | 0.13062 | 0.58745 | 0.99064 | 0.77330 |
| 12 | 0 | 0.31542 | 1.57659 | 0.94493 | 0.10480 |
| 12 | 64 | 0.29521 | 0.89317 | 0.95193 | 0.43933 |
| 12 | 256 | 0.24355 | 0.72206 | 0.96757 | 0.68649 |
| 23 | 0 | 0.21735 | 1.20930 | 0.96342 | 0.25488 |
| 23 | 64 | 0.20316 | 0.76760 | 0.96820 | 0.52079 |
| 23 | 256 | 0.16706 | 0.63097 | 0.97882 | 0.69970 |

## Gate/Up/Down output: shared rank 256, by layer

| Layer | Projection | Residual rank | Train pooled L2 | Held-out pooled L2 |
|---:|---|---:|---:|---:|
| 0 | gate | 0 | 0.01095 | 0.71240 |
| 0 | gate | 64 | 0.00626 | 0.47475 |
| 0 | gate | 256 | 0.00555 | 0.34413 |
| 0 | up | 0 | 0.01315 | 0.87758 |
| 0 | up | 64 | 0.00716 | 0.58455 |
| 0 | up | 256 | 0.00621 | 0.42519 |
| 0 | down | 0 | 0.17530 | 0.68554 |
| 0 | down | 64 | 0.16241 | 0.60473 |
| 0 | down | 256 | 0.13049 | 0.44402 |
| 12 | gate | 0 | 0.02729 | 1.38751 |
| 12 | gate | 64 | 0.01062 | 0.58000 |
| 12 | gate | 256 | 0.00870 | 0.40783 |
| 12 | up | 0 | 0.03059 | 1.54027 |
| 12 | up | 64 | 0.01104 | 0.65473 |
| 12 | up | 256 | 0.00905 | 0.46531 |
| 12 | down | 0 | 0.31464 | 0.80719 |
| 12 | down | 64 | 0.29504 | 0.72461 |
| 12 | down | 256 | 0.24333 | 0.54831 |
| 23 | gate | 0 | 0.02633 | 1.16519 |
| 23 | gate | 64 | 0.01069 | 0.53547 |
| 23 | gate | 256 | 0.00851 | 0.38712 |
| 23 | up | 0 | 0.02927 | 1.35996 |
| 23 | up | 64 | 0.01146 | 0.63716 |
| 23 | up | 256 | 0.00914 | 0.46430 |
| 23 | down | 0 | 0.21635 | 0.68671 |
| 23 | down | 64 | 0.20292 | 0.61862 |
| 23 | down | 256 | 0.16677 | 0.47026 |

## Interpretation

At shared rank 256 and residual rank 0, Gate pooled output L2 changes from 0.02610 to 1.19827; Up from 0.02917 to 1.38588.
Full SwiGLU: residual rank 0→256 changes train pooled L2 0.22185→0.17057 and held-out pooled L2 1.22675→0.63506.
The small train Gate/Up errors do not persist on this held-out C4 shard. The large train/held-out gaps across Gate, Up, and full SwiGLU indicate calibration-specific reconstruction and poor generalization of these saved factors to the new routed inputs.
Residual SVD improves held-out full SwiGLU error by 0.59169 absolute (48.2% relative) from residual rank 0 to 256.

## Limits

The train calibration cache did not record its selected document IDs or a source checksum. Its association with shard 00000 follows the original run's configured default. We verified exact disjointness of the complete source-shard text sets, but cannot independently prove the identity of each cached train document. C4 may still contain similar or overlapping passages across different documents. The comparisons cover only routed experts and samples from the selected layers, not full-model language evaluation.
