# Shared Base + Low-rank Residual + QJL Tail

A uses QJL(R); B uses stored BF16 low-rank factors; C adds QJL(E) to those same factors. R and E are FP32 weight residuals. Output error is measured on the same input within each row.

Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]; expert IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]; projections: ['gate', 'up', 'down']; ranks: [0, 64, 128, 256]; QJL D=4096.
Grouping source: ResCode expert-output descriptors and saliency.

Aggregated residual norms, energies, and output errors exclude singleton experts (which retain their original BF16 weights and have zero residual). The per-expert CSV retains them with `singleton=True` for auditability. Norm ratios use sums of squared norms over non-singleton experts.

## Output diagnostics

| Projection | Rank | Input | Experts | A relative L2 | B relative L2 | C relative L2 | Tail / W |
|---|---:|---|---:|---:|---:|---:|---:|
| down | 0 | routed | 745 | 0.70702 | 0.98327 | 0.70702 | 0.98418 |
| down | 64 | routed | 745 | 0.70702 | 0.88065 | 0.64130 | 0.89317 |
| down | 128 | routed | 745 | 0.70702 | 0.80839 | 0.59104 | 0.82313 |
| down | 256 | routed | 745 | 0.70702 | 0.68481 | 0.50342 | 0.70125 |
| gate | 0 | routed | 745 | 0.71196 | 0.90868 | 0.71196 | 0.98286 |
| gate | 64 | routed | 745 | 0.71196 | 0.72021 | 0.63618 | 0.87736 |
| gate | 128 | routed | 745 | 0.71196 | 0.64558 | 0.58374 | 0.80516 |
| gate | 256 | routed | 745 | 0.71196 | 0.53610 | 0.49545 | 0.68317 |
| up | 0 | routed | 745 | 0.80992 | 0.98270 | 0.80992 | 0.98417 |
| up | 64 | routed | 745 | 0.80992 | 0.83309 | 0.73654 | 0.89465 |
| up | 128 | routed | 745 | 0.80992 | 0.75262 | 0.67813 | 0.82381 |
| up | 256 | routed | 745 | 0.80992 | 0.62811 | 0.57696 | 0.70108 |

## Checkpoint storage

Analyze mode does not write checkpoints; storage is measured in compress mode.

Skipped ranks exceeding the selected matrix dimensions: 0.
Singleton experts remain exact BF16. The checkpoint contains selected expert tensors only; the original HF backbone, router, and shared expert remain dependencies for evaluation and are excluded from expert bpw.
When routed calibration inputs are unavailable, `input_source=synthetic` is recorded. SVD energy metrics are Frobenius quantities; QJL accuracy uses output vectors and is reported separately.
