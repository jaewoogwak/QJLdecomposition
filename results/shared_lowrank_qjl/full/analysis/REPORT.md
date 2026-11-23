# Shared Base + Low-rank Residual + QJL Tail

A uses QJL(R); B uses stored BF16 low-rank factors; C adds QJL(E) to those same factors. R and E are FP32 weight residuals. Output error is measured on the same input within each row.

Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]; expert IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]; projections: ('gate', 'up', 'down'); ranks: [0, 64, 128, 256]; QJL D=4096.
Grouping source: ResCode expert-output descriptors and saliency.

## Output diagnostics

| Projection | Rank | Input | A relative L2 | B relative L2 | C relative L2 | Tail / W |
|---|---:|---|---:|---:|---:|---:|
| down | 0 | routed | 0.36579 | 0.50871 | 0.36579 | 0.70658 |
| down | 64 | routed | 0.36579 | 0.45561 | 0.33179 | 0.64124 |
| down | 128 | routed | 0.36579 | 0.41823 | 0.30578 | 0.59096 |
| down | 256 | routed | 0.36579 | 0.35429 | 0.26045 | 0.50346 |
| gate | 0 | routed | 0.36834 | 0.47012 | 0.36834 | 0.70611 |
| gate | 64 | routed | 0.36834 | 0.37261 | 0.32914 | 0.63031 |
| gate | 128 | routed | 0.36834 | 0.33400 | 0.30201 | 0.57844 |
| gate | 256 | routed | 0.36834 | 0.27736 | 0.25632 | 0.49080 |
| up | 0 | routed | 0.41902 | 0.50841 | 0.41902 | 0.70668 |
| up | 64 | routed | 0.41902 | 0.43101 | 0.38106 | 0.64240 |
| up | 128 | routed | 0.41902 | 0.38938 | 0.35084 | 0.59153 |
| up | 256 | routed | 0.41902 | 0.32496 | 0.29849 | 0.50341 |

## Checkpoint storage

Analyze mode does not write checkpoints; storage is measured in compress mode.

Skipped ranks exceeding the selected matrix dimensions: 0.
Singleton experts remain exact BF16. The checkpoint contains selected expert tensors only; the original HF backbone, router, and shared expert remain dependencies for evaluation and are excluded from expert bpw.
When routed calibration inputs are unavailable, `input_source=synthetic` is recorded. SVD energy metrics are Frobenius quantities; QJL accuracy uses output vectors and is reported separately.
