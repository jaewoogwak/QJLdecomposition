# Pairwise QJL Base + low-rank residual

A uses one shared packed sketch per pair and expert-specific FP32 norms.
B adds BF16 SVD factors for the residual after bundling.

Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]; D factor: 2.0; rank: 512.

| Case | Mean acc | Mean acc_norm | Checkpoint bytes | Effective BPW |
|---|---:|---:|---:|---:|
| original | 0.5511 | 0.5465 | 28632144944 | 16.000 |
| A | 0.2330 | 0.2633 | 1587022608 | 1.019 |
| B | 0.3539 | 0.3521 | 16877760380 | 10.839 |

Original checkpoint bytes count all model safetensors; A/B checkpoint bytes count only compressed routed layers. Effective BPW and BF16 reduction use routed weights in the selected layers.

Analyzed 4320 residual matrices. Means across matrices:

- Residual/weight Frobenius ratio: 1.133021
- Pre-bundle bit agreement: 0.500997
- Post-bundle bit preservation: 0.750499
- Rank-512 FP32 residual energy retention: 0.748784
- Optimal rank-512 weight error: 0.567823
- BF16 factor weight error: 0.567797
