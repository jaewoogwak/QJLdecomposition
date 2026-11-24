# Pairwise QJL Base + low-rank residual

A uses one shared packed sketch per pair and expert-specific FP32 norms.
B adds BF16 SVD factors for the residual after bundling.

Layers: [12]; D factor: 2.0; rank: 512.

| Case | Mean acc | Mean acc_norm | Checkpoint bytes | Effective BPW |
|---|---:|---:|---:|---:|
| original | 0.5000 | 1.0000 | 28632144944 | 16.000 |
| A | 1.0000 | 1.0000 | 66126297 | 1.019 |
| B | 1.0000 | 1.0000 | 703240550 | 10.839 |

Original checkpoint bytes count all model safetensors; A/B checkpoint bytes count only compressed routed layers. Effective BPW and BF16 reduction use routed weights in the selected layers.
