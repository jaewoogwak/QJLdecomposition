# Activation-aware QJL binary bundling

Checkpoint: `/home/jaewoo/QJLdecomposition/results/activation_bundle/optimized_exact_k3_rank_sweep/checkpoint`

Calibrated layer count: 24

Per-expert records: 34560

Compression time across completed layers: 6633.6 s.
Peak GPU allocation during compression: 35361362432 bytes.
Peak CPU RSS during compression: 6843392000 bytes.
Selected covariance cache bytes: 71158212024.

Calibration backend: layerwise.
Calibration cache bytes: 71158214677.

| Compression stage | Seconds |
|---|---:|
| checkpoint_io_seconds | 35.363 |

| Case | Mean accuracy | Mean acc_norm | Checkpoint bytes | BPW |
|---|---:|---:|---:|---:|
| E | 0.3773 | 0.3732 | 2981394482 | 1.915 |
| E | 0.4167 | 0.4020 | 4892439570 | 3.142 |
| E | 0.5394 | 0.5314 | 16358625810 | 10.506 |

A and B contain packed group sketches and expert scales only. C and D use B's base; E uses a refined code. Calibration cache bytes are excluded from inference checkpoint bytes. Original checkpoint bytes count the whole BF16 model, while compressed checkpoint bytes count routed weights of selected layers.
