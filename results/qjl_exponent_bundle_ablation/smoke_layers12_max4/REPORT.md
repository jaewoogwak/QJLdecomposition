# Exponent Base and QJL sketch ablation

A: existing exponent-mantissa k=0 Base (stored FP16 row beta), no QJL. B: QJL of original BF16 W. C: QJL of the signed Exponent Base, with its own row norms. D: group majority vote of C's sketches, retaining each expert's Exponent Base row norms. Every residual is original W minus the case's effective Base; no model inference was performed.

Layers: [12]; D=2048; FP16 Gaussian seed 0; non-singleton expert/projection pairs: 12. All original weights are BF16. Full FP32 SVD uses TF32 disabled.

| Case | Mean residual rho | Mean E128 | Mean E512 | Mean optimal error128 | Mean optimal error512 | Mean r90 | Mean r95 | Mean r99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.1948 | 0.2524 | 0.7091 | 0.1684 | 0.1050 | 841.3333 | 999.3333 | 1237.9167 |
| B | 1.1810 | 0.4231 | 0.8260 | 0.8975 | 0.4925 | 684.0000 | 868.3333 | 1164.9167 |
| C | 1.1747 | 0.4150 | 0.8209 | 0.8988 | 0.4969 | 692.9167 | 875.8333 | 1169.5000 |
| D | 1.6241 | 0.3526 | 0.7976 | 1.3068 | 0.7306 | 722.2500 | 896.5833 | 1178.4167 |

## Paired changes

Changes below are right minus left for the same expert and projection. A smaller rho/error/rank or larger retained energy is favorable.

| Comparison | Mean Δrho | Fraction with smaller rho | Mean ΔE128 | Mean ΔE512 |
| --- | ---: | ---: | ---: | ---: |
| A_to_B | +0.9862 | 0.0% | +0.1707 | +0.1168 |
| B_to_C | -0.0063 | 100.0% | -0.0080 | -0.0051 |
| A_to_C | +0.9799 | 0.0% | +0.1626 | +0.1118 |
| C_to_D | +0.4494 | 0.0% | -0.0624 | -0.0234 |

## Signed binary sketch agreement

These are `sign(BG)` sketches of the signed Exponent Base. The unsigned control uses `abs(B)` and is reported separately; it must not be interpreted as signed-sketch agreement.

| Metric | Mean agreement |
| --- | ---: |
| Same expert sign(WG) vs sign(BG) | 93.76% |
| Signed Base expert pairs | 50.02% |
| Original W expert pairs | 50.02% |
| Unsigned abs(Base) expert pairs | 70.80% |
| Random neuron pairing of signed Base | 50.02% |
| Random-row Base vs random majority | 57.03% |
| Independent sign chance | 50.00% |
| Individual signed Base vs majority | 58.59% |

## Validation and limits

Maximum `W - (Base + Residual)` relative error: 3.47e-08. Maximum FP32 SVD energy relative gap: 0.000366. Existing Case B spectrum cross-check: matched (12 matrices).

The random-row control permutes neuron rows independently for each expert while preserving its sketch bits and bit balance. The independent-sign chance baseline uses each expert's observed 1-bit fraction. Neither control is a compressed-model accuracy result.

Optimal rank errors are full-SVD lower bounds. This experiment does not store low-rank factors, create a compressed checkpoint, or run ARC evaluation. A favorable residual spectrum does not by itself prove end-to-end inference gains.

Raw full singular values, expert IDs, cases, per-group sign controls, and per-expert metrics are in `layers/layer_XX.npz`, `group_sign_stats.csv`, `sign_agreement.csv`, and `per_expert.csv`.
