# Exponent Base and QJL sketch ablation

A: existing exponent-mantissa k=0 Base (stored FP16 row beta), no QJL. B: QJL of original BF16 W. C: QJL of the signed Exponent Base, with its own row norms. D: group majority vote of C's sketches, retaining each expert's Exponent Base row norms. Every residual is original W minus the case's effective Base; no model inference was performed.

Layers: [12]; D=2048; FP16 Gaussian seed 0; non-singleton expert/projection pairs: 96. All original weights are BF16. Full FP32 SVD uses TF32 disabled.

| Case | Mean residual rho | Mean E128 | Mean E512 | Mean optimal error128 | Mean optimal error512 | Mean r90 | Mean r95 | Mean r99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 0.1947 | 0.2527 | 0.7093 | 0.1684 | 0.1050 | 841.0833 | 999.3438 | 1237.9271 |
| B | 1.1808 | 0.4214 | 0.8252 | 0.8987 | 0.4937 | 685.4479 | 869.5417 | 1165.7292 |
| C | 1.1746 | 0.4135 | 0.8202 | 0.8999 | 0.4980 | 694.1875 | 877.0729 | 1170.2083 |
| D | 1.6429 | 0.3533 | 0.7981 | 1.3213 | 0.7380 | 721.4062 | 895.8438 | 1178.0833 |

## Paired changes

Changes below are right minus left for the same expert and projection. A smaller rho/error/rank or larger retained energy is favorable.

| Comparison | Mean Δrho | Fraction with smaller rho | Mean ΔE128 | Mean ΔE512 |
| --- | ---: | ---: | ---: | ---: |
| A_to_B | +0.9861 | 0.0% | +0.1688 | +0.1159 |
| B_to_C | -0.0062 | 100.0% | -0.0079 | -0.0050 |
| A_to_C | +0.9799 | 0.0% | +0.1609 | +0.1109 |
| C_to_D | +0.4682 | 0.0% | -0.0602 | -0.0221 |

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
| Individual signed Base vs majority | 57.03% |

## Projection-specific means

| Projection | A rho | B rho | C rho | D rho | Signed Base pairs | Random-row pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gate | 0.1949 | 1.2521 | 1.2436 | 1.6914 | 50.05% | 50.05% |
| Up | 0.1947 | 1.2523 | 1.2436 | 1.6913 | 50.00% | 50.00% |
| Down | 0.1946 | 1.0381 | 1.0367 | 1.5460 | 50.00% | 50.00% |

## Questions answered by this diagnostic

Q1: A retains mean residual rho 0.1947.
Q2: C changes rho from B by -0.0062 on average, improving 100.0% of paired matrices.
Q3: QJL of the exponent Base raises rho over A by +0.9799.
Q4: Majority voting raises rho over C by +0.4682.
Q5: C minus B mean retained energy is -0.0079 at rank 128 and -0.0050 at rank 512. D minus C is -0.0602 and -0.0221.
Q6: Signed Base pairwise agreement is 50.017%, versus 50.015% under random row pairing. Actual individual-vs-majority agreement is 57.034%, versus 57.032% under the random-row control. The unsigned abs(Base) value of 70.803% does not indicate signed sketch sharing.


## Validation and limits

Maximum `W - (Base + Residual)` relative error: 3.48e-08. Maximum FP32 SVD energy relative gap: 0.000375. Existing Case B spectrum cross-check: matched (96 matrices).

The random-row control permutes neuron rows independently for each expert while preserving its sketch bits and bit balance. The independent-sign chance baseline uses each expert's observed 1-bit fraction. Neither control is a compressed-model accuracy result.

Optimal rank errors are full-SVD lower bounds. This experiment does not store low-rank factors, create a compressed checkpoint, or run ARC evaluation. A favorable residual spectrum does not by itself prove end-to-end inference gains.

Raw full singular values, expert IDs, cases, per-group sign controls, and per-expert metrics are in `layers/layer_XX.npz`, `group_sign_stats.csv`, `sign_agreement.csv`, and `per_expert.csv`.
