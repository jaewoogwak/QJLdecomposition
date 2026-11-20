# Exponent-conditioned mantissa sharing

Weight-only residual diagnostics. The existing encoder and QJL path were not changed.
Model: `/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat`. Layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]; experts: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]; projections: ['gate', 'up', 'down']; minimum exponent count: 1.

## Overall comparison

| Method | Residual rho | Change vs A | Base bits/weight | D=n total bits | D=2n total bits |
|---|---:|---:|---:|---:|---:|
| A: existing FP16 row beta | 0.194752 | 0.000000 | 5.077 | 6.095 | 7.095 |
| B1: exponent FP32 oracle | 0.178986 | 0.015766 | N/A | N/A | N/A |
| B2: exponent FP16 beta | 0.178986 | 0.015766 | 5.262 | 6.280 | 7.280 |

B1 uses FP32 beta only as an oracle; its storage fields are N/A. A Base bytes are measured from the existing packed payload. B2 Base bytes and all QJL totals are projected, not checkpoint file sizes.

## By projection

| Projection | A rho | B1 rho | B2 rho | B2 rho reduction | B2 residual energy / A |
|---|---:|---:|---:|---:|---:|
| gate | 0.194854 | 0.178963 | 0.178963 | 0.015891 | 0.843544 |
| up | 0.194729 | 0.179087 | 0.179087 | 0.015642 | 0.845797 |
| down | 0.194674 | 0.178907 | 0.178907 | 0.015767 | 0.844579 |

## Findings

- B2 changes global rho by 0.015766 (8.095% relative) and its residual energy ratio is 0.844647.
- B2 lowers rho on 4320/4320 matrices, leaves 0 effectively unchanged, and raises it on 0. FP16 storage worsens A on 0 matrices and 0 rows.
- B2 minus B1 rho is 0.00000012. This is the observed FP16 rounding penalty.
- Mean within-row exponent-beta standard deviation across matrices is 0.15409; the mean count-weighted absolute difference from row beta is 0.12704. The exponent-conditioned betas are measured separately from the row beta.
- B2 adds 0.185 Base bits per original weight. Projected D=n and D=2n total bits are 6.280 and 7.280. These estimates include the same QJL sign bits and FP32 row norms; no QJL encoding was run.
- Projected memory reduction relative to 16-bit BF16 is 60.75% for D=n and 54.50% for D=2n.
- In this weight-only comparison, residual energy falls by 15.54% while estimated Base storage rises by 3.64% (0.185 bits/weight). This is a material residual reduction at the estimated overhead. Whether it improves QJL accuracy or runtime remains unmeasured.

## By layer

| Layer | A rho | B1 rho | B2 rho | B2 rho reduction |
|---:|---:|---:|---:|---:|
| 0 | 0.195272 | 0.178775 | 0.178775 | 0.016497 |
| 1 | 0.195313 | 0.178668 | 0.178668 | 0.016645 |
| 2 | 0.195148 | 0.178670 | 0.178670 | 0.016478 |
| 3 | 0.195033 | 0.178664 | 0.178664 | 0.016370 |
| 4 | 0.194965 | 0.178700 | 0.178701 | 0.016264 |
| 5 | 0.194913 | 0.178770 | 0.178770 | 0.016143 |
| 6 | 0.194871 | 0.178879 | 0.178879 | 0.015992 |
| 7 | 0.194811 | 0.178911 | 0.178911 | 0.015900 |
| 8 | 0.194764 | 0.178976 | 0.178976 | 0.015787 |
| 9 | 0.194717 | 0.178993 | 0.178993 | 0.015724 |
| 10 | 0.194735 | 0.178990 | 0.178990 | 0.015745 |
| 11 | 0.194755 | 0.179054 | 0.179055 | 0.015701 |
| 12 | 0.194745 | 0.179059 | 0.179059 | 0.015685 |
| 13 | 0.194747 | 0.179059 | 0.179059 | 0.015688 |
| 14 | 0.194712 | 0.179096 | 0.179096 | 0.015616 |
| 15 | 0.194679 | 0.179090 | 0.179090 | 0.015589 |
| 16 | 0.194633 | 0.179084 | 0.179084 | 0.015549 |
| 17 | 0.194619 | 0.179054 | 0.179054 | 0.015565 |
| 18 | 0.194570 | 0.179040 | 0.179040 | 0.015530 |
| 19 | 0.194565 | 0.179061 | 0.179061 | 0.015505 |
| 20 | 0.194535 | 0.179071 | 0.179071 | 0.015463 |
| 21 | 0.194491 | 0.179097 | 0.179098 | 0.015394 |
| 22 | 0.194428 | 0.179191 | 0.179191 | 0.015237 |
| 23 | 0.194358 | 0.179462 | 0.179462 | 0.014896 |

## Storage representation

B2's estimate reuses the existing packed sign bytes, independently addressable lossless exponent blocks, and special-value table. Each present finite nonzero (row, original exponent) group stores one uint8 exponent ID and one FP16 beta. A uint32 row-offset array has rows+1 entries. Absent groups occupy no table entry. The dense [rows, 256] fitting array is temporary and is not counted as storage. Rare-group fallbacks still have one sparse entry with the row beta. B1 has no deployable storage claim.

See `exponent_statistics.json` for group-size, beta-variation, fallback, and BF16 special-value counts. A full-model conclusion requires all 24 layers; this report describes only the selected sweep.
