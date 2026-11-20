# Qwen shared subspace ablation

Weight-only FP32 oracle and FP16-factor reconstruction; no QJL encoding or inference.
Layers: [0, 12, 23]; projections: ['gate', 'up', 'down']; experts: 60; target rho <= 0.195.

## Best practical rank in the requested sweep

| Projection | Grouping | Group size | Method | Minimum rank meeting target | FP16 rho | Base reduction | Base bpw |
|---|---|---:|---|---:|---:|---:|---:|
| down | random | 1 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | random | 1 | shared_input | 1024 | 0.188957 | -22.73% | 19.636 |
| down | random | 1 | shared_output | 1024 | 0.188957 | -22.73% | 19.636 |
| down | random | 4 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | random | 4 | shared_input | 1408 | 0.000293 | -17.19% | 18.750 |
| down | random | 4 | shared_output | not met | 0.308099 | 6.25% | 15.000 |
| down | random | 8 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | random | 8 | shared_input | 1408 | 0.000293 | -9.17% | 17.467 |
| down | random | 8 | shared_output | not met | 0.367987 | 17.92% | 13.133 |
| down | random | 60 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | random | 60 | shared_input | 1408 | 0.000293 | -1.15% | 16.183 |
| down | random | 60 | shared_output | not met | 0.453414 | 29.58% | 11.267 |
| down | sequential | 1 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | sequential | 1 | shared_input | 1024 | 0.188957 | -22.73% | 19.636 |
| down | sequential | 1 | shared_output | 1024 | 0.188957 | -22.73% | 19.636 |
| down | sequential | 4 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | sequential | 4 | shared_input | 1408 | 0.000293 | -17.19% | 18.750 |
| down | sequential | 4 | shared_output | not met | 0.308193 | 6.25% | 15.000 |
| down | sequential | 8 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | sequential | 8 | shared_input | 1408 | 0.000293 | -9.17% | 17.467 |
| down | sequential | 8 | shared_output | not met | 0.368177 | 17.92% | 13.133 |
| down | sequential | 60 | individual_svd | 1024 | 0.188957 | -22.73% | 19.636 |
| down | sequential | 60 | shared_input | 1408 | 0.000293 | -1.15% | 16.183 |
| down | sequential | 60 | shared_output | not met | 0.453414 | 29.58% | 11.267 |
| gate | random | 1 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | random | 1 | shared_input | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | random | 1 | shared_output | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | random | 4 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | random | 4 | shared_input | not met | 0.301844 | 6.25% | 15.000 |
| gate | random | 4 | shared_output | 1408 | 0.000293 | -17.19% | 18.750 |
| gate | random | 8 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | random | 8 | shared_input | not met | 0.359657 | 17.92% | 13.133 |
| gate | random | 8 | shared_output | 1408 | 0.000293 | -9.17% | 17.467 |
| gate | random | 60 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | random | 60 | shared_input | not met | 0.443142 | 29.58% | 11.267 |
| gate | random | 60 | shared_output | 1408 | 0.000293 | -1.15% | 16.183 |
| gate | sequential | 1 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | sequential | 1 | shared_input | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | sequential | 1 | shared_output | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | sequential | 4 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | sequential | 4 | shared_input | not met | 0.301660 | 6.25% | 15.000 |
| gate | sequential | 4 | shared_output | 1408 | 0.000293 | -17.19% | 18.750 |
| gate | sequential | 8 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | sequential | 8 | shared_input | not met | 0.359838 | 17.92% | 13.133 |
| gate | sequential | 8 | shared_output | 1408 | 0.000293 | -9.17% | 17.467 |
| gate | sequential | 60 | individual_svd | 1024 | 0.186391 | -22.73% | 19.636 |
| gate | sequential | 60 | shared_input | not met | 0.443142 | 29.58% | 11.267 |
| gate | sequential | 60 | shared_output | 1408 | 0.000293 | -1.15% | 16.183 |
| up | random | 1 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | random | 1 | shared_input | 1408 | 0.000293 | -68.75% | 27.000 |
| up | random | 1 | shared_output | 1408 | 0.000293 | -68.75% | 27.000 |
| up | random | 4 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | random | 4 | shared_input | not met | 0.311583 | 6.25% | 15.000 |
| up | random | 4 | shared_output | 1408 | 0.000293 | -17.19% | 18.750 |
| up | random | 8 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | random | 8 | shared_input | not met | 0.371350 | 17.92% | 13.133 |
| up | random | 8 | shared_output | 1408 | 0.000293 | -9.17% | 17.467 |
| up | random | 60 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | random | 60 | shared_input | not met | 0.456837 | 29.58% | 11.267 |
| up | random | 60 | shared_output | 1408 | 0.000293 | -1.15% | 16.183 |
| up | sequential | 1 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | sequential | 1 | shared_input | 1408 | 0.000293 | -68.75% | 27.000 |
| up | sequential | 1 | shared_output | 1408 | 0.000293 | -68.75% | 27.000 |
| up | sequential | 4 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | sequential | 4 | shared_input | not met | 0.311216 | 6.25% | 15.000 |
| up | sequential | 4 | shared_output | 1408 | 0.000293 | -17.19% | 18.750 |
| up | sequential | 8 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | sequential | 8 | shared_input | not met | 0.371106 | 17.92% | 13.133 |
| up | sequential | 8 | shared_output | 1408 | 0.000293 | -9.17% | 17.467 |
| up | sequential | 60 | individual_svd | 1408 | 0.000293 | -68.75% | 27.000 |
| up | sequential | 60 | shared_input | not met | 0.456837 | 29.58% | 11.267 |
| up | sequential | 60 | shared_output | 1408 | 0.000293 | -1.15% | 16.183 |

## Conclusions

### gate: measured exponent-mantissa reference

The existing Base gives rho=0.194862 at 5.075 Base bpw on these weights.

- individual_svd, group 1: best rho=0.690137 at rank 256 within the reference Base bpw; first all-layer target at rank 1024 (rho=0.186391, reduction=-22.73%, Base=19.636 bpw).
- shared_input, group 4: best rho=0.808196 at rank 256 within the reference Base bpw; target not reached in the sweep.
- shared_output, group 4: best rho=0.797467 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-17.19%, Base=18.750 bpw).
- shared_input, group 8: best rho=0.718666 at rank 512 within the reference Base bpw; target not reached in the sweep.
- shared_output, group 8: best rho=0.826064 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-9.17%, Base=17.467 bpw).
- shared_input, group 60: best rho=0.776037 at rank 512 within the reference Base bpw; target not reached in the sweep.
- shared_output, group 60: best rho=0.875631 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-1.15%, Base=16.183 bpw).
At the reference storage budget, individual SVD does not reach rho <= 0.195 (rank 256, rho=0.690137).
No tested shared group reaches the target with positive Base parameter reduction across every selected layer.
At the same rank 256, group 4 has rho 0.808196 (input) and 0.797467 (output), versus 0.690137 for individual SVD. The shared basis retains less energy at this rank.

### up: measured exponent-mantissa reference

The existing Base gives rho=0.194705 at 5.076 Base bpw on these weights.

- individual_svd, group 1: best rho=0.710711 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-68.75%, Base=27.000 bpw).
- shared_input, group 4: best rho=0.828338 at rank 256 within the reference Base bpw; target not reached in the sweep.
- shared_output, group 4: best rho=0.815492 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-17.19%, Base=18.750 bpw).
- shared_input, group 8: best rho=0.737437 at rank 512 within the reference Base bpw; target not reached in the sweep.
- shared_output, group 8: best rho=0.841710 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-9.17%, Base=17.467 bpw).
- shared_input, group 60: best rho=0.794052 at rank 512 within the reference Base bpw; target not reached in the sweep.
- shared_output, group 60: best rho=0.883896 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-1.15%, Base=16.183 bpw).
At the reference storage budget, individual SVD does not reach rho <= 0.195 (rank 256, rho=0.710711).
No tested shared group reaches the target with positive Base parameter reduction across every selected layer.
At the same rank 256, group 4 has rho 0.828338 (input) and 0.815492 (output), versus 0.710711 for individual SVD. The shared basis retains less energy at this rank.

### down: measured exponent-mantissa reference

The existing Base gives rho=0.194696 at 5.081 Base bpw on these weights.

- individual_svd, group 1: best rho=0.705684 at rank 256 within the reference Base bpw; first all-layer target at rank 1024 (rho=0.188957, reduction=-22.73%, Base=19.636 bpw).
- shared_input, group 4: best rho=0.812813 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-17.19%, Base=18.750 bpw).
- shared_output, group 4: best rho=0.824180 at rank 256 within the reference Base bpw; target not reached in the sweep.
- shared_input, group 8: best rho=0.839530 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-9.17%, Base=17.467 bpw).
- shared_output, group 8: best rho=0.732771 at rank 512 within the reference Base bpw; target not reached in the sweep.
- shared_input, group 60: best rho=0.882776 at rank 256 within the reference Base bpw; first all-layer target at rank 1408 (rho=0.000293, reduction=-1.15%, Base=16.183 bpw).
- shared_output, group 60: best rho=0.789495 at rank 512 within the reference Base bpw; target not reached in the sweep.
At the reference storage budget, individual SVD does not reach rho <= 0.195 (rank 256, rho=0.705684).
No tested shared group reaches the target with positive Base parameter reduction across every selected layer.
At the same rank 256, group 4 has rho 0.812813 (input) and 0.824180 (output), versus 0.705684 for individual SVD. The shared basis retains less energy at this rank.

## Research answers

1. **Are individual experts low rank at the reference storage budget?** No tested projection reaches the target; see the individual SVD residuals and ranks above.
2. **Is there an effective shared low-rank subspace?** The rank-256 group-4 comparisons above show a larger residual than individual SVD. Shared directions exist, but the tested ranks do not preserve enough energy for the target with parameter savings.
3. **Can a shared basis meet rho <= 0.195 with parameter savings?** No tested shared group does so across the selected layers.
4. **What rank and reduction are needed?** The minimum-rank table gives the first FP16-factor rank meeting the target across every selected layer and its signed parameter reduction.
5. **Does it beat the exponent-mantissa Base?** No tested shared configuration reaches the target within the measured reference Base budget.

The individual SVD rows measure the best rank approximation per expert. The shared rows show the additional error from imposing one input or output basis across experts, and whether the target can be met with positive parameter reduction. All residual ratios above use squared-norm aggregation across groups and layers.

Skipped ranks beyond min(rows, cols): 18 entries; see `summary.json`. FP16 factor bpw counts Base parameters only. Projected QJL D=n and D=2n storage in the CSV/JSON is a size estimate, not measured QJL behavior.
