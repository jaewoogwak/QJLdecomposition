# Individual versus majority QJL: full Qwen experiment

Qwen1.5-MoE-A2.7B-Chat, all 24 layers, the saved ResCode grouping cache, D=2048, projection seed 0, randomized SVD seed 42, oversampling 8, two iterations. Both modes use the same original BF16 expert weights, expert-specific FP32 row norms, grouping, routing, and BF16 singleton experts. Across the cache, 748 experts are non-singletons and 28 or 29 experts per layer are singletons. Gate, Up, and Down were evaluated separately. The ARC results are full zero-shot evaluations (`eval-limit=0`) using Triton.

## Downstream accuracy and storage

Checkpoint sizes below are the **actual saved routed-expert compression checkpoints**, including packed signs, row norms, low-rank factors, singleton weights, and serialization overhead. The original routed BF16 weights are approximately 24.91 GB. These checkpoints do not include the rest of the Hugging Face model.

| Rank | Mode | ARC-Easy (%) | ARC-Challenge (%) | Checkpoint (GB) | Bits/routed weight | Saving versus routed BF16 (%) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | Majority | 28.20 | 21.25 | 13.012 | 8.357 | 47.77 |
| 64 | Individual | 36.70 | 22.27 | 13.915 | 8.936 | 44.15 |
| 128 | Majority | 34.81 | 22.27 | 14.005 | 8.994 | 43.79 |
| 128 | Individual | 50.55 | 28.41 | 14.907 | 9.574 | 40.17 |
| 512 | Majority | 64.98 | 36.18 | 19.961 | 12.819 | 19.88 |
| 512 | Individual | 68.43 | 37.37 | 20.863 | 13.399 | 16.26 |

Individual sketch storage adds 0.902 GB or 0.579 bits per routed weight at every tested rank. Individual accuracy is higher for both ARC tasks at all three ranks. The largest observed difference is 15.74 percentage points on ARC-Easy at rank 128. The rank 64 ARC-Challenge difference is 1.02 points, so this result alone is less decisive than the consistent direction across all configurations.

## Full residual spectra

The full FP32 SVD was computed for all 4,488 residual matrices: 748 non-singleton experts × three projections × two modes. The table shows unweighted means over the 2,244 expert/projection matrices in each mode. `rho` is `||R||_F / ||W||_F`; `r90/r95/r99` are ranks required to retain the stated fraction of residual energy. Optimal errors are the full-SVD tail norm divided by `||W||_F`; they are **lower bounds**, not the errors of the saved randomized SVD factors.

| Mode | rho | r90 | r95 | r99 | Energy at r128 | Optimal weight error r128 | Optimal weight error r512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Individual | 1.181 | 687.7 | 871.4 | 1166.8 | 0.419 | 0.900 | 0.496 |
| Majority | 1.661 | 720.7 | 895.3 | 1177.8 | 0.355 | 1.333 | 0.745 |

Individual QJL has a smaller residual norm in the aggregate, faster singular-value decay, and lower optimal weight reconstruction error. Majority bundling raises mean rho by 0.480, raises r90 by 33.0 and r95 by 23.9, and raises the rank-128 optimal weight error by 0.433. The direction holds separately for Gate, Up, and Down in `spectrum_summary.json`.

The **actual** randomized SVD with stored BF16 factors has additional mean weight-relative error above the full-SVD optimum: Individual +0.013/+0.016/+0.016 at ranks 64/128/512, Majority +0.019/+0.026/+0.025. The joined per-matrix values are in `actual_vs_optimal.csv`. These actual and optimal quantities are kept separate throughout the outputs.

## Validation and limitations

- All nine unit tests passed, including synthetic full-SVD energy versus direct truncated SVD, base/residual reconstruction, individual sign storage and reload, routing/singleton preservation, and Triton/reference parity.
- Layer-12 smoke compression, checkpoint reload, limited ARC evaluation, and two-mode spectrum plotting passed before the full runs.
- All three individual full checkpoints report zero maximum logit difference after save/reload. In layers 0, 12, and 23, sampled Gate/Up/Down individual packed sketches and row norms exactly match sketches regenerated from original BF16 weights; saved group membership equals the ResCode cache and majority checkpoint.
- The maximum full-SVD squared-energy discrepancy is 3.19e-4 relative, within the 1e-3 FP32 numerical tolerance. Maximum `W - (B + R)` relative discrepancy is 3.52e-8. Cumulative energy is monotonic and ends at one, and all records satisfy `r90 <= r95 <= r99`.
- The experiment does not equalize saved checkpoint size between modes; Individual pays the measured sketch storage premium. Accuracy comparisons therefore isolate the bundling operation at the same D and rank, while the storage column exposes the cost difference.
- No majority checkpoint or majority evaluation was recomputed. The analysis reads them and records input fingerprints in `spectrum_summary.json`.

## Reproduction

```bash
cd /home/jaewoo/QJLdecomposition
.venv/bin/python -m unittest tests.test_qjl_spectrum tests.test_qjl_bundle -v
LAYERS=12 RANKS=64 EVAL_LIMIT=2 EVAL_BATCH_SIZE=4 bash scripts/QJL/run_individual_sweep.sh
LAYERS=12 MAX_EXPERTS=2 OUTPUT_DIR=results/qjl_bundle/spectrum_comparison_smoke bash scripts/QJL/run_spectrum_comparison.sh
RANKS='64 128 512' LAYERS=all EVAL_LIMIT=0 EVAL_BATCH_SIZE=16 bash scripts/QJL/run_individual_sweep.sh
bash scripts/QJL/run_spectrum_comparison.sh
```

The scripts reuse matching existing checkpoints and saved per-layer spectra. The full spectrum run writes `singular_values.npz`, `spectrum_metrics.csv`, `spectrum_summary.json`, `actual_vs_optimal.csv`, `evaluation.csv`, per-layer resume files, and 54 PNG plus 54 PDF plots. The individual checkpoints are too large for GitHub and remain on the server; their metrics, diagnostics, and evaluations are committed separately.
