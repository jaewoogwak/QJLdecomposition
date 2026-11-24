# D=2n residual sketch sharing: ARC rerun

Both evaluations use Qwen/Qwen1.5-MoE-A2.7B-Chat revision
`ec052fda178e241c7c443468d2fa1db6618996be`, all 24 MoE layers,
the exponent-mantissa Base, QJL projection seed 0, the Triton backend,
zero-shot `arc_easy,arc_challenge`, no evaluation limit, and batch size 64.
The Individual rerun uses the existing `run_codebook_base.py` path.
The majority run uses similarity pairing with pairing and bundle seeds 0.

| Metric | Individual QJL D=2n | Majority bundle D=2n |
| --- | ---: | ---: |
| ARC-Easy accuracy | 0.649411 | 0.680976 |
| ARC-Easy normalized accuracy | 0.629630 | 0.648569 |
| ARC-Challenge accuracy | 0.363481 | 0.375427 |
| ARC-Challenge normalized accuracy | 0.390785 | 0.398464 |
| Checkpoint bytes | 11,059,112,989 | 9,501,328,917 |

The scores are in
[`../../qjl_exponent_mantissa_base_2d/rerun/eval/metrics.json`](../../qjl_exponent_mantissa_base_2d/rerun/eval/metrics.json)
and
[`majority_d2n_similarity/eval/metrics.json`](majority_d2n_similarity/eval/metrics.json).
Compression metrics and diagnostics are stored beside the evaluation files.

The `.pt` checkpoints remain on the server at the paths recorded in each
`metrics.json`. They exceed ordinary GitHub file size limits and are excluded
from this commit, consistent with the repository's earlier result records.
The interrupted `analysis.log` and failed preliminary evaluation logs are
excluded. No full eight-task evaluation is claimed here.

## Full synthetic-input analysis

[`majority_d2n_similarity/analysis_full/`](majority_d2n_similarity/analysis_full/)
contains the completed independent analysis of all 24 layers, three
projections, and 30 pairs per projection. Its 12,960 expert-method rows compare
Individual QJL, majority, and magnitude-aware on the same synthetic inputs.
The majority sketch is read from the checkpoint; the other sketches are
reconstructed using its saved pair map and seeds.

| Mean over expert-method rows | Individual | Majority | Magnitude-aware |
| --- | ---: | ---: | ---: |
| Bit preservation | 1.0000 | 0.7503 | 0.7503 |
| Residual output relative L2 error | 0.8855 | 1.1333 | 1.1703 |
| Full weight output relative L2 error | 0.1725 | 0.2208 | 0.2280 |
| Full output cosine similarity | 0.9854 | 0.9758 | 0.9745 |

The mean pre-bundle bit agreement is 0.5006. These are synthetic per-projection
metrics, not downstream task accuracy; the ARC scores above are separate
evaluations.
