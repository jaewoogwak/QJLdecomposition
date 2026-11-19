# Majority-bundled QJL results

Qwen1.5-MoE-A2.7B-Chat, all 24 MoE layers, ResCode grouping cache
`groups-0c0fb7aea36af7a8.pt`, majority signs, D=2048, BF16 low-rank factors.
ARC scores are zero-shot accuracy with `--eval-limit 0` (the full evaluation
set), using the Triton backend.

| Rank | ARC-Easy acc | ARC-Challenge acc | Checkpoint bytes | Routed BF16 reduction |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 0.281987 | 0.212457 | 13,012,452,659 | 47.77% |
| 128 | 0.348064 | 0.222696 | 14,005,127,135 | 43.79% |
| 512 | 0.649832 | 0.361775 | 19,961,186,247 | 19.88% |

The `full_majority_d2048_r64/`, `full_majority_d2048_r128/`, and
`full_majority_d2048_r512/` directories
contain the compression metrics, per-expert diagnostics, and `eval_arc/`
results. The `smoke_*` directories contain single-layer validation runs.
All JSON, CSV, and evaluation logs are versioned here. The large
`bundle_qjl_state.pt` checkpoints remain on the GPU server and are omitted
because they exceed GitHub's file size limit. The original Qwen model is
required along with a checkpoint to rerun inference. No eight-task evaluation
was performed for these runs.
