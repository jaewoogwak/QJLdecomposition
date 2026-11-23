# Shared Base + Low-rank Residual + QJL Tail

Run from the repository root with the CUDA environment active. The script is
`scripts/QJL/run_shared_lowrank_qjl.py`. It writes one row per
layer/expert/projection/rank, a summary, three figures, and `REPORT.md`.

## Grouping and activation cache

The experiment needs ResCode group labels and saliency. Pass a trusted `.pt` file
containing `group_state_dict` and `saliency_state_dict`, keyed by
`model.layers.<layer>.mlp`, or a directory containing `group_state_dict.pt` and
`saliency_state_dict.pt`. An optional `routed_samples` entry (or
`routed_samples.pt`) contains the `train` result of
`rescode.qjl.calibration.collect_projection_samples`. These inputs take priority
over deterministic Gaussian diagnostic inputs, which are explicitly marked
`synthetic` in the output.

If no grouping cache is available, pass `--calibration-data PATH` to compute
output descriptors, saliency, singleton experts, and groups with the existing
ResCode Qwen grouper. This loads the original HF model and writes
`grouping_cache.pt` for later modes. The default singleton ratio is 0.15 and the
default group count is 30. An existing label cache determines the grouping; the
CLI group settings apply when generating new labels.

## Weight-only analysis

```bash
.venv/bin/python scripts/QJL/run_shared_lowrank_qjl.py \
  --mode analyze \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --grouping-cache /path/to/grouping_cache.pt \
  --layers all --projections gate,up,down \
  --ranks 0,32,64,128,256,512 --qjl-d 4096 \
  --output-dir results/shared_lowrank_qjl/analysis
```

For a small run, add `--layers 0 --max-experts 4 --ranks 0,32` and use a cache
that covers those experts. `analyze` does not write checkpoints.

## Compression

```bash
.venv/bin/python scripts/QJL/run_shared_lowrank_qjl.py \
  --mode compress \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --grouping-cache /path/to/grouping_cache.pt \
  --layers all --projections gate,up,down \
  --ranks 0,32,64,128,256,512 --qjl-d 4096 \
  --output-dir results/shared_lowrank_qjl/compressed
```

Each `checkpoints/rank_R/{A,B,C}` directory contains a manifest and one layer
file per selected layer. A stores shared BF16 bases and QJL(R); B stores the same
bases and BF16 low-rank factors; C stores both factors and QJL(E). Singleton
expert weights are BF16 in every case. All three use the same groups and the same
Gaussian projection seed. The storage summary uses actual serialized file sizes
and counts the selected expert weights against their BF16 original size. It does
not count the unchanged HF backbone, router, or shared expert.

## Full zero-shot evaluation

```bash
.venv/bin/python scripts/QJL/run_shared_lowrank_qjl.py \
  --mode eval \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --checkpoint results/shared_lowrank_qjl/compressed/checkpoints/rank_128/C/manifest.json \
  --tasks arc_easy,arc_challenge --eval-limit 0 \
  --output-dir results/shared_lowrank_qjl/eval_rank_128_C
```

`--eval-limit 0` evaluates the complete selected tasks. The loader overlays the
expert checkpoint on the original Qwen model. Original routing, shared expert,
and SwiGLU remain in use. Select an A or B manifest to evaluate those cases.

The experiment's FP32 SVD energy is a weight Frobenius diagnostic. QJL metrics
compare output vectors on a common input. The Tail is always computed after
reconstructing the stored BF16 factors.
