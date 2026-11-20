# Shared low-rank Base diagnostics

`diagnose_shared_rank_base.py` reads BF16 expert matrices directly from indexed
safetensors. It does not load the model, calibrate, encode QJL, or run inference.
For each layer and projection it groups the selected experts independently.
The `rank_aware` merge uses the increase in rank-constrained residual cost;
`cosine` merges the nearest group sums by weight cosine; `random` makes a seeded
balanced partition. `random_size_matched` shuffles experts into the sizes found
by `rank_aware`. Each target rank gets a separate rank-aware clustering.

The default candidate shortlist evaluates the eight cosine-nearest active group
pairs by exact residual cost at each merge. This is **approximate greedy** because
the lowest-cost pair can fall outside the shortlist. Set
`--candidate-shortlist 0` for exhaustive greedy clustering. SVDs are computed
in FP32; group sums, costs, and singular-value spectra are cached across group
counts and ranks for each layer/projection. Ties use sorted expert IDs. This is
an exact truncated SVD objective for every evaluated pair;
the shortlist changes which pairs are evaluated.

Run the default sweep on a GPU with the selected checkpoint shards present:

```bash
python scripts/QJL/diagnose_shared_rank_base.py \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --device cuda --candidate-shortlist 8
```

For the same defaults limited to Layer 12, add `--layers 12` to that command.

For a smaller run:

```bash
python scripts/QJL/diagnose_shared_rank_base.py \
  --model-path /home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat \
  --layers 12 --projections gate --group-counts 4,8 \
  --ranks 0,32,64 --device cuda --seed 0 --qjl-d 1024
```

`--layers`, `--experts`, `--projections`, `--group-counts`, `--ranks`, and
`--methods` accept comma-separated values. `--output-dir` defaults to
`results/shared_rank_diagnostics`. The files are updated after every
configuration: `summary.csv`, `summary.json`, `expert_metrics.csv`, and
`groups.json`. A PNG plot is saved for every completed layer/projection.

`rho_global` uses the sum of squared residual norms divided by the sum of
squared weight norms. The per-expert distribution is reported separately.
`bf16_base_bits` estimates factor storage as `16 * C * r_eff * (rows + cols)`;
it excludes metadata. `--qjl-d D` adds a **theoretical** residual storage
estimate using the existing packed sign bits plus FP32 row norms convention.
No QJL data is produced. `clustering_seconds` and `diagnostics_seconds` are
wall times for this weight-only calculation, not inference latency.
`peak_gpu_memory_bytes` is the PyTorch allocator peak during diagnostic
reconstruction. The per-expert exponent-mantissa reference calls the existing
encoder and decoder; use `--skip-exponent-reference` to omit it.

CPU tests:

```bash
python -m unittest tests.test_shared_rank_diagnostics -v
```

This diagnostic measures residual norms and storage estimates. It does not
establish QJL accuracy or downstream task quality.
