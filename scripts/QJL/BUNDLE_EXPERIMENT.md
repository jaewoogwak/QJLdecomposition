# Majority-bundled QJL + low-rank residual

`run_bundle.py` reuses the ResCode group partition and the QJL projection,
packing, and Triton estimator. It leaves singleton experts as BF16 modules.
For each non-singleton expert it reconstructs the effective QJL base from the
**stored** group sign sketch and expert row norm, then fits a randomized FP32
SVD of `W - effective_base`. The factors are stored as BF16. This is a new
checkpoint format (`qjl-majority-lowrank-v1`); existing checkpoints are not
changed.

Modes:

- `--bundle-mode none --residual-mode lowrank`: individual sketch control A.
- `--bundle-mode majority --residual-mode exact --skip-eval`: diagnostic B;
  exact FP32 residuals are temporary and no checkpoint is saved.
- `--bundle-mode majority --residual-mode lowrank`: compressed method C.

If `--groups-path` is omitted, the script finds a compatible cached ResCode
group file. If none exists, it collects ResCode calibration statistics from
`--train-data-path` and calls `form_clusters`. The `--diagnostic-samples` flag
collects routed calibration activations for projection, expert, and MoE output
error measurements. Zero skips that additional pass and leaves those values
unmeasured. The script never changes grouping or performs training.

Example compression:

```bash
.venv/bin/python scripts/QJL/run_bundle.py \
  --mode compress --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --groups-path results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt \
  --layers all --qjl-d 2048 --rank 128 \
  --bundle-mode majority --residual-mode lowrank \
  --output-dir results/qjl_bundle/majority_d2048_r128 --skip-eval
```

Example evaluation:

```bash
.venv/bin/python scripts/QJL/run_bundle.py \
  --mode eval --model-path Qwen/Qwen1.5-MoE-A2.7B-Chat \
  --checkpoint results/qjl_bundle/majority_d2048_r128/bundle_qjl_state.pt \
  --output-dir results/qjl_bundle/majority_d2048_r128/eval \
  --backend triton --eval-limit 0 --eval-batch-size 16
```

The sweep script accepts `D_VALUES`, `RANKS`, `MODEL_PATH`, `GROUPS_PATH`,
`OUTPUT_DIR`, `TASKS`, `EVAL_LIMIT`, `EVAL_BATCH_SIZE`, `LAYERS`,
`DIAGNOSTIC_SAMPLES`, and `RUN_EVAL`. Each combination has separate logs and
outputs. `failures.tsv` records failed combinations without deleting successes.
Full checkpoint files can be many gigabytes and are ignored by Git.
