# Exponent-Mantissa Base + QJL, D factor 2

- `full/` contains the first compression diagnostics and `eval_all/` (ARC Easy and ARC Challenge).
- `full/rebuilt/` contains the second compression diagnostics. Its evaluation files are in `full/eval_8tasks/`.
- `full/eval_8tasks/` contains these completed tasks: arc_easy, arc_challenge, mmlu, openbookqa, rte. The directory name does not indicate eight completed tasks.
- `rerun/` contains a new D=2n Individual QJL compression and full ARC Easy/Challenge evaluation at batch size 64. Its local checkpoint is `rerun/compact_qjl_state.pt`.
- The `.pt` checkpoints exceed ordinary GitHub file size limits and are not committed.
- The two compression runs have identical aggregate Frobenius statistics and Base/QJL tensor byte counts. Timing records are kept separately.
