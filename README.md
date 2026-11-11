# QJLdecomposition

Standalone Git repository for QJL-based compression experiments on
`Qwen/Qwen1.5-MoE-A2.7B-Chat`. The main implementation supports a packed
2-bit codebook Base or a sign/exponent Base with a row-wise mantissa, plus a
packed QJL residual for every routed expert's Gate, Up, and Down projections.

This repository starts from the source tree of
[`jaewoogwak/ResCode`](https://github.com/jaewoogwak/ResCode),
`feature/codebook-base-qjl` at commit `2c0f706d4a3e76d32a62f561ad58b22b289c69ff`.
It has its own Git history and no ResCode Git remote or submodule. The Python
package remains named `rescode` so the experiment's imports continue to work.
The earlier ResCode project documentation is preserved in
[`README.rescode.md`](README.rescode.md).

## Environment

On Linux x86_64 with an NVIDIA GPU:

```bash
bash setup.sh
source .venv/bin/activate
```

The included `setup.sh` is identical to the one on ResCode's
`qjl-residual-scale` branch. It installs the pinned packages from
`requirements-rescode.lock` without cloning ResCode. Its optional environment
directory variable is still named `RESCODE_ENV_DIR` for compatibility.

Download the model revision expected by the scripts:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen1.5-MoE-A2.7B-Chat",
    revision="ec052fda178e241c7c443468d2fa1db6618996be",
)
PY
```

If the model is stored elsewhere, pass its snapshot directory with
`--model-path`.

## Compress and evaluate

For the exponent/mantissa Base with QJL dimension four times the input width:

```bash
.venv/bin/python scripts/QJL/run_codebook_base.py \
  --mode compress --base-method exponent-mantissa \
  --layers all --qjl-d-factor 4 \
  --output-dir results/local_4d/full --skip-eval

.venv/bin/python scripts/QJL/run_codebook_base.py \
  --mode eval \
  --checkpoint results/local_4d/full/compact_qjl_state.pt \
  --output-dir results/local_4d/full/eval_8tasks \
  --backend triton \
  --task winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte \
  --eval-limit 0 --eval-batch-size 16
```

Use `--base-method codebook` for the 2-bit codebook Base. See
`scripts/QJL/run_codebook_base.py --help` for other options. Generated `.pt`
checkpoints are ignored by Git; the committed result files contain diagnostics
and evaluation metrics.

## Tests and attribution

```bash
PYTHONPATH=. .venv/bin/python tests/test_codebook_base_qjl.py
PYTHONPATH=. .venv/bin/python tests/test_exponent_mantissa_base_qjl.py
```

The source code includes work derived from ResCode and its dependencies.
Retain the included [MIT license](LICENSE) and any file-level notices when
redistributing it. The original documentation lists additional upstream
attributions.
