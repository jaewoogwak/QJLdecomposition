"""FP32-residual branch of the Base QJL dimension sweep.

This module is called by run_base_qjl_sweep.py only for --residual-mode fp32.
It copies the Base sketch from each B checkpoint instead of projecting it again.
"""

from __future__ import annotations

import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from rescode.evaluation import evaluate_fewshot
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.base_qjl_fp32_model import (
    FORMAT, FP32Residual, QJLBaseFP32Expert, load_checkpoint,
    resident_breakdown, save_checkpoint,
)
from rescode.qjl.base_qjl_model import (
    FORMAT as B_FORMAT, PROJECTIONS, DualQJLExpert, baseline_payload,
    residual_digest,
)
from rescode.qjl.codebook_model import restore_base
from rescode.qjl.core import PackedResidual, ProjectionBank
from rescode.qjl.exponent_mantissa_base import reconstruct_base_from_weight
from run_base_qjl_sweep import (
    _run, diagnostic_inputs, inspect_original_weight, load_model,
    parse_factors, rel_l2,
)

ROOT = Path(__file__).resolve().parents[2]
ERROR_FIELDS = ("d_factor", "layer", "expert", "projection", "samples",
                "base_rel_l2", "total_rel_l2_B", "total_rel_l2_C")


def _check_b(payload: dict, factor: int, baseline_hash: str, seed: int) -> None:
    config = payload.get("config", {})
    if (payload.get("format") != B_FORMAT or config.get("base_factor") != factor
            or config.get("base_seed") != seed
            or config.get("baseline_residual_sha256") != baseline_hash):
        raise ValueError("B checkpoint does not match the requested Base QJL sweep")


@torch.no_grad()
def convert(args, model_dir: Path) -> dict:
    baseline = baseline_payload(args.baseline_ckpt)
    baseline_hash = residual_digest(baseline)
    b_payload = torch.load(args.b_checkpoint, map_location="cpu", weights_only=False)
    _check_b(b_payload, args.d_factor, baseline_hash, args.base_seed)
    b_config = b_payload["config"]
    if b_config["residual_seed"] != baseline["config"]["projection_seed"]:
        raise ValueError("B and baseline residual projection seeds differ")
    layers = sorted(b_payload["layers"])
    if args.layers != "all":
        requested = sorted({int(v) for v in args.layers.split(",")})
        if any(layer not in layers for layer in requested):
            raise ValueError("requested layer missing from B checkpoint")
        layers = requested
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for Base QJL conversion")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    banks = {2048: ProjectionBank(2048 * args.d_factor, args.base_seed),
             1408: ProjectionBank(1408 * args.d_factor, args.base_seed)}
    residual_banks = {2048: ProjectionBank(4096, b_config["residual_seed"]),
                      1408: ProjectionBank(2816, b_config["residual_seed"])}
    data, errors, weight_count = {}, [], 0
    with ExitStack() as stack:
        handles = {}

        def original_weight(layer, expert_id, key):
            name = f"model.layers.{layer}.mlp.experts.{expert_id}.{key}_proj.weight"
            shard = index[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(
                    str(model_dir / shard), framework="pt", device="cpu"))
            shape = (2048, 1408) if key == "down" else (1408, 2048)
            return inspect_original_weight(handles[shard], name, shape, device)

        for layer in layers:
            layer_data = {}
            expert_ids = sorted(b_payload["layers"][layer])
            if args.max_experts is not None:
                expert_ids = expert_ids[:args.max_experts]
            for expert_id in expert_ids:
                x_hidden = diagnostic_inputs(layer, expert_id, 2048,
                                             args.diagnostic_samples, device)
                original, decoded_bases = {}, {}
                base_modules, dense_modules, b_residual_modules = {}, {}, {}
                parts = {}
                for key in PROJECTIONS:
                    a_part = baseline["layers"][layer][expert_id][key]
                    b_part = b_payload["layers"][layer][expert_id][key]
                    b_base = b_part["base"]
                    b_residual = b_part["residual"]
                    if (not torch.equal(b_residual["packed"], a_part["packed"])
                            or not torch.equal(b_residual["norms"], a_part["norms"])):
                        raise AssertionError("B residual QJL differs from the baseline")
                    w = original_weight(layer, expert_id, key)
                    base = restore_base(a_part["base"], device, "exponent-mantissa")
                    decoded = base.decode()
                    if not torch.equal(decoded, reconstruct_base_from_weight(w, base.beta)):
                        raise AssertionError("stored Base differs from BF16 weight and FP16 beta")
                    residual = (w.float() - decoded).contiguous()
                    if (residual.dtype != torch.float32 or residual.shape != w.shape
                            or not torch.allclose(decoded + residual, w.float(),
                                                   rtol=1e-6, atol=1e-6)):
                        raise AssertionError("W = Base + FP32 residual reconstruction failed")
                    input_dim = w.shape[1]
                    bank = banks[input_dim]
                    if (b_base["input_dim"] != input_dim or b_base["d"] != bank.d
                            or b_base["packed"].shape != (w.shape[0], (bank.d + 7) // 8)
                            or b_base["norms"].shape != (w.shape[0],)
                            or b_base["norms"].dtype != torch.float32):
                        raise ValueError("B Base QJL shape, norm, or D mismatch")
                    if not torch.allclose(torch.linalg.vector_norm(decoded, dim=1).cpu(),
                                          b_base["norms"], rtol=1e-6, atol=1e-7):
                        raise AssertionError("B Base row norms differ from the stored Base")
                    # No Base projection or sign packing occurs in the C conversion.
                    base_modules[key] = PackedResidual(
                        b_base["packed"].to(device), b_base["norms"].to(device),
                        input_dim, bank.d, bank, args.backend)
                    r_bank = residual_banks[input_dim]
                    b_residual_modules[key] = PackedResidual(
                        b_residual["packed"].to(device), b_residual["norms"].to(device),
                        input_dim, r_bank.d, r_bank, args.backend)
                    dense_modules[key] = FP32Residual(residual)
                    parts[key] = {"base": b_base, "residual": residual.cpu()}
                    original[key], decoded_bases[key] = w, decoded
                    weight_count += w.numel()
                gate = F.linear(x_hidden, original["gate"])
                up = F.linear(x_hidden, original["up"])
                x_down = F.silu(gate) * up
                for key in PROJECTIONS:
                    x = x_down if key == "down" else x_hidden
                    base_ref = F.linear(x.float(), decoded_bases[key])
                    weight_ref = F.linear(x.float(), original[key].float())
                    base_out = base_modules[key](x)
                    b_out = base_out + b_residual_modules[key](x)
                    dense_ref = x.float() @ dense_modules[key].weight.T
                    if not torch.allclose(dense_modules[key].forward_fp32(x), dense_ref,
                                          rtol=1e-5, atol=1e-5):
                        raise AssertionError("FP32 residual F.linear differs from reference")
                    c_out = base_out + dense_modules[key](x)
                    errors.append({"d_factor": args.d_factor, "layer": layer,
                                   "expert": expert_id, "projection": key,
                                   "samples": args.diagnostic_samples,
                                   "base_rel_l2": rel_l2(base_out, base_ref),
                                   "total_rel_l2_B": rel_l2(b_out, weight_ref),
                                   "total_rel_l2_C": rel_l2(c_out, weight_ref)})
                b_expert = DualQJLExpert(base_modules, b_residual_modules)
                c_expert = QJLBaseFP32Expert(base_modules, dense_modules)
                exact = F.linear(x_down, original["down"])
                errors.append({"d_factor": args.d_factor, "layer": layer,
                               "expert": expert_id, "projection": "expert_output",
                               "samples": args.diagnostic_samples,
                               "base_rel_l2": None,
                               "total_rel_l2_B": rel_l2(b_expert(x_hidden), exact),
                               "total_rel_l2_C": rel_l2(c_expert(x_hidden), exact)})
                layer_data[expert_id] = parts
                del original, decoded_bases, base_modules, dense_modules, b_residual_modules
            data[layer] = layer_data
            print(f"[base-qjl-fp32] D_B={args.d_factor}x layer={layer} experts={len(layer_data)}",
                  flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    config = {"model": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "model_path": str(model_dir),
              "baseline_checkpoint": str(args.baseline_ckpt),
              "b_checkpoint": str(args.b_checkpoint),
              "baseline_residual_sha256": baseline_hash,
              "base_method": "qjl", "residual_method": "fp32",
              "base_factor": args.d_factor, "base_seed": args.base_seed,
              "input_dims": {"gate": 2048, "up": 2048, "down": 1408},
              "output_dims": {"gate": 1408, "up": 1408, "down": 2048},
              "layers": layers, "max_experts": args.max_experts,
              "weight_count": weight_count, "diagnostic_samples": args.diagnostic_samples}
    payload = {"format": FORMAT, "config": config, "layers": data}
    sizes = save_checkpoint(payload, args.checkpoint)
    reloaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for layer, experts in data.items():
        for expert_id, projections in experts.items():
            for key, part in projections.items():
                loaded = reloaded["layers"][layer][expert_id][key]
                b_base = b_payload["layers"][layer][expert_id][key]["base"]
                if (not torch.equal(loaded["base"]["packed"], b_base["packed"])
                        or not torch.equal(loaded["base"]["norms"], b_base["norms"])
                        or loaded["residual"].dtype != torch.float32
                        or not torch.equal(loaded["residual"], part["residual"])):
                    raise AssertionError("saved C checkpoint changed Base QJL or FP32 residual")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(errors)
    summary = {**config, **sizes,
               "base_bits_per_weight": 8 * (sizes["base_packed_bytes"] + sizes["base_norm_bytes"]) / weight_count,
               "fp32_residual_bits_per_weight": 8 * sizes["fp32_residual_bytes"] / weight_count,
               "checkpoint_bits_per_weight": 8 * sizes["checkpoint_bytes"] / weight_count,
               "bf16_saving_pct": 100 * (1 - sizes["checkpoint_bytes"] / (2 * weight_count))}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@torch.no_grad()
def evaluate(args, model_dir: Path) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_dir, args.gpu_memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    payload = load_checkpoint(model, args.checkpoint, args.backend,
                              args.fp32_residual_device)
    config = payload["config"]
    del payload
    metrics = {"checkpoint": str(args.checkpoint),
               "checkpoint_bytes": args.checkpoint.stat().st_size,
               "checkpoint_config": config, "tasks": args.tasks,
               "eval_limit": args.eval_limit, "backend": args.backend}
    if not args.skip_eval:
        result_path = args.output_dir / "lm_eval.txt"
        results = evaluate_fewshot(model, tokenizer, args.tasks, 0,
            eval_batch_size=args.eval_batch_size, output_path=str(result_path),
            limit=args.eval_limit or None)
        metrics["task_summary"] = report_eval_summary(
            results, args.tasks, str(result_path),
            header="QJL Base + FP32 Residual evaluation")
        metrics["task_results"] = results.get("results", {})
    metrics.update(resident_breakdown(model))
    if torch.cuda.is_available():
        metrics["gpu_process_allocated_bytes"] = [torch.cuda.memory_allocated(i)
                                                   for i in range(torch.cuda.device_count())]
    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n")
    return metrics


def _read_metrics(path: Path, tasks: str, limit: int) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if (data.get("tasks") != tasks or data.get("eval_limit") != limit
            or "task_summary" not in data):
        return None
    return data


def _b_sizes(folder: Path, checkpoint: Path) -> dict:
    metrics_path = folder / "metrics.json"
    if metrics_path.is_file():
        sizes = json.loads(metrics_path.read_text())
        if "base_packed_bytes" not in sizes and "weight_count" in sizes:
            experts = sizes["weight_count"] // (3 * 1408 * 2048)
            norm_bytes = experts * (1408 + 1408 + 2048) * 4
            sizes["base_norm_bytes"] = norm_bytes
            sizes["base_packed_bytes"] = sizes["base_packed_and_norm_bytes"] - norm_bytes
            sizes["residual_norm_bytes"] = norm_bytes
            sizes["residual_packed_bytes"] = sizes["residual_packed_and_norm_bytes"] - norm_bytes
        return sizes
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base_packed = base_norms = residual_packed = residual_norms = 0
    for experts in payload["layers"].values():
        for parts in experts.values():
            for part in parts.values():
                base, residual = part["base"], part["residual"]
                base_packed += base["packed"].numel()
                base_norms += base["norms"].numel() * 4
                residual_packed += residual["packed"].numel()
                residual_norms += residual["norms"].numel() * 4
    return {"base_packed_bytes": base_packed, "base_norm_bytes": base_norms,
            "residual_packed_and_norm_bytes": residual_packed + residual_norms,
            "metadata_bytes": (checkpoint.stat().st_size - base_packed - base_norms
                               - residual_packed - residual_norms)}


def _validate_c_against_b(c_path: Path, b_path: Path, factor: int,
                          baseline_hash: str, base_seed: int) -> dict:
    c = torch.load(c_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    _check_b(b, factor, baseline_hash, base_seed)
    config = c.get("config", {})
    if (c.get("format") != FORMAT or config.get("base_factor") != factor
            or config.get("base_seed") != base_seed
            or config.get("baseline_residual_sha256") != baseline_hash):
        raise ValueError(f"existing C checkpoint is incompatible: {c_path}")
    for layer, experts in c["layers"].items():
        for expert_id, projections in experts.items():
            for key, part in projections.items():
                reference = b["layers"][layer][expert_id][key]["base"]
                base = part["base"]
                if (not torch.equal(base["packed"], reference["packed"])
                        or not torch.equal(base["norms"], reference["norms"])):
                    raise AssertionError(f"Base QJL differs from B: D={factor}, "
                                         f"layer={layer}, expert={expert_id}, {key}")
                if part["residual"].dtype != torch.float32:
                    raise AssertionError("C checkpoint lost FP32 residual")
    return config


def sweep(args, model_dir: Path) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    b_root = args.b_output_dir
    baseline_path = args.baseline_ckpt or (b_root / "baseline" / "compact_qjl_state.pt")
    if not baseline_path.is_file():
        if args.baseline_ckpt:
            raise FileNotFoundError(f"BASELINE_CKPT missing: {baseline_path}")
        baseline_path = args.output_dir / "baseline" / "compact_qjl_state.pt"
        if not baseline_path.is_file():
            _run([sys.executable, "scripts/QJL/run_codebook_base.py", "--mode", "compress",
                  "--base-method", "exponent-mantissa", "--model-path", str(model_dir),
                  "--layers", args.layers, "--qjl-d-factor", "2",
                  "--projection-seed", str(args.residual_seed), "--backend", args.backend,
                  "--output-dir", str(baseline_path.parent), "--skip-eval"])
    args.baseline_ckpt = baseline_path
    baseline = baseline_payload(baseline_path)
    if baseline["config"]["projection_seed"] != args.residual_seed:
        raise ValueError("baseline residual seed differs from requested seed")
    layers = sorted(baseline["layers"])
    if args.layers == "all" and layers != list(range(24)):
        raise ValueError("full sweep requires all 24 baseline layers")
    if args.layers != "all" and layers != sorted({int(v) for v in args.layers.split(",")}):
        raise ValueError("baseline layers differ from selected layers")
    baseline_hash = residual_digest(baseline)
    del baseline
    if args.max_experts is not None and not args.skip_eval:
        raise ValueError("partial expert sweep cannot run task evaluation")
    factors = parse_factors(args.d_factors)
    summary_rows, all_errors = [], []
    for factor in factors:
        b_folder = b_root / f"base_d{factor}"
        b_path = b_folder / "compact_qjl_state.pt"
        if not b_path.is_file():
            # Produce a B reference in C's own directory; existing B files remain untouched.
            b_folder = args.output_dir / "_b_reference" / f"base_d{factor}"
            b_path = b_folder / "compact_qjl_state.pt"
            if not b_path.is_file():
                _run([sys.executable, "scripts/QJL/run_base_qjl_sweep.py", "--mode", "convert",
                      "--residual-mode", "qjl", "--model-path", str(model_dir),
                      "--baseline-ckpt", str(baseline_path), "--output-dir", str(b_folder),
                      "--checkpoint", str(b_path), "--d-factor", str(factor),
                      "--base-seed", str(args.base_seed), "--layers", args.layers,
                      "--backend", args.backend, "--diagnostic-samples",
                      str(args.diagnostic_samples), "--qjl-row-chunk", str(args.qjl_row_chunk)]
                    + (["--max-experts", str(args.max_experts)] if args.max_experts else []))
        c_folder = args.output_dir / f"base_d{factor}"
        c_path = c_folder / "compact_qjl_state.pt"
        if not c_path.is_file():
            _run([sys.executable, "scripts/QJL/run_base_qjl_sweep.py", "--mode", "convert",
                  "--residual-mode", "fp32", "--model-path", str(model_dir),
                  "--baseline-ckpt", str(baseline_path), "--b-checkpoint", str(b_path),
                  "--output-dir", str(c_folder), "--checkpoint", str(c_path),
                  "--d-factor", str(factor), "--base-seed", str(args.base_seed),
                  "--layers", args.layers, "--backend", args.backend,
                  "--diagnostic-samples", str(args.diagnostic_samples)]
                + (["--max-experts", str(args.max_experts)] if args.max_experts else []))
        elif not (c_folder / "errors.csv").is_file():
            raise FileNotFoundError(f"C checkpoint exists without error diagnostics: {c_folder}")
        c_config = _validate_c_against_b(c_path, b_path, factor, baseline_hash,
                                         args.base_seed)
        with (c_folder / "errors.csv").open() as f:
            factor_errors = list(csv.DictReader(f))
        all_errors.extend(factor_errors)
        projection_errors = [row for row in factor_errors if row["projection"] in PROJECTIONS]
        expert_errors = [row for row in factor_errors if row["projection"] == "expert_output"]
        def mean_error(rows, key):
            return sum(float(row[key]) for row in rows) / len(rows) if rows else None
        b_metrics_path = b_folder / "eval_all" / "eval_metrics.json"
        b_metrics = _read_metrics(b_metrics_path, args.tasks, args.eval_limit)
        if not args.skip_eval and b_metrics is None:
            b_metrics_path = args.output_dir / "_b_eval" / f"base_d{factor}" / "eval_metrics.json"
            b_metrics = _read_metrics(b_metrics_path, args.tasks, args.eval_limit)
            if b_metrics is None:
                _run([sys.executable, "scripts/QJL/run_base_qjl_sweep.py", "--mode", "eval",
                      "--residual-mode", "qjl", "--model-path", str(model_dir),
                      "--checkpoint", str(b_path), "--output-dir", str(b_metrics_path.parent),
                      "--tasks", args.tasks, "--eval-limit", str(args.eval_limit),
                      "--eval-batch-size", str(args.eval_batch_size), "--backend", args.backend])
                b_metrics = _read_metrics(b_metrics_path, args.tasks, args.eval_limit)
        c_metrics_path = c_folder / "eval_all" / "eval_metrics.json"
        c_metrics = _read_metrics(c_metrics_path, args.tasks, args.eval_limit)
        if not args.skip_eval and c_metrics is None:
            _run([sys.executable, "scripts/QJL/run_base_qjl_sweep.py", "--mode", "eval",
                  "--residual-mode", "fp32", "--model-path", str(model_dir),
                  "--checkpoint", str(c_path), "--output-dir", str(c_metrics_path.parent),
                  "--tasks", args.tasks, "--eval-limit", str(args.eval_limit),
                  "--eval-batch-size", str(args.eval_batch_size), "--backend", args.backend,
                  "--fp32-residual-device", args.fp32_residual_device])
            c_metrics = _read_metrics(c_metrics_path, args.tasks, args.eval_limit)
        b_sizes = _b_sizes(b_folder, b_path)
        c_sizes = json.loads((c_folder / "metrics.json").read_text())
        for method, path, sizes, metrics in (("B", b_path, b_sizes, b_metrics),
                                             ("C", c_path, c_sizes, c_metrics)):
            tasks = {f"acc_{name}": acc for name, acc, _ in
                     metrics["task_summary"]["tasks"]} if metrics else {}
            summary_rows.append({"method": method, "d_factor": factor,
                **{f"acc_{name}": tasks.get(f"acc_{name}") for name in args.tasks.split(",")},
                "avg_accuracy": metrics["task_summary"]["avg"] if metrics else None,
                "delta_avg_vs_B": (metrics["task_summary"]["avg"] - b_metrics["task_summary"]["avg"]
                                   if method == "C" and metrics and b_metrics else
                                   0.0 if method == "B" and metrics else None),
                "base_rel_l2_mean": mean_error(projection_errors, "base_rel_l2"),
                "total_rel_l2_mean": mean_error(
                    projection_errors, "total_rel_l2_B" if method == "B" else "total_rel_l2_C"),
                "expert_output_rel_l2_mean": mean_error(
                    expert_errors, "total_rel_l2_B" if method == "B" else "total_rel_l2_C"),
                "checkpoint_bytes": path.stat().st_size,
                "bits_per_weight": 8 * path.stat().st_size / c_config["weight_count"],
                "saving_vs_bf16_pct": 100 * (1 - path.stat().st_size /
                                             (2 * c_config["weight_count"])),
                "base_packed_bytes": sizes.get("base_packed_bytes"),
                "base_norm_bytes": sizes.get("base_norm_bytes"),
                "residual_packed_bytes": sizes.get("residual_packed_bytes"),
                "residual_norm_bytes": sizes.get("residual_norm_bytes"),
                "residual_stored_bytes": sizes.get("fp32_residual_bytes",
                                                   sizes.get("residual_packed_and_norm_bytes")),
                "metadata_bytes": sizes.get("metadata_bytes"),
                "gpu_resident_compact_bytes": (metrics or {}).get("gpu_resident_compact_total_bytes"),
                "gpu_process_allocated_total_bytes": (
                    sum(metrics.get("gpu_process_allocated_bytes", [])) if metrics else None),
                "fp32_residual_gpu_bytes": (metrics or {}).get("fp32_residual_gpu_bytes"),
                "fp32_residual_cpu_bytes": (metrics or {}).get("fp32_residual_cpu_bytes"),
                "base_projection_gpu_bytes": (metrics or {}).get("base_projection_gpu_bytes")})
    with (args.output_dir / "errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(all_errors)
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary_rows)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "residual_mode": "fp32", "tasks": args.tasks, "eval_limit": args.eval_limit,
        "b_output_dir": str(b_root), "baseline_checkpoint": str(baseline_path),
        "d_factors": factors, "rows": summary_rows}, indent=2) + "\n")
    print("method D_B avg_acc delta_vs_B bits/weight checkpoint_GiB")
    for row in summary_rows:
        acc = "pending" if row["avg_accuracy"] is None else f"{row['avg_accuracy']:.4f}"
        delta = "pending" if row["delta_avg_vs_B"] is None else f"{row['delta_avg_vs_B']:+.4f}"
        print(f"{row['method']:>6} {row['d_factor']:>3} {acc:>7} {delta:>10} "
              f"{row['bits_per_weight']:.3f} {row['checkpoint_bytes']/2**30:.2f}")


def run(args, model_dir: Path):
    if args.mode == "convert":
        if args.baseline_ckpt is None or args.b_checkpoint is None or args.checkpoint is None:
            raise ValueError("FP32 convert needs --baseline-ckpt, --b-checkpoint, --checkpoint")
        return convert(args, model_dir)
    if args.mode == "eval":
        if args.checkpoint is None:
            raise ValueError("FP32 eval needs --checkpoint")
        return evaluate(args, model_dir)
    return sweep(args, model_dir)
