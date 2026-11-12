#!/usr/bin/env python3
"""Sweep QJL dimension on the Base while reusing one exponent/mantissa residual checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rescode.evaluation import evaluate_fewshot
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.base_qjl_model import (
    FORMAT, PROJECTIONS, DualQJLExpert, baseline_payload, encode_qjl_base,
    gpu_resident_breakdown, load_checkpoint, packed_bytes, residual_digest,
    save_checkpoint,
)
from rescode.qjl.codebook_model import CodebookExpert, base_state_bytes, restore_base
from rescode.qjl.core import PackedResidual, ProjectionBank
from rescode.qjl.exponent_mantissa_base import reconstruct_base_from_weight

DEFAULT_TASKS = "arc_easy,arc_challenge,mmlu,openbookqa,rte"
ALL_TASKS = "winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte"
ERROR_FIELDS = ("d_factor", "layer", "expert", "projection", "samples", "base_rel_l2_A",
                "base_rel_l2_B", "total_rel_l2_A", "total_rel_l2_B")


def parse_factors(value: str) -> list[int]:
    values = [int(v.strip()) for v in value.split(",")]
    if not values or len(set(values)) != len(values) or any(v <= 0 for v in values):
        raise ValueError("D_FACTORS must be unique positive integers")
    return values


def resolve_model(path: Path | None) -> Path:
    if path:
        root = path.expanduser().resolve()
    else:
        candidates = sorted((Path("/workspace/.hf_home/hub") /
            "models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots").glob("*"))
        candidates += sorted((Path.home() / ".cache/huggingface/hub" /
            "models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots").glob("*"))
        root = next((p for p in candidates if (p / "model.safetensors.index.json").is_file()), None)
        if root is None:
            root = Path(snapshot_download("Qwen/Qwen1.5-MoE-A2.7B-Chat",
                allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json",
                                "tokenizer*", "special_tokens_map.json", "added_tokens.json",
                                "vocab*", "merges*"]))
    index_path = root / "model.safetensors.index.json"
    if not (root / "config.json").is_file() or not index_path.is_file():
        raise FileNotFoundError(f"complete Qwen model snapshot missing: {root}")
    index = json.loads(index_path.read_text())["weight_map"]
    missing = [shard for shard in set(index.values()) if not (root / shard).is_file()]
    if missing:
        raise FileNotFoundError(f"Qwen model shards missing from {root}: {missing[:3]}")
    return root


def rel_l2(actual: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - target.float())
    denominator = torch.linalg.vector_norm(target.float())
    return float(numerator / denominator) if denominator > 0 else (0.0 if numerator == 0 else math.inf)


def diagnostic_inputs(layer: int, expert: int, input_dim: int, samples: int,
                      device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1729 + 1009 * layer + expert)
    return torch.randn(samples, input_dim, generator=generator).to(device=device, dtype=torch.bfloat16)


def inspect_original_weight(handle, key: str, expected: tuple[int, int], device) -> torch.Tensor:
    weight = handle.get_tensor(key)
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != expected:
        raise ValueError(f"expected original BF16 {expected} for {key}, found {weight.dtype} {tuple(weight.shape)}")
    return weight.to(device)


@torch.no_grad()
def convert(args, model_dir: Path) -> dict:
    baseline = baseline_payload(args.baseline_ckpt)
    config_a = baseline["config"]
    if config_a["projection_seed"] == args.base_seed:
        raise ValueError("Base and residual seeds must be different")
    selected_layers = sorted(baseline["layers"])
    if args.layers != "all":
        requested = sorted({int(v) for v in args.layers.split(",")})
        if any(layer not in selected_layers for layer in requested):
            raise ValueError("requested layer not found in baseline checkpoint")
        selected_layers = requested
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Triton Base QJL compression")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    residual_hash = residual_digest(baseline)
    base_banks = {2048: ProjectionBank(2048 * args.d_factor, args.base_seed),
                  1408: ProjectionBank(1408 * args.d_factor, args.base_seed)}
    residual_banks = {2048: ProjectionBank(4096, config_a["projection_seed"]),
                      1408: ProjectionBank(2816, config_a["projection_seed"])}
    data, errors = {}, []
    weight_count = 0
    with ExitStack() as stack:
        handles = {}

        def weight(layer, expert, key):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{key}_proj.weight"
            shard = index[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(str(model_dir / shard),
                                                                framework="pt", device="cpu"))
            shape = (2048, 1408) if key == "down" else (1408, 2048)
            return inspect_original_weight(handles[shard], name, shape, device)

        for layer in selected_layers:
            layer_data = {}
            expert_ids = sorted(baseline["layers"][layer])
            if args.max_experts is not None:
                expert_ids = expert_ids[:args.max_experts]
            for expert_id in expert_ids:
                parts_b, base_modules, residual_modules = {}, {}, {}
                original = {}
                decoded_bases = {}
                x_hidden = diagnostic_inputs(layer, expert_id, 2048,
                                             args.diagnostic_samples, device)
                for key in PROJECTIONS:
                    part_a = baseline["layers"][layer][expert_id][key]
                    base_state = part_a["base"]
                    input_dim = 1408 if key == "down" else 2048
                    rows = 2048 if key == "down" else 1408
                    if tuple(base_state["shape"]) != (rows, input_dim):
                        raise ValueError(f"baseline Base shape mismatch: {layer}/{expert_id}/{key}")
                    w = weight(layer, expert_id, key)
                    a_base = restore_base(base_state, device, "exponent-mantissa")
                    b_float = a_base.decode()
                    if not torch.equal(b_float, reconstruct_base_from_weight(w, a_base.beta)):
                        raise AssertionError("stored Base does not match original BF16 and FP16 beta")
                    residual = w.float() - b_float
                    if not torch.allclose(b_float + residual, w.float(), rtol=1e-6, atol=1e-6):
                        raise AssertionError("W = B + R reconstruction failed")
                    r_state = {"packed": part_a["packed"], "norms": part_a["norms"]}
                    if not torch.allclose(torch.linalg.vector_norm(residual, dim=1).cpu(),
                                          r_state["norms"], rtol=1e-6, atol=1e-7):
                        raise AssertionError("baseline residual row norms do not match W - B")
                    r_bank = residual_banks[input_dim]
                    r_module = PackedResidual(r_state["packed"].to(device),
                                              r_state["norms"].to(device), input_dim,
                                              r_bank.d, r_bank, args.backend)
                    b_state, b_module = encode_qjl_base(base_state, base_banks[input_dim],
                                                        device, args.qjl_row_chunk)
                    b_module.backend = args.backend
                    parts_b[key] = {"base": b_state, "residual": r_state}
                    base_modules[key], residual_modules[key] = b_module, r_module
                    original[key], decoded_bases[key] = w, b_float
                    weight_count += w.numel()
                # The Down input is the original BF16 expert's SwiGLU activation.
                original_gate = F.linear(x_hidden, original["gate"])
                original_up = F.linear(x_hidden, original["up"])
                x_down = F.silu(original_gate) * original_up
                for key in PROJECTIONS:
                    x = x_down if key == "down" else x_hidden
                    b_ref = F.linear(x.float(), decoded_bases[key])
                    w_ref = F.linear(x.float(), original[key].float())
                    a_base = restore_base(baseline["layers"][layer][expert_id][key]["base"],
                                          device, "exponent-mantissa")
                    a_b = a_base(x)
                    b_b = base_modules[key](x)
                    r_y = residual_modules[key](x)
                    errors.append({"d_factor": args.d_factor, "layer": layer,
                                   "expert": expert_id, "projection": key,
                                   "samples": args.diagnostic_samples,
                                   "base_rel_l2_A": rel_l2(a_b, b_ref),
                                   "base_rel_l2_B": rel_l2(b_b, b_ref),
                                   "total_rel_l2_A": rel_l2(a_b + r_y, w_ref),
                                   "total_rel_l2_B": rel_l2(b_b + r_y, w_ref)})
                a_expert = CodebookExpert(
                    {key: restore_base(baseline["layers"][layer][expert_id][key]["base"],
                                       device, "exponent-mantissa") for key in PROJECTIONS},
                    residual_modules)
                b_expert = DualQJLExpert(base_modules, residual_modules)
                exact_expert = F.linear(x_down, original["down"])
                errors.append({"d_factor": args.d_factor, "layer": layer,
                               "expert": expert_id, "projection": "expert_output",
                               "samples": args.diagnostic_samples,
                               "base_rel_l2_A": None, "base_rel_l2_B": None,
                               "total_rel_l2_A": rel_l2(a_expert(x_hidden), exact_expert),
                               "total_rel_l2_B": rel_l2(b_expert(x_hidden), exact_expert)})
                layer_data[expert_id] = parts_b
                del original, decoded_bases, base_modules, residual_modules
            data[layer] = layer_data
            print(f"[base-qjl] D_B={args.d_factor}x layer={layer} experts={len(layer_data)}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    config_b = {"model": "Qwen/Qwen1.5-MoE-A2.7B-Chat", "model_path": str(model_dir),
                "baseline_checkpoint": str(args.baseline_ckpt),
                "baseline_residual_sha256": residual_hash,
                "base_method": "qjl", "residual_method": "qjl",
                "base_factor": args.d_factor, "residual_factor": 2,
                "base_seed": args.base_seed, "residual_seed": config_a["projection_seed"],
                "input_dims": {"gate": 2048, "up": 2048, "down": 1408},
                "backend": args.backend, "layers": selected_layers,
                "max_experts": args.max_experts, "weight_count": weight_count,
                "diagnostic_samples": args.diagnostic_samples}
    payload_b = {"format": FORMAT, "config": config_b, "layers": data}
    sizes = save_checkpoint(payload_b, args.checkpoint)
    # Verify that every residual tensor survives save/reload byte-for-byte.
    reloaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    for layer, experts in data.items():
        for expert_id, projections in experts.items():
            for key, part in projections.items():
                old = baseline["layers"][layer][expert_id][key]
                new = reloaded["layers"][layer][expert_id][key]["residual"]
                if not torch.equal(new["packed"], old["packed"]) or not torch.equal(new["norms"], old["norms"]):
                    raise AssertionError("residual packed bits or norms changed in saved B checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(errors)
    summary = {**config_b, **sizes,
               "base_bits_per_weight": 8 * sizes["base_packed_and_norm_bytes"] / weight_count,
               "residual_bits_per_weight": 8 * sizes["residual_packed_and_norm_bytes"] / weight_count,
               "checkpoint_bits_per_weight": 8 * sizes["checkpoint_bytes"] / weight_count,
               "bf16_saving_pct": 100 * (1 - sizes["checkpoint_bytes"] / (2 * weight_count)),
               "errors_csv": str(args.output_dir / "errors.csv")}
    (args.output_dir / "config.json").write_text(json.dumps(config_b, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_model(model_dir: Path, memory_fraction: float):
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * memory_fraction)
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True, max_memory=max_memory)
    return model.eval()


@torch.no_grad()
def evaluate(args, model_dir: Path) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_dir, args.gpu_memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    payload = load_checkpoint(model, args.checkpoint, args.backend)
    config = payload["config"]
    del payload
    metrics = {"checkpoint": str(args.checkpoint), "checkpoint_bytes": args.checkpoint.stat().st_size,
               "checkpoint_config": config, "tasks": args.tasks,
               "eval_limit": args.eval_limit, "backend": args.backend}
    if not args.skip_eval:
        result_path = args.output_dir / "lm_eval.txt"
        results = evaluate_fewshot(model, tokenizer, args.tasks, 0,
            eval_batch_size=args.eval_batch_size, output_path=str(result_path),
            limit=args.eval_limit or None)
        metrics["task_summary"] = report_eval_summary(
            results, args.tasks, str(result_path), header="Base QJL + Residual QJL evaluation")
        metrics["task_results"] = results.get("results", {})
    metrics.update(gpu_resident_breakdown(model))
    if torch.cuda.is_available():
        metrics["gpu_process_allocated_bytes"] = [torch.cuda.memory_allocated(i)
                                                   for i in range(torch.cuda.device_count())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    return metrics


def _run(command: list[str]) -> None:
    print("[base-qjl] running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=ROOT)


def sweep(args, model_dir: Path) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline_ckpt or (args.output_dir / "baseline" / "compact_qjl_state.pt")
    if not baseline.is_file():
        if args.baseline_ckpt:
            raise FileNotFoundError(f"BASELINE_CKPT missing: {baseline}")
        _run([sys.executable, "scripts/QJL/run_codebook_base.py", "--mode", "compress",
              "--base-method", "exponent-mantissa", "--model-path", str(model_dir),
              "--layers", args.layers, "--qjl-d-factor", "2",
              "--projection-seed", str(args.residual_seed), "--backend", args.backend,
              "--output-dir", str(args.output_dir / "baseline"), "--skip-eval"])
    old = baseline_payload(baseline)
    if old["config"]["projection_seed"] != args.residual_seed:
        raise ValueError("existing baseline residual seed differs from requested seed")
    baseline_layers = sorted(old["layers"])
    if args.layers == "all":
        if baseline_layers != list(range(24)):
            raise ValueError("full sweep requires a baseline with all 24 layers")
    elif baseline_layers != sorted(int(v) for v in args.layers.split(",")):
        raise ValueError("baseline layers differ from requested layers")
    full_weight_count = 60 * 3 * 2048 * 1408 * len(baseline_layers)
    baseline_hash = residual_digest(old)
    if args.max_experts is not None and not args.skip_eval:
        raise ValueError("partial-expert sweep cannot run full-model evaluation")
    args.baseline_ckpt = baseline
    baseline_metrics_path = args.output_dir / "baseline" / "eval_all" / "metrics.json"
    baseline_metrics = (json.loads(baseline_metrics_path.read_text())
                        if baseline_metrics_path.is_file() else {})
    baseline_tasks = [t[0] for t in baseline_metrics.get("task_summary", {}).get("tasks", [])]
    if not args.skip_eval and baseline_tasks != args.tasks.split(","):
        _run([sys.executable, "scripts/QJL/run_codebook_base.py", "--mode", "eval",
              "--model-path", str(model_dir), "--checkpoint", str(baseline),
              "--output-dir", str(baseline_metrics_path.parent), "--backend", args.backend,
              "--task", args.tasks, "--eval-limit", str(args.eval_limit),
              "--eval-batch-size", str(args.eval_batch_size)])
    factors = parse_factors(args.d_factors)
    for factor in factors:
        folder = args.output_dir / f"base_d{factor}"
        checkpoint = folder / "compact_qjl_state.pt"
        if not checkpoint.is_file():
            command = [sys.executable, "scripts/QJL/run_base_qjl_sweep.py", "--mode", "convert",
                "--model-path", str(model_dir), "--baseline-ckpt", str(baseline),
                "--output-dir", str(folder), "--checkpoint", str(checkpoint),
                "--d-factor", str(factor), "--base-seed", str(args.base_seed),
                "--layers", args.layers, "--backend", args.backend,
                "--diagnostic-samples", str(args.diagnostic_samples),
                "--qjl-row-chunk", str(args.qjl_row_chunk)]
            if args.max_experts is not None:
                command += ["--max-experts", str(args.max_experts)]
            _run(command)
        elif not (folder / "errors.csv").is_file():
            raise FileNotFoundError(f"checkpoint exists without diagnostics: {folder}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = saved.get("config", {})
        if (saved.get("format") != FORMAT or cfg.get("base_factor") != factor
                or cfg.get("base_seed") != args.base_seed
                or cfg.get("residual_seed") != args.residual_seed
                or cfg.get("baseline_residual_sha256") != baseline_hash):
            raise ValueError(f"existing B checkpoint is incompatible: {checkpoint}")
        for layer, experts in saved["layers"].items():
            for expert_id, projections in experts.items():
                for key, part in projections.items():
                    original = old["layers"][layer][expert_id][key]
                    residual = part["residual"]
                    if (not torch.equal(residual["packed"], original["packed"])
                            or not torch.equal(residual["norms"], original["norms"])):
                        raise AssertionError(f"residual QJL differs from A at D_B={factor}, "
                                             f"layer={layer}, expert={expert_id}, projection={key}")
        del saved
        eval_path = folder / "eval_all" / "eval_metrics.json"
        eval_metrics = json.loads(eval_path.read_text()) if eval_path.is_file() else {}
        if (not args.skip_eval and
                (eval_metrics.get("tasks") != args.tasks or
                 eval_metrics.get("eval_limit") != args.eval_limit or
                 "task_summary" not in eval_metrics)):
            _run([sys.executable, "scripts/QJL/run_base_qjl_sweep.py", "--mode", "eval",
                  "--model-path", str(model_dir), "--checkpoint", str(checkpoint),
                  "--output-dir", str(eval_path.parent), "--backend", args.backend,
                  "--tasks", args.tasks, "--eval-limit", str(args.eval_limit),
                  "--eval-batch-size", str(args.eval_batch_size)])
    # Rebuild CSVs from completed factor directories; reruns are idempotent.
    all_errors = []
    for factor in factors:
        with (args.output_dir / f"base_d{factor}" / "errors.csv").open() as f:
            all_errors.extend(csv.DictReader(f))
    with (args.output_dir / "errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ERROR_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(all_errors)
    del old
    if args.skip_eval:
        fields = ("method", "d_factor", "checkpoint_bytes", "bits_per_weight",
                  "saving_vs_bf16_pct", "avg_accuracy", "delta_avg_vs_A")
        size_rows = [{"method": "A", "d_factor": 0,
                      "checkpoint_bytes": baseline.stat().st_size,
                      "bits_per_weight": 8 * baseline.stat().st_size / full_weight_count,
                      "saving_vs_bf16_pct": 100 * (1 - baseline.stat().st_size / (2 * full_weight_count)),
                      "avg_accuracy": None, "delta_avg_vs_A": None}]
        for factor in factors:
            folder = args.output_dir / f"base_d{factor}"
            summary = json.loads((folder / "metrics.json").read_text())
            size_rows.append({"method": "B", "d_factor": factor,
                "checkpoint_bytes": summary["checkpoint_bytes"],
                "bits_per_weight": summary["checkpoint_bits_per_weight"],
                "saving_vs_bf16_pct": summary["bf16_saving_pct"],
                "avg_accuracy": None, "delta_avg_vs_A": None})
        with (args.output_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            writer.writeheader(); writer.writerows(size_rows)
        return
    baseline_metrics = json.loads(baseline_metrics_path.read_text())
    a_summary = baseline_metrics.get("task_summary", {})
    if [t[0] for t in a_summary.get("tasks", [])] != args.tasks.split(","):
        raise ValueError("existing A task metrics do not match requested TASKS")
    rows = []
    baseline_summary_path = args.output_dir / "baseline" / "summary.json"
    baseline_summary = json.loads(baseline_summary_path.read_text()) if baseline_summary_path.is_file() else {}
    if not baseline_summary:
        old = baseline_payload(baseline)
        base_bytes = sum(base_state_bytes(part["base"])
                         for experts in old["layers"].values() for parts in experts.values()
                         for part in parts.values())
        residual_bytes = sum(part["packed"].numel() + part["norms"].numel() * 4
                             for experts in old["layers"].values() for parts in experts.values()
                             for part in parts.values())
        baseline_summary = {"base_tensor_bytes": base_bytes, "qjl_tensor_bytes": residual_bytes,
                            "metadata_bytes": baseline.stat().st_size - base_bytes - residual_bytes}
        del old
    def make_row(name, factor, checkpoint_path, metrics, summary, weight_count):
        task_values = {f"acc_{t[0]}": t[1] for t in metrics["task_summary"]["tasks"]}
        task_values.update({f"acc_norm_{t[0]}": t[2]
                            for t in metrics["task_summary"]["tasks"]})
        return {"method": name, "d_factor": factor,
                "d_gate_up": 2048 * factor if factor else None,
                "d_down": 1408 * factor if factor else None,
                **task_values, "avg_accuracy": metrics["task_summary"]["avg"],
                "avg_norm_accuracy": metrics["task_summary"]["avg_norm"],
                "delta_avg_vs_A": (metrics["task_summary"]["avg"] - a_summary["avg"]
                                   if name != "A" else 0.0),
                "delta_avg_norm_vs_A": (metrics["task_summary"]["avg_norm"] - a_summary["avg_norm"]
                                        if name != "A" else 0.0),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "base_stored_bytes": summary.get("base_packed_and_norm_bytes", summary.get("base_tensor_bytes")),
                "residual_stored_bytes": summary.get("residual_packed_and_norm_bytes", summary.get("qjl_tensor_bytes")),
                "base_packed_bytes": summary.get("base_packed_bytes"),
                "base_norm_bytes": summary.get("base_norm_bytes"),
                "residual_packed_bytes": summary.get("residual_packed_bytes"),
                "residual_norm_bytes": summary.get("residual_norm_bytes"),
                "metadata_bytes": summary.get("metadata_bytes"),
                "bits_per_weight": 8 * checkpoint_path.stat().st_size / weight_count,
                "saving_vs_bf16_pct": 100 * (1 - checkpoint_path.stat().st_size / (2 * weight_count)),
                "gpu_resident_compact_bytes": metrics.get("gpu_resident_compact_total_bytes"),
                "base_projection_gpu_bytes": metrics.get("base_projection_gpu_bytes"),
                "residual_projection_gpu_bytes": metrics.get("residual_projection_gpu_bytes")}
    rows.append(make_row("A", 0, baseline, baseline_metrics, baseline_summary, full_weight_count))
    for factor in factors:
        folder = args.output_dir / f"base_d{factor}"
        checkpoint = folder / "compact_qjl_state.pt"
        metrics = json.loads((folder / "eval_all" / "eval_metrics.json").read_text())
        summary = json.loads((folder / "metrics.json").read_text())
        rows.append(make_row("B", factor, checkpoint, metrics, summary, summary["weight_count"]))
    fields = list(rows[0])
    with (args.output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("method D_B avg_acc delta_vs_A bits/weight checkpoint_GiB")
    for r in rows:
        print(f"{r['method']:>6} {r['d_factor']:>3} {r['avg_accuracy']:.4f} "
              f"{r['delta_avg_vs_A']:+.4f} {r['bits_per_weight']:.3f} "
              f"{r['checkpoint_bytes']/2**30:.2f}")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sweep", "convert", "eval"), default="sweep")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--baseline-ckpt", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/base_qjl_sweep")
    p.add_argument("--d-factors", default="2,4,8,16")
    p.add_argument("--d-factor", type=int, default=2)
    p.add_argument("--layers", default="all")
    p.add_argument("--max-experts", type=int)
    p.add_argument("--base-seed", type=int, default=1)
    p.add_argument("--residual-seed", type=int, default=0)
    p.add_argument("--backend", choices=("triton", "reference"), default="triton")
    p.add_argument("--qjl-row-chunk", type=int, default=128)
    p.add_argument("--diagnostic-samples", type=int, default=4)
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    if args.base_seed == args.residual_seed or args.d_factor < 1:
        p.error("Base seed must differ from residual seed and D factor must be positive")
    if args.max_experts is not None and not 1 <= args.max_experts <= 60:
        p.error("max experts must be within 1..60")
    if args.mode == "convert" and (args.baseline_ckpt is None or args.checkpoint is None):
        p.error("convert requires --baseline-ckpt and --checkpoint")
    if args.mode == "eval" and args.checkpoint is None:
        p.error("eval requires --checkpoint")
    return args


def main():
    args = arguments()
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model_dir = resolve_model(args.model_path)
    try:
        if args.mode == "sweep":
            sweep(args, model_dir)
        elif args.mode == "convert":
            convert(args, model_dir)
        else:
            evaluate(args, model_dir)
    except torch.OutOfMemoryError as exc:
        free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        raise RuntimeError(f"CUDA OOM at D_B factor {args.d_factor}; free={free} total={total} bytes") from exc


if __name__ == "__main__":
    main()
