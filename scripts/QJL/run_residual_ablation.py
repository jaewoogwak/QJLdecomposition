#!/usr/bin/env python3
"""Independent A–E residual ablation with one fixed exponent/mantissa Base."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rescode.evaluation import evaluate_fewshot, get_calib_dataloder
from rescode.merging.eval_summary import report_eval_summary
from rescode.merging.expert_clusters import form_clusters, residual_params_fn_for
from rescode.qjl.calibration import collect_projection_samples
from rescode.qjl.codebook_model import base_state_bytes
from rescode.qjl.core import ProjectionBank, pack_residual
from rescode.qjl.exponent_mantissa_base import (
    PackedExponentMantissaBase, encode_exponent_mantissa_base,
    reconstruct_base_from_weight,
)
from rescode.qjl.residual_ablation_methods import (
    fit_shared_lowrank, gptq_int2, lowrank_group_bytes, matched_rank,
)
from rescode.qjl.residual_ablation_model import (
    FORMAT, PROJECTIONS, build_expert, load_variant, tensor_bytes,
)

MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
DEFAULT_TASKS = "arc_easy,arc_challenge,mmlu,openbookqa,rte"
FIELDS = ("case", "factor", "layer", "expert", "projection", "group_id", "group_size",
          "rank", "samples", "residual_norm_ratio", "weight_reconstruction_error",
          "residual_output_error", "residual_zero_reference", "weight_output_error",
          "expert_output_error", "base_sha256", "source_sha256")


def parse_list(value: str, choices: set[str] | None = None) -> list:
    values = [v.strip() for v in value.split(",") if v.strip()]
    if not values or len(set(values)) != len(values) or (choices and set(values) - choices):
        raise ValueError(f"invalid list: {value}")
    return values


def variants(cases: list[str], factors: list[int]) -> list[tuple[str, int]]:
    return [(case, factor) for case in cases
            for factor in (factors if case in ("C", "D") else [0])]


def tag(case: str, factor: int) -> str:
    return f"{case}_d{factor}" if case in ("C", "D") else case


def digest(data) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def atomic_torch(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    with temporary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ratio(actual: torch.Tensor, target: torch.Tensor) -> tuple[float | None, bool]:
    a, b = actual.float(), target.float()
    denominator = torch.linalg.vector_norm(b)
    if denominator == 0:
        return (0.0 if torch.linalg.vector_norm(a) == 0 else None), True
    return float(torch.linalg.vector_norm(a - b) / denominator), False


def sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(fn, x, warmup: int, repeats: int) -> tuple[float, float, float | None]:
    for _ in range(warmup):
        fn(x)
    sync(x.device)
    times, events = [], []
    for _ in range(repeats):
        if x.is_cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        start = time.perf_counter()
        fn(x)
        if x.is_cuda:
            end.record()
        sync(x.device)
        times.append((time.perf_counter() - start) * 1000)
        if x.is_cuda:
            events.append(begin.elapsed_time(end))
    milliseconds = statistics.median(times)
    return milliseconds, 1000 * x.shape[0] / milliseconds, (
        statistics.median(events) if events else None)


def load_samples(args, model_dir: Path, layers: list[int]) -> tuple[dict, dict, str]:
    """Capture calibration and held-out routed inputs from distinct corpora."""
    calibration = Path(args.gptq_calibration_path)
    heldout = Path(args.diagnostic_path)
    need_train = "E" in parse_list(args.cases, set("ABCDE"))
    if need_train and calibration.resolve() == heldout.resolve():
        raise ValueError("GPTQ calibration and diagnostic corpora must be different files")
    for path in ((calibration, heldout) if need_train else (heldout,)):
        name = path.name.removesuffix("-calib.jsonl")
        default = ROOT / "rescode/data" / f"{name}-calib.jsonl"
        is_default = name in ("dclm", "wikipedia") and path.resolve() == default.resolve()
        wanted = max(1000, args.n_sentences * 17)
        count = sum(1 for _ in path.open()) if path.is_file() and is_default else None
        if not path.is_file() or (is_default and count < wanted):
            if not is_default:
                raise FileNotFoundError(f"real activation corpus missing: {path}")
            command = [sys.executable,
                       str(ROOT / "scripts/RESCODE/prepare_calib_datasets.py"),
                       "--dataset", name, "--n_docs", str(wanted)]
            if path.is_file():
                command.append("--overwrite")
            subprocess.run(command, check=True, cwd=ROOT)
    key = {"model": str(model_dir.resolve()), "layers": layers,
           "calibration": [str(calibration.resolve()), calibration.stat().st_size,
                           calibration.stat().st_mtime_ns] if need_train else None,
           "diagnostic": [str(heldout.resolve()), heldout.stat().st_size,
                          heldout.stat().st_mtime_ns],
           "n_sentences": args.n_sentences, "calib_seed": args.calib_seed,
           "train_per_expert": args.gptq_calibration_samples,
           "heldout_per_expert": args.diagnostic_samples}
    sample_hash = digest(key)
    cache = Path(args.sample_cache)
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("key") != key:
            raise ValueError(f"activation cache has different inputs/settings: {cache}")
        atomic_json(args.output_dir / "activation_metadata.json",
                    {"sample_hash": sample_hash, "cache": str(cache),
                     "collection_seconds": payload.get("collection_seconds"),
                     "calibration_seconds": payload.get("calibration_seconds"),
                     "diagnostic_collection_seconds": payload.get("diagnostic_collection_seconds"),
                     "reused": True, "calibration_dataset": str(calibration) if need_train else None,
                     "diagnostic_dataset": str(heldout)})
        return payload["train"], payload["heldout"], sample_hash
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True).eval()
    def collect(path: Path, seed: int, train_cap: int, heldout_cap: int):
        loader = get_calib_dataloder(
            dataset="c4", tokenizer=tokenizer, max_block_size=args.calib_seqlen,
            n_blocks_for_stat=args.n_sentences, batch_size=1, num_workers=2,
            seed=seed, data_path=str(path))
        return collect_projection_samples(model, loader, "qwen", layers,
                                          train_cap, heldout_cap)
    started = time.perf_counter()
    train = {}
    if need_train:
        train, _, _ = collect(calibration, args.calib_seed,
                              args.gptq_calibration_samples, 0)
    calibration_seconds = time.perf_counter() - started if need_train else 0.0
    diagnostic_start = time.perf_counter()
    _, test, _ = collect(heldout, args.calib_seed + 1,
                          0, args.diagnostic_samples)
    payload = {"key": key, "train": train, "heldout": test,
               "collection_seconds": time.perf_counter() - started,
               "calibration_seconds": calibration_seconds,
               "diagnostic_collection_seconds": time.perf_counter() - diagnostic_start}
    atomic_torch(cache, payload)
    atomic_json(args.output_dir / "activation_metadata.json",
                {"sample_hash": sample_hash, "cache": str(cache),
                 "collection_seconds": payload["collection_seconds"],
                 "calibration_seconds": calibration_seconds,
                 "diagnostic_collection_seconds": payload["diagnostic_collection_seconds"],
                 "reused": False, "calibration_dataset": str(calibration) if need_train else None,
                 "diagnostic_dataset": str(heldout)})
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return train, test, sample_hash


def grouping(args, model_dir: Path) -> tuple[dict, dict, str]:
    if "D" not in parse_list(args.cases, set("ABCDE")):
        return {}, {}, "none"
    path = Path(args.grouping_cache)
    if not path.is_file():
        raise FileNotFoundError(
            f"ResCode grouping statistics missing: {path}. "
            "Run scripts/QJL/run_residual_norm_ablation.sh to build the cache first.")
    stats = torch.load(path, map_location="cpu", weights_only=False)
    if set(stats) != set(range(24)):
        raise ValueError("ResCode grouping cache must cover all 24 layers")
    for layer, item in stats.items():
        if item["saliency"].shape != (60,) or item["dist"].shape != (60, 60):
            raise ValueError(f"invalid grouping cache at layer {layer}")
    config = json.loads((model_dir / "config.json").read_text())
    d_model, d_ff = int(config["hidden_size"]), int(config["moe_intermediate_size"])
    groups, _ = form_clusters(
        stats, num_groups=args.num_groups, singleton_ratio=args.singleton_ratio,
        expert_params=3 * d_model * d_ff,
        residual_params_fn=residual_params_fn_for(d_model, d_ff, args.grouping_rank, 3),
        linkage="average", budget_aware=True)
    for layer, partition in groups.items():
        if sorted(e for members in partition for e in members) != list(range(60)):
            raise AssertionError(f"grouping does not partition layer {layer}")
    # Include the cache bytes and grouping parameters in the resume key.
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return groups, stats, digest([h, args.num_groups, args.singleton_ratio,
                                  args.grouping_rank, "average", True])


def source_layer(model_dir: Path, index: dict, layer: int, device) -> tuple[dict, str]:
    sources, h = {}, hashlib.sha256()
    with ExitStack() as stack:
        handles = {}
        for expert in range(60):
            parts = {}
            for key in PROJECTIONS:
                name = f"model.layers.{layer}.mlp.experts.{expert}.{key}_proj.weight"
                shard = index[name]
                if shard not in handles:
                    handles[shard] = stack.enter_context(safe_open(
                        str(model_dir / shard), framework="pt", device="cpu"))
                weight = handles[shard].get_tensor(name)
                expected = (2048, 1408) if key == "down" else (1408, 2048)
                if weight.dtype != torch.bfloat16 or tuple(weight.shape) != expected:
                    raise ValueError(f"invalid original BF16 weight: {name}")
                h.update(name.encode())
                h.update(weight.contiguous().view(torch.uint8).numpy().tobytes())
                parts[key] = weight.to(device)
            sources[expert] = parts
    return sources, h.hexdigest()


def base_and_residual(sources: dict) -> tuple[dict, dict, str]:
    bases, residuals, h = {}, {}, hashlib.sha256()
    for expert, parts in sources.items():
        bases[expert], residuals[expert] = {}, {}
        for key in PROJECTIONS:
            weight = parts[key]
            state = encode_exponent_mantissa_base(weight)
            decoded = reconstruct_base_from_weight(weight, state["beta"].to(weight.device))
            residual = weight.float() - decoded
            if not torch.allclose(decoded + residual, weight.float(), atol=1e-6, rtol=1e-6):
                raise AssertionError("Case B reconstruction failed")
            # Encode once; all variant payloads refer to this exact state.
            bases[expert][key], residuals[expert][key] = state, residual
            h.update(str(expert).encode() + key.encode())
            for field in ("beta", "codes", "exp_data", "exp_offsets", "exp_modes",
                          "special_indices", "special_bits"):
                value = state[field]
                h.update(value.numpy().tobytes() if isinstance(value, torch.Tensor) else value)
    return bases, residuals, h.hexdigest()


def check_samples(samples: dict, layer: int, expert: int, key: str, cap: int) -> torch.Tensor:
    item = samples.get(layer, {}).get(key, {}).get(expert)
    if item is None or item[0].shape[0] < cap:
        raise ValueError(f"insufficient real routed inputs: {layer}/{expert}/{key}")
    return item[0][:cap]


def variant_config(args, model_dir: Path, layers: list[int], case: str, factor: int,
                   sample_hash: str, group_hash: str) -> dict:
    relevant = {"format": FORMAT, "case": case, "factor": factor,
                "model": str(model_dir.resolve()), "layers": layers,
                "max_experts": args.max_experts, "qjl_seed": args.qjl_seed,
                "backend": args.backend, "sample_hash": sample_hash,
                "fp32_residual_device": args.fp32_residual_device if case == "B" else None,
                "group_hash": group_hash if case == "D" else None,
                "gptq_damp": args.gptq_damp if case == "E" else None,
                "gptq_block_size": args.gptq_block_size if case == "E" else None,
                "lowrank_alpha": args.lowrank_alpha if case == "D" else None,
                "storage": {"base": "packed-exponent-sign+FP16-beta",
                            "B": "FP32", "C": "packed-sign+FP32-norm",
                            "D": "shared-BF16-factors+BF16-codes",
                            "E": "packed-INT2+FP16-scale+FP16-offset"}}
    relevant["digest"] = digest(relevant)
    return relevant


def make_groups(args, layer: int, factor: int, residuals: dict,
                groups: dict, stats: dict) -> tuple[dict, dict]:
    output, membership = {}, {}
    for group_id, members in enumerate(groups[layer]):
        if args.max_experts is not None and not any(e < args.max_experts for e in members):
            continue
        output[group_id] = {}
        for local, expert in enumerate(members):
            membership[expert] = (group_id, local, len(members))
        for key in PROJECTIONS:
            sample = residuals[members[0]][key]
            m, n = sample.shape
            rank, budget = matched_rank(len(members), m, n, factor * n)
            selected = torch.stack([residuals[e][key] for e in members])
            frequencies = stats[layer]["freq"][members].to(selected.device).float()
            if frequencies.sum() <= 0:
                frequencies = torch.ones_like(frequencies)
            left, right, codes = fit_shared_lowrank(
                selected, rank, args.qjl_seed, freqs=frequencies,
                alpha=args.lowrank_alpha)
            actual = left.numel() * 2 + right.numel() * 2 + codes.numel() * 2
            if actual != lowrank_group_bytes(len(members), m, n, rank) or actual > budget:
                raise AssertionError("low-rank stored tensors exceed matched QJL budget")
            output[group_id][key] = {"a": left.cpu(), "b": right.cpu(),
                                     "codes": codes.cpu(), "members": members,
                                     "alpha": args.lowrank_alpha, "rank": rank,
                                     "qjl_budget_bytes": budget,
                                     "actual_bytes": actual,
                                     "budget_utilization": actual / budget}
    return output, membership


def diagnostics(case: str, factor: int, layer: int, expert: int, parts: dict,
                groups: dict, sources: dict, residuals: dict, heldout: dict,
                config: dict, args, source_hash: str, base_hash: str) -> tuple[list[dict], list[dict]]:
    device = sources[expert]["gate"].device
    residual_device = device
    if case == "B":
        if args.fp32_residual_device == "cpu":
            residual_device = torch.device("cpu")
        elif args.fp32_residual_device == "auto" and device.type == "cuda":
            needed = len(config["layers"]) * (args.max_experts or 60) * 3 * 1408 * 2048 * 4
            if needed + 2 * 2**30 >= torch.cuda.get_device_properties(device).total_memory:
                residual_device = torch.device("cpu")
    runtime = build_expert(parts, groups, case, factor, args.qjl_seed, device,
                           args.backend, residual_device, {})
    rows, latency = [], []
    for key in PROJECTIONS:
        x = check_samples(heldout, layer, expert, key,
                          args.diagnostic_samples).to(device)
        weight = sources[expert][key].float()
        r = residuals[expert][key]
        base = reconstruct_base_from_weight(sources[expert][key],
                                            parts[key]["base"]["beta"].to(device))
        residual_target = F.linear(x.float(), r)
        if case == "A":
            residual_actual = torch.zeros_like(residual_target)
        elif case == "B":
            residual_actual = F.linear(x.float(), parts[key]["residual"].to(device))
        elif case == "C":
            module = runtime.residuals[key]
            projection = module.bank.get(module.input_dim, device)
            projected = x.to(projection.dtype) @ projection
            residual_actual = module.forward_projected(projected, torch.float32)
        else:
            residual_actual = F.linear(x.float(), runtime.residuals[key].decode())
        full_actual = F.linear(x.float(), base) + residual_actual
        er, zero = ratio(residual_actual, residual_target)
        ew, _ = ratio(full_actual, F.linear(x.float(), weight))
        if case == "A":
            reconstructed = base
        elif case == "B":
            reconstructed = base + parts[key]["residual"].to(device)
        elif case == "D":
            state = groups[parts[key]["residual"]["group_id"]][key]
            reconstructed = base + args.lowrank_alpha * (
                state["a"].to(device).float() *
                state["codes"][parts[key]["residual"]["code_index"]].to(device).float()[None, :]) @ state["b"].to(device).float()
        elif case == "E":
            reconstructed = base + runtime.residuals[key].decode()
        else:
            reconstructed = None
        wr = ratio(reconstructed, weight)[0] if reconstructed is not None else None
        rr = ratio(r, weight)[0]
        group_id = parts[key].get("residual", {}).get("group_id") if case == "D" else None
        group_size = len(groups[group_id][key]["members"]) if group_id is not None else None
        rank = groups[group_id][key]["rank"] if group_id is not None else None
        rows.append({"case": case, "factor": factor, "layer": layer,
                     "expert": expert, "projection": key, "group_id": group_id,
                     "group_size": group_size, "rank": rank, "samples": x.shape[0],
                     "residual_norm_ratio": rr, "weight_reconstruction_error": wr,
                     "residual_output_error": er, "residual_zero_reference": zero,
                     "weight_output_error": ew, "expert_output_error": None,
                     "base_sha256": base_hash, "source_sha256": source_hash})
        if expert < args.latency_experts:
            sample = x[:args.latency_batch_size]
            ms, throughput, event_ms = timed_forward(
                lambda y: runtime.projection(key, y), sample,
                args.latency_warmup, args.latency_repeats)
            latency.append({"case": case, "factor": factor, "layer": layer,
                            "expert": expert, "projection": key,
                            "stage": "full_projection",
                            "backend": "dequantize+GEMM" if case == "E" else args.backend if case == "C" else "dense/factorized",
                            "batch_size": sample.shape[0], "latency_ms": ms,
                            "cuda_event_ms": event_ms,
                            "tokens_per_second": throughput,
                            "theoretical_ops_per_token": theoretical_ops(case, factor, key, rank),
                            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                            "residual_device": str(residual_device)})
            if case == "E":
                correction = runtime.residuals[key]
                decoded = correction.decode()
                for stage, function in (("residual_decode", lambda y: correction.decode()),
                                        ("residual_fp32_gemm", lambda y: F.linear(y.float(), decoded))):
                    stage_ms, stage_tps, stage_event = timed_forward(
                        function, sample, args.latency_warmup, args.latency_repeats)
                    latency.append({"case": case, "factor": factor, "layer": layer,
                                    "expert": expert, "projection": key,
                                    "stage": stage, "backend": "dequantize+GEMM",
                                    "batch_size": sample.shape[0], "latency_ms": stage_ms,
                                    "cuda_event_ms": stage_event,
                                    "tokens_per_second": stage_tps,
                                    "theoretical_ops_per_token": None,
                                    "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                                    "residual_device": str(residual_device)})
    hidden = check_samples(heldout, layer, expert, "gate", args.diagnostic_samples).to(device)
    wg, wu, wd = (sources[expert][key] for key in PROJECTIONS)
    original_middle = F.silu(F.linear(hidden, wg)) * F.linear(hidden, wu)
    original_output = F.linear(original_middle, wd)
    expert_error, _ = ratio(runtime(hidden), original_output)
    rows.append({"case": case, "factor": factor, "layer": layer,
                 "expert": expert, "projection": "expert_output",
                 "samples": hidden.shape[0], "expert_output_error": expert_error,
                 "base_sha256": base_hash, "source_sha256": source_hash})
    return rows, latency


def theoretical_ops(case: str, factor: int, key: str, rank: int | None = None) -> int:
    m, n = ((2048, 1408) if key == "down" else (1408, 2048))
    base = 2 * m * n  # packed exponent Base dequantization/GEMM; explanatory only
    if case == "A":
        return base
    if case in ("B", "E"):
        return base + 2 * m * n
    if case == "C":
        return base + 2 * n * (factor * n) + 2 * m * (factor * n)
    return base + 2 * (m + n) * int(rank or 0) + int(rank or 0)


def compress(args, model_dir: Path, layers: list[int], selected: list[tuple[str, int]]) -> None:
    train, heldout, sample_hash = load_samples(args, model_dir, layers)
    groups, stats, group_hash = grouping(args, model_dir)
    configs = {(case, factor): variant_config(args, model_dir, layers, case, factor,
                                             sample_hash, group_hash)
               for case, factor in selected}
    for (case, factor), config in configs.items():
        folder = args.output_dir / tag(case, factor)
        path = folder / "config.json"
        if path.is_file() and json.loads(path.read_text()) != config:
            raise ValueError(f"existing variant config differs: {path}")
        atomic_json(path, config)
    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    for layer in layers:
        sources, source_hash = source_layer(model_dir, index, layer, device)
        pending = []
        for case, factor in selected:
            folder = args.output_dir / tag(case, factor)
            meta_path = folder / f"layer_{layer}.meta.json"
            checkpoint = folder / f"layer_{layer}.pt"
            metric_path = folder / f"layer_{layer}.errors.csv"
            if checkpoint.is_file() and meta_path.is_file() and metric_path.is_file():
                meta = json.loads(meta_path.read_text())
                if (meta.get("config_digest") != configs[(case, factor)]["digest"] or
                        meta.get("source_sha256") != source_hash or
                        meta.get("checkpoint_bytes") != checkpoint.stat().st_size):
                    raise ValueError(f"resume source/config mismatch: {checkpoint}")
                print(f"[ablation] reuse {tag(case, factor)} layer {layer}", flush=True)
            else:
                pending.append((case, factor))
        if not pending:
            continue
        bases, residuals, base_hash = base_and_residual(sources)
        lowrank = {}
        membership = {}
        lowrank_initialization_seconds = {}
        lowrank_initialization_peak_bytes = {}
        for case, factor in pending:
            if case == "D":
                sync(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                init_start = time.perf_counter()
                lowrank[factor], membership[factor] = make_groups(
                    args, layer, factor, residuals, groups, stats)
                sync(device)
                lowrank_initialization_seconds[factor] = time.perf_counter() - init_start
                lowrank_initialization_peak_bytes[factor] = (
                    torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        banks = {factor: {2048: ProjectionBank(2048 * factor, args.qjl_seed),
                          1408: ProjectionBank(1408 * factor, args.qjl_seed)}
                 for case, factor in pending if case == "C"}
        for case, factor in pending:
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            folder = args.output_dir / tag(case, factor)
            expert_states, metric_rows, latency_rows = {}, [], []
            group_state = lowrank.get(factor, {}) if case == "D" else {}
            count = args.max_experts or 60
            for expert in range(count):
                expert_states[expert] = {}
                for key in PROJECTIONS:
                    r = residuals[expert][key]
                    if case == "A":
                        correction = None
                    elif case == "B":
                        correction = r.cpu().clone()
                    elif case == "C":
                        packed, norms = pack_residual(r, banks[factor][r.shape[1]],
                                                      row_chunk=args.qjl_row_chunk)
                        correction = {"packed": packed.cpu(), "norms": norms.cpu(),
                                      "input_dim": r.shape[1], "d": factor * r.shape[1]}
                        if not torch.equal(norms, torch.linalg.vector_norm(r, dim=1)):
                            raise AssertionError("QJL FP32 row norms changed")
                    elif case == "D":
                        group_id, local, _ = membership[factor][expert]
                        correction = {"group_id": group_id, "code_index": local}
                    else:
                        x = check_samples(train, layer, expert, key,
                                          args.gptq_calibration_samples).to(device)
                        correction = gptq_int2(r, x, args.gptq_damp,
                                               args.gptq_block_size)
                    expert_states[expert][key] = {"base": bases[expert][key],
                                                  "residual": correction}
                rows, timing = diagnostics(case, factor, layer, expert,
                                           expert_states[expert], group_state, sources,
                                           residuals, heldout, configs[(case, factor)],
                                           args, source_hash, base_hash)
                metric_rows.extend(rows)
                latency_rows.extend(timing)
            payload = {"format": FORMAT, "case": case, "factor": factor,
                       "layer": layer, "config_digest": configs[(case, factor)]["digest"],
                       "source_sha256": source_hash, "base_sha256": base_hash,
                       "experts": expert_states, "groups": group_state}
            checkpoint = folder / f"layer_{layer}.pt"
            atomic_torch(checkpoint, payload)
            # A tiny reload check detects a truncated or incompatible save.
            reloaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if (reloaded["source_sha256"] != source_hash or
                    reloaded["base_sha256"] != base_hash):
                raise AssertionError("checkpoint reload mismatch")
            amounts = tensor_bytes(payload)
            sync(device)
            encode_and_diagnostics_seconds = time.perf_counter() - start
            initialization_seconds = (lowrank_initialization_seconds.get(factor, 0.0)
                                      if case == "D" else 0.0)
            meta = {"layer": layer, "case": case, "factor": factor,
                    "config_digest": configs[(case, factor)]["digest"],
                    "source_sha256": source_hash, "base_sha256": base_hash,
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "initialization_seconds": initialization_seconds,
                    "encode_and_diagnostics_seconds": encode_and_diagnostics_seconds,
                    "compression_seconds": initialization_seconds + encode_and_diagnostics_seconds,
                    **amounts}
            meta["peak_gpu_allocated_bytes"] = (
                max(torch.cuda.max_memory_allocated(device),
                    lowrank_initialization_peak_bytes.get(factor) or 0)
                if device.type == "cuda" else None)
            meta["metadata_bytes"] = meta["checkpoint_bytes"] - sum(
                amounts[k] for k in ("base_bytes", "residual_bytes"))
            atomic_json(folder / f"layer_{layer}.meta.json", meta)
            write_csv(folder / f"layer_{layer}.errors.csv", metric_rows, FIELDS)
            write_csv(folder / f"layer_{layer}.latency.csv", latency_rows)
            if case == "D":
                budget_rows = []
                for group_id, projections in group_state.items():
                    for key, state in projections.items():
                        budget_rows.append({"case": case, "factor": factor,
                                            "layer": layer, "group_id": group_id,
                                            "projection": key,
                                            "group_size": len(state["members"]),
                                            "rank": state["rank"],
                                            "stored_tensor_bytes": state["actual_bytes"],
                                            "qjl_budget_bytes": state["qjl_budget_bytes"],
                                            "budget_utilization": state["budget_utilization"]})
                write_csv(folder / f"layer_{layer}.budgets.csv", budget_rows)
            print(f"[ablation] {tag(case, factor)} layer={layer} "
                  f"experts={count} seconds={meta['compression_seconds']:.1f} "
                  f"init={initialization_seconds:.1f}", flush=True)
            del payload, reloaded, expert_states
        del sources, bases, residuals
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate(args, selected)


def aggregate(args, selected: list[tuple[str, int]]) -> None:
    errors, latency, memory, summary, budgets = [], [], [], [], []
    activation_path = args.output_dir / "activation_metadata.json"
    activation = json.loads(activation_path.read_text()) if activation_path.is_file() else {}
    for case, factor in selected:
        folder = args.output_dir / tag(case, factor)
        config = json.loads((folder / "config.json").read_text())
        metas = [json.loads(p.read_text()) for p in sorted(folder.glob("layer_*.meta.json"))]
        if sorted(m["layer"] for m in metas) != config["layers"]:
            raise ValueError(f"incomplete variant: {folder}")
        for meta in metas:
            layer = meta["layer"]
            errors.extend(read_csv(folder / f"layer_{layer}.errors.csv"))
            latency.extend(read_csv(folder / f"layer_{layer}.latency.csv"))
            memory.append({"case": case, "factor": factor, **meta})
            if case == "D":
                budgets.extend(read_csv(folder / f"layer_{layer}.budgets.csv"))
        metric_path = folder / "eval_metrics.json"
        accuracy = json.loads(metric_path.read_text()) if metric_path.is_file() else {}
        task_values = {f"acc_{t[0]}": t[1] for t in
                       accuracy.get("task_summary", {}).get("tasks", [])}
        rows = [r for r in errors if r["case"] == case and int(r["factor"]) == factor]
        timing = [r for r in latency if r["case"] == case and int(r["factor"]) == factor
                  and r["stage"] == "full_projection"]
        weights = len(config["layers"]) * (args.max_experts or 60) * 3 * 1408 * 2048
        bytes_total = sum(m["checkpoint_bytes"] for m in metas)
        summary.append({"case": case, "factor": factor,
                        "label": {"A": "Base only", "B": "Exact FP32 residual",
                                  "C": "QJL residual", "D": "ResCode-style Low-rank (Initialization Only)",
                                  "E": "GPTQ INT2 residual"}[case],
                        "accuracy_mean": accuracy.get("task_summary", {}).get("avg"),
                        **task_values,
                        "base_bytes": sum(m["base_bytes"] for m in metas),
                        "residual_bytes": sum(m["residual_bytes"] for m in metas),
                        "metadata_bytes": sum(m["metadata_bytes"] for m in metas),
                        "checkpoint_bytes": bytes_total,
                        "bits_per_weight": 8 * bytes_total / weights,
                        "projection_bank_gpu_bytes": accuracy.get("projection_bank_gpu_bytes"),
                        "expert_gpu_resident_bytes": accuracy.get("expert_gpu_resident_bytes"),
                        "gpu_allocated_bytes": accuracy.get("gpu_allocated_bytes"),
                        "peak_compression_gpu_allocated_bytes": max(
                            (m.get("peak_gpu_allocated_bytes") or 0 for m in metas), default=0),
                        "compression_seconds": sum(m["compression_seconds"] for m in metas),
                        "lowrank_initialization_seconds": (
                            sum(m.get("initialization_seconds", 0) for m in metas)
                            if case == "D" and all("initialization_seconds" in m for m in metas)
                            else None),
                        "calibration_collection_seconds": (
                            activation.get("calibration_seconds") if case == "E" else 0),
                        "mean_matched_rank": mean_field(
                            [r for r in budgets if r["case"] == case and int(r["factor"]) == factor],
                            "rank") if case == "D" else None,
                        "mean_budget_utilization": mean_field(
                            [r for r in budgets if r["case"] == case and int(r["factor"]) == factor],
                            "budget_utilization") if case == "D" else None,
                        "mean_residual_output_error": mean_field(rows, "residual_output_error"),
                        "mean_weight_output_error": mean_field(rows, "weight_output_error"),
                        "mean_expert_output_error": mean_field(rows, "expert_output_error"),
                        "median_projection_latency_ms": statistics.median(
                            float(r["latency_ms"]) for r in timing) if timing else None,
                        "median_projection_throughput_tps": statistics.median(
                            float(r["tokens_per_second"]) for r in timing) if timing else None,
                        "median_cuda_event_ms": statistics.median(
                            float(r["cuda_event_ms"]) for r in timing
                            if r["cuda_event_ms"] not in (None, "", "None")) if any(
                                r["cuda_event_ms"] not in (None, "", "None") for r in timing) else None})
    write_csv(args.output_dir / "projection_errors.csv", errors, FIELDS)
    write_csv(args.output_dir / "latency.csv", latency)
    write_csv(args.output_dir / "memory_breakdown.csv", memory)
    write_csv(args.output_dir / "group_budgets.csv", budgets)
    write_csv(args.output_dir / "summary.csv", summary)
    labels = {tag(r["case"], int(r["factor"])): r for r in summary}
    lines = ["# Residual A–E ablation", "", "All cases use the same packed exponent/mantissa Base.",
             "Held-out routed activations are from a separate corpus from GPTQ calibration.",
             "", "## Equal-budget comparison", ""]
    for name in ("C_d2", "D_d2", "E"):
        row = labels.get(name)
        if row:
            lines.append(f"- {name}: residual bytes {row['residual_bytes']}; "
                         f"mean weight output error {row['mean_weight_output_error']}; "
                         f"task accuracy {row['accuracy_mean']}.")
    lines += ["", "Case D uses ResCode covariance initialization only, on the fixed mantissa residual.",
              "Case E inference dequantizes packed INT2 before GEMM; its latency is not an optimized INT2 kernel.",
              "No downstream accuracy is reported until evaluation completes.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))


def mean_field(rows: list[dict], field: str) -> float | None:
    values = [float(r[field]) for r in rows if r.get(field) not in (None, "", "None")]
    return statistics.mean(values) if values else None


def evaluate(args, model_dir: Path, case: str, factor: int) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    folder = args.output_dir / tag(case, factor)
    config = json.loads((folder / "config.json").read_text())
    if config["case"] != case or config["factor"] != factor:
        raise ValueError("evaluation variant mismatch")
    if config["layers"] != list(range(24)) or config["max_experts"] is not None:
        raise ValueError("downstream evaluation requires all layers and experts")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    sizes = load_variant(model, folder, config, args.backend, args.fp32_residual_device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    metas = [json.loads(p.read_text()) for p in sorted(folder.glob("layer_*.meta.json"))]
    results = evaluate_fewshot(model, tokenizer, args.tasks, 0,
                               eval_batch_size=args.eval_batch_size,
                               output_path=str(folder / "lm_eval.txt"),
                               limit=args.eval_limit or None)
    task_summary = report_eval_summary(results, args.tasks, str(folder / "lm_eval.txt"),
                                       header=f"Residual ablation {tag(case, factor)}")
    metrics = {"case": case, "factor": factor, "tasks": args.tasks,
               "eval_limit": args.eval_limit, "task_summary": task_summary,
               "config_digest": config["digest"],
               "source_sha256s": [m["source_sha256"] for m in metas],
               **sizes, "gpu_allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}
    atomic_json(folder / "eval_metrics.json", metrics)


def sweep(args, model_dir: Path, selected: list[tuple[str, int]], layers: list[int]) -> None:
    command_base = [sys.executable, str(Path(__file__).resolve()),
                    "--model-path", str(model_dir), "--output-dir", str(args.output_dir)]
    compress(args, model_dir, layers, selected)
    if not args.skip_eval:
        if args.layers != "all" or args.max_experts is not None:
            raise ValueError("partial sweep requires --skip-eval")
        for case, factor in selected:
            folder = args.output_dir / tag(case, factor)
            path = folder / "eval_metrics.json"
            data = json.loads(path.read_text()) if path.is_file() else {}
            variant = json.loads((folder / "config.json").read_text())
            source_hashes = [json.loads(p.read_text())["source_sha256"]
                             for p in sorted(folder.glob("layer_*.meta.json"))]
            if (data.get("tasks") == args.tasks and data.get("eval_limit") == args.eval_limit
                    and data.get("config_digest") == variant["digest"]
                    and data.get("source_sha256s") == source_hashes):
                continue
            subprocess.run(command_base + ["--mode", "eval", "--cases", case,
                                           "--d-factors", str(factor or 1),
                                           "--tasks", args.tasks, "--eval-limit", str(args.eval_limit),
                                           "--eval-batch-size", str(args.eval_batch_size),
                                           "--backend", args.backend,
                                           "--fp32-residual-device", args.fp32_residual_device],
                           check=True, cwd=ROOT)
        aggregate(args, selected)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("sweep", "compress", "eval"), default="sweep")
    p.add_argument("--cases", default="A,B,C,D,E")
    p.add_argument("--d-factors", default="1,2,4")
    p.add_argument("--layers", default="all")
    p.add_argument("--max-experts", type=int)
    p.add_argument("--model-path", type=Path)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/residual_ablation")
    p.add_argument("--tasks", default=DEFAULT_TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--backend", choices=("triton", "reference"), default="triton")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fp32-residual-device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--qjl-seed", type=int, default=0)
    p.add_argument("--qjl-row-chunk", type=int, default=128)
    p.add_argument("--gptq-calibration-path", type=Path,
                   default=ROOT / "rescode/data/dclm-calib.jsonl")
    p.add_argument("--diagnostic-path", type=Path,
                   default=ROOT / "rescode/data/wikipedia-calib.jsonl")
    p.add_argument("--sample-cache", type=Path,
                   default=ROOT / "results/residual_ablation/routed_samples.pt")
    p.add_argument("--n-sentences", type=int, default=2400)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--gptq-calibration-samples", type=int, default=128)
    p.add_argument("--diagnostic-samples", type=int, default=16)
    p.add_argument("--gptq-damp", type=float, default=0.01)
    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--grouping-cache", type=Path,
                   default=ROOT / "results/qwen/calib_stats/dclm-n2400-seed42.pt")
    p.add_argument("--num-groups", type=int, default=45)
    p.add_argument("--singleton-ratio", type=float, default=0.15)
    p.add_argument("--grouping-rank", type=int, default=128)
    p.add_argument("--lowrank-alpha", type=float, default=2.0)
    p.add_argument("--latency-batch-size", type=int, default=4)
    p.add_argument("--latency-experts", type=int, default=1,
                   help="Measure the same first N expert IDs in every layer and case")
    p.add_argument("--latency-warmup", type=int, default=1)
    p.add_argument("--latency-repeats", type=int, default=2)
    args = p.parse_args()
    cases = parse_list(args.cases, set("ABCDE"))
    factors = [int(x) for x in parse_list(args.d_factors)]
    if any(f < 1 for f in factors) or not 1 <= (args.max_experts or 60) <= 60:
        p.error("factors and max-experts must be positive and within range")
    if args.latency_repeats < 1 or args.latency_batch_size < 1 or args.latency_experts < 1:
        p.error("latency repeats and batch size must be positive")
    return args, variants(cases, factors)


def main():
    args, selected = arguments()
    model_dir = Path(args.model_path).resolve() if args.model_path else None
    if model_dir is None:
        from run_base_qjl_sweep import resolve_model
        model_dir = resolve_model(None)
    if not (model_dir / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("complete original Qwen safetensors required")
    config = json.loads((model_dir / "config.json").read_text())
    if (config.get("num_hidden_layers"), config.get("num_experts"),
            config.get("hidden_size"), config.get("moe_intermediate_size")) != (24, 60, 2048, 1408):
        raise ValueError("expected Qwen1.5-MoE-A2.7B-Chat model configuration")
    layers = list(range(24)) if args.layers == "all" else sorted({int(x) for x in args.layers.split(",")})
    if not layers or layers[0] < 0 or layers[-1] > 23:
        raise ValueError("layers outside 0..23")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "config.json"
    if not config_path.is_file():
        atomic_json(config_path, {"cases": args.cases, "d_factors": args.d_factors,
                                  "layers": layers, "model_path": str(model_dir),
                                  "tasks": args.tasks, "eval_limit": args.eval_limit,
                                  "calibration_path": str(args.gptq_calibration_path),
                                  "diagnostic_path": str(args.diagnostic_path),
                                  "grouping_cache": str(args.grouping_cache),
                                  "qjl_seed": args.qjl_seed, "gptq_damp": args.gptq_damp,
                                  "lowrank_alpha": args.lowrank_alpha})
    if args.mode == "compress":
        compress(args, model_dir, layers, selected)
    elif args.mode == "eval":
        if len(selected) != 1:
            raise ValueError("eval needs one case/factor")
        evaluate(args, model_dir, *selected[0])
    else:
        sweep(args, model_dir, selected, layers)


if __name__ == "__main__":
    main()
