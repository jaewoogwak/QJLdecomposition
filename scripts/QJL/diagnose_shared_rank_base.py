#!/usr/bin/env python3
"""Training-free, weight-only shared low-rank Base and grouping diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open

from rescode.qjl.exponent_mantissa_base import (
    PackedExponentMantissaBase, encode_exponent_mantissa_base,
    exponent_mantissa_base_bytes,
)
from scripts.QJL.shared_rank_clustering import ResidualCosts, group_metrics, random_groups

PROJECTIONS = ("gate", "up", "down")
MODEL_PATH = Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat")
if not MODEL_PATH.exists():
    MODEL_PATH = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
                  / "snapshots/ec052fda178e241c7c443468d2fa1db6618996be")


def selection(value: str, size: int) -> list[int]:
    result = list(range(size)) if value == "all" else [int(v) for v in value.split(",")]
    if not result or len(result) != len(set(result)) or any(v < 0 or v >= size for v in result):
        raise ValueError(f"invalid selection {value!r}; valid range 0..{size - 1}")
    return result


def integer_list(value: str, minimum: int = 0) -> list[int]:
    result = [int(v) for v in value.split(",")]
    if not result or len(result) != len(set(result)) or any(v < minimum for v in result):
        raise ValueError(f"invalid integer list {value!r}")
    return result


def atomic_json(path: Path, value) -> None:
    temp = path.with_name(path.name + f".tmp{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temp = path.with_name(path.name + f".tmp{os.getpid()}")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def load_weights(model_path: Path, weight_map: dict, layer: int, projection: str,
                 experts: list[int], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only selected expert tensors from indexed safetensors shards."""
    original, converted = [], []
    with ExitStack() as stack:
        handles = {}
        for expert in experts:
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
            shard = weight_map[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(
                    str(model_path / shard), framework="pt", device="cpu"))
            weight = handles[shard].get_tensor(name)
            if weight.ndim != 2 or weight.dtype != torch.bfloat16:
                raise ValueError(f"expected BF16 matrix: {name}")
            original.append(weight)
            converted.append(weight.to(device=device, dtype=torch.float32))
    return torch.stack(converted), torch.stack(original)


def exponent_reference(original: torch.Tensor, device: str) -> dict:
    """Reuse the existing exponent-mantissa encoder and decoder per expert."""
    weight_sum, residual_sum, base_bytes = 0.0, 0.0, 0
    for weight in original:
        w = weight.to(device)
        state = encode_exponent_mantissa_base(w)
        decoded = PackedExponentMantissaBase(state, device).decode()
        fp32 = w.float()
        weight_sum += float(fp32.square().sum(dtype=torch.float64))
        residual_sum += float((fp32 - decoded).square().sum(dtype=torch.float64))
        base_bytes += exponent_mantissa_base_bytes(state)
        del w, state, decoded, fp32
    return {
        "exponent_mantissa_rho_global": math.sqrt(residual_sum / weight_sum) if weight_sum else None,
        "exponent_mantissa_base_bytes": base_bytes,
    }


def save_plot(out: Path, layer: int, projection: str, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relevant = [r for r in rows if r["layer"] == layer and r["projection"] == projection]
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(dict.fromkeys(r["method"] for r in relevant))
    for count in sorted(set(r["group_count"] for r in relevant)):
        for method in methods:
            curve = sorted((r for r in relevant if r["group_count"] == count
                            and r["method"] == method), key=lambda r: r["rank"])
            if curve:
                ax.plot([r["rank"] for r in curve], [r["rho_global"] for r in curve],
                        marker="o", label=f"C={count} {method}")
    reference = relevant[0].get("exponent_mantissa_rho_global") if relevant else None
    if reference is not None:
        ax.axhline(reference, color="black", linestyle=":", label="per-expert exponent-mantissa")
    ax.set(xlabel="Target rank", ylabel="Global residual Frobenius ratio",
           title=f"Layer {layer}, {projection} projection")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f"residual_vs_rank_layer{layer}_{projection}.png", dpi=160)
    plt.close(fig)


def run(args) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    model_path = args.model_path.expanduser().resolve()
    config = json.loads((model_path / "config.json").read_text())
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    layers = selection(args.layers, config["num_hidden_layers"])
    experts = selection(args.experts, config["num_experts"])
    projections = args.projections.split(",")
    if not projections or len(set(projections)) != len(projections) or any(p not in PROJECTIONS for p in projections):
        raise ValueError("--projections must be a unique comma-separated subset of gate,up,down")
    counts = integer_list(args.group_counts, 1)
    ranks = integer_list(args.ranks)
    methods = args.methods.split(",")
    if not methods or len(set(methods)) != len(methods) or any(
            m not in ("random", "cosine", "rank_aware") for m in methods):
        raise ValueError("--methods must be a unique subset of random,cosine,rank_aware")
    if any(c > len(experts) for c in counts):
        raise ValueError("group count exceeds selected expert count")
    if args.candidate_shortlist < 0 or args.qjl_d is not None and args.qjl_d < 1:
        raise ValueError("candidate shortlist must be nonnegative and QJL D positive")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_rows, expert_rows, groups_rows = [], [], []
    metadata = {
        "model_path": str(model_path), "device": device, "seed": args.seed,
        "layers": layers, "experts": experts, "projections": projections,
        "group_counts": counts, "ranks": ranks, "methods": methods,
        "candidate_shortlist": args.candidate_shortlist,
        "rank_aware_search": "exhaustive" if args.candidate_shortlist == 0 else "approximate_greedy",
        "qjl_d": args.qjl_d,
        "storage_note": "BF16 low-rank base and optional QJL residual are theoretical estimates; no checkpoint is written",
    }

    def flush():
        write_csv(out / "summary.csv", summary_rows)
        write_csv(out / "expert_metrics.csv", expert_rows)
        atomic_json(out / "summary.json", {"configuration": metadata, "records": summary_rows})
        atomic_json(out / "groups.json", groups_rows)

    for layer in layers:
        for projection in projections:
            print(f"[shared-rank] load layer={layer} projection={projection}", flush=True)
            weights, original = load_weights(model_path, weight_map, layer, projection, experts, device)
            reference = (exponent_reference(original, device) if not args.skip_exponent_reference
                         else {"exponent_mantissa_rho_global": None,
                               "exponent_mantissa_base_bytes": None})
            del original
            n, rows, cols = weights.shape
            spectrum_cache = {}
            for rank in ranks:
                costs = ResidualCosts(weights, rank, spectrum_cache)
                metric_cache = {}
                for count in counts:
                    seed = args.seed + layer * 1000003 + PROJECTIONS.index(projection) * 10007 + rank * 101 + count
                    groupings = {}
                    cluster_times = {}
                    cluster_peaks = {}
                    if "rank_aware" in methods:
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        groupings["rank_aware"] = costs.cluster(
                            count, "rank_aware", args.candidate_shortlist)
                        cluster_times["rank_aware"] = time.perf_counter() - start
                        cluster_peaks["rank_aware"] = (torch.cuda.max_memory_allocated()
                                                       if device == "cuda" else 0)
                    if "cosine" in methods:
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        groupings["cosine"] = costs.cluster(count, "cosine")
                        cluster_times["cosine"] = time.perf_counter() - start
                        cluster_peaks["cosine"] = (torch.cuda.max_memory_allocated()
                                                   if device == "cuda" else 0)
                    if "random" in methods:
                        start = time.perf_counter()
                        groupings["random"] = random_groups(n, count, seed)
                        cluster_times["random"] = time.perf_counter() - start
                        cluster_peaks["random"] = (torch.cuda.max_memory_allocated()
                                                   if device == "cuda" else 0)
                    if "rank_aware" in methods and "random" in methods:
                        start = time.perf_counter()
                        sizes = tuple(len(g) for g in groupings["rank_aware"])
                        groupings["random_size_matched"] = random_groups(n, count, seed, sizes)
                        cluster_times["random_size_matched"] = time.perf_counter() - start
                        cluster_peaks["random_size_matched"] = (torch.cuda.max_memory_allocated()
                                                                if device == "cuda" else 0)
                    for method, groups in groupings.items():
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        if groups not in metric_cache:
                            metric_cache[groups] = group_metrics(weights, groups, rank)
                        stats, items = metric_cache[groups]
                        diagnostics_seconds = time.perf_counter() - start
                        bits = stats["bf16_base_bits"]
                        row = {
                            "layer": layer, "projection": projection, "method": method,
                            "group_count": count, "rank": rank,
                            "effective_rank": stats["effective_rank"],
                            "rho_global": stats["rho_global"],
                            "captured_energy_fraction": stats["captured_energy_fraction"],
                            "expert_ratio_mean": stats["expert_ratio_mean"],
                            "expert_ratio_median": stats["expert_ratio_median"],
                            "expert_ratio_p90": stats["expert_ratio_p90"],
                            "expert_ratio_max": stats["expert_ratio_max"],
                            "weight_squared_norm": stats["weight_squared_norm"],
                            "residual_squared_norm": stats["residual_squared_norm"],
                            "group_sizes": json.dumps([len(g) for g in groups]),
                            "bf16_base_bits": bits,
                            "bf16_base_bits_per_weight": bits / (n * rows * cols),
                            "bf16_original_bits": stats["bf16_original_bits"],
                            "base_storage_ratio_to_bf16": bits / stats["bf16_original_bits"],
                            "theoretical_qjl_residual_bits": (8 * n * rows * ((args.qjl_d + 7) // 8 + 4)
                                                               if args.qjl_d is not None else None),
                            "clustering_seconds": cluster_times[method],
                            "diagnostics_seconds": diagnostics_seconds,
                            "peak_gpu_memory_bytes": (max(cluster_peaks[method],
                                                          torch.cuda.max_memory_allocated())
                                                      if device == "cuda" else 0),
                            "rank_aware_search": (metadata["rank_aware_search"]
                                                  if method == "rank_aware" else "not_applicable"),
                            **reference,
                        }
                        summary_rows.append(row)
                        groups_rows.append({
                            "layer": layer, "projection": projection, "method": method,
                            "group_count": count, "rank": rank,
                            "members": [[experts[i] for i in group] for group in groups],
                            "sizes": [len(group) for group in groups],
                        })
                        expert_rows.extend({
                            "layer": layer, "projection": projection, "method": method,
                            "group_count": count, "rank": rank,
                            "expert": experts[item["expert"]], "group": item["group"],
                            "weight_squared_norm": item["weight_squared_norm"],
                            "residual_squared_norm": item["residual_squared_norm"],
                            "residual_ratio": item["residual_ratio"],
                        } for item in items)
                        print(f"[shared-rank] L{layer} {projection} C={count} r={rank} "
                              f"{method} rho={stats['rho_global']:.6f}", flush=True)
                        flush()
            save_plot(out, layer, projection, summary_rows)
            del weights
            if device == "cuda":
                torch.cuda.empty_cache()
    print(f"[shared-rank] wrote {len(summary_rows)} configurations to {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--layers", default="0,12,23")
    parser.add_argument("--experts", default="all")
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--group-counts", default="1,4,8,60")
    parser.add_argument("--ranks", default="0,32,64,128,256")
    parser.add_argument("--methods", default="random,cosine,rank_aware")
    parser.add_argument("--candidate-shortlist", type=int, default=8,
                        help="Top cosine-nearest candidate pairs per merge; 0 evaluates all pairs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qjl-d", type=int, default=None,
                        help="Optional theoretical QJL residual width; no encoding is done")
    parser.add_argument("--skip-exponent-reference", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/shared_rank_diagnostics")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
