#!/usr/bin/env python3
"""Weight-only shared input/output subspace ablation for Qwen MoE experts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open

from rescode.qjl.exponent_mantissa_base import (
    encode_exponent_mantissa_base, exponent_mantissa_base_bytes,
    reconstruct_base_from_weight,
)
from scripts.QJL.diagnose_codebook_base import MODEL_PATH, PROJECTIONS, atomic_json, select

METHODS = ("individual_svd", "shared_input", "shared_output")
DEFAULT_MODEL = Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = MODEL_PATH


def groups_for(experts: list[int], group_size: int, grouping: str, seed: int
               ) -> list[tuple[int, ...]]:
    if group_size < 1 or grouping not in ("sequential", "random"):
        raise ValueError("invalid group size or grouping")
    order = list(experts)
    if grouping == "random":
        random.Random(seed).shuffle(order)
    return [tuple(sorted(order[i:i + group_size]))
            for i in range(0, len(order), group_size)]


def parameter_count(method: str, members: int, rows: int, cols: int, rank: int) -> int:
    if method == "individual_svd":
        return members * rank * (rows + cols)
    if method == "shared_input":
        return cols * rank + members * rows * rank
    if method == "shared_output":
        return rows * rank + members * cols * rank
    raise ValueError(f"unknown method: {method}")


def baseline(weight: torch.Tensor) -> dict:
    """Measure the existing stored FP16 exponent-mantissa Base unchanged."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("expected a 2D BF16 matrix")
    state = encode_exponent_mantissa_base(weight)
    base = reconstruct_base_from_weight(weight, state["beta"].to(weight.device))
    fp32 = weight.float()
    energy = float(fp32.double().square().sum())
    residual = float((fp32 - base).double().square().sum())
    return {"weight_squared_norm": energy, "residual_squared_norm": residual,
            "rho": math.sqrt(residual / energy) if energy else None,
            "base_bytes": exponent_mantissa_base_bytes(state)}


def _energy(weight: torch.Tensor) -> float:
    return float(weight.double().square().sum())


def _metric(weight: torch.Tensor, oracle_preserved: float, oracle_residual: float,
            base_fp16: torch.Tensor) -> dict:
    energy = _energy(weight)
    residual_fp16 = _energy(weight - base_fp16)
    preserved_fp16 = _energy(base_fp16)
    if energy == 0:
        raise ValueError("zero-energy expert weight has undefined residual ratio")
    if abs(oracle_preserved + oracle_residual - energy) > 2e-4 * energy:
        raise ArithmeticError("orthogonal-basis energy identity failed")
    return {"weight_squared_norm": energy,
            "oracle_preserved_squared_norm": oracle_preserved,
            "oracle_residual_squared_norm": oracle_residual,
            "oracle_rho": math.sqrt(max(oracle_residual, 0.0) / energy),
            "oracle_eta": oracle_preserved / energy,
            "fp16_preserved_squared_norm": preserved_fp16,
            "fp16_residual_squared_norm": residual_fp16,
            "fp16_rho": math.sqrt(residual_fp16 / energy),
            "fp16_eta": preserved_fp16 / energy}


def _prefix_energies(coefficients: torch.Tensor, side: str) -> torch.Tensor:
    if side == "input":
        squared = coefficients.double().square().sum(dim=0)
    else:
        squared = coefficients.double().square().sum(dim=1)
    return torch.cat((torch.zeros(1, dtype=torch.float64, device=squared.device),
                      squared.cumsum(0)))


def _rank_metric(weight: torch.Tensor, coefficients: torch.Tensor,
                 basis: torch.Tensor, rank: int, side: str,
                 prefix: torch.Tensor | None = None) -> dict:
    """Compute oracle projection energy and actual FP16-factor reconstruction."""
    energy = _energy(weight)
    if rank == 0:
        return _metric(weight, 0.0, energy, torch.zeros_like(weight))
    if prefix is None:
        prefix = _prefix_energies(coefficients, side)
    captured = float(prefix[rank])
    tail = float(prefix[-1] - prefix[rank])
    if side == "input":
        left = coefficients[:, :rank].half().float()
        right = basis[:, :rank].half().float().T
    else:
        left = basis[:, :rank].half().float()
        right = coefficients[:rank, :].half().float()
    base_fp16 = left @ right
    return _metric(weight, captured, tail, base_fp16)


def individual_metrics(weight: torch.Tensor, ranks: list[int]) -> dict[int, dict]:
    """One SVD per expert; reuse its factors and spectrum across ranks."""
    if weight.dtype != torch.float32 or weight.ndim != 2:
        raise ValueError("individual SVD expects a 2D FP32 weight")
    # cuSOLVER's default Jacobi driver can lose ~1e-3 of Frobenius energy on
    # these large expert matrices; QR-based gesvd preserves the SVD optimum.
    u, singular, vh = torch.linalg.svd(
        weight, full_matrices=False, driver="gesvd" if weight.is_cuda else None)
    coefficients = u * singular[None, :]
    basis = vh.T.contiguous()
    prefix = _prefix_energies(coefficients, "input")
    return {rank: _rank_metric(weight, coefficients, basis, rank, "input", prefix)
            for rank in ranks}


def shared_metrics(weights: dict[int, torch.Tensor], ranks: list[int], side: str
                   ) -> dict[int, dict[int, dict]]:
    """One group Gram eigendecomposition for all requested ranks."""
    if not weights or side not in ("input", "output"):
        raise ValueError("expected nonempty weights and input/output side")
    shape = next(iter(weights.values())).shape
    if any(w.dtype != torch.float32 or w.shape != shape for w in weights.values()):
        raise ValueError("group weights must have identical 2D FP32 shape")
    rows, cols = shape
    dim = cols if side == "input" else rows
    gram = torch.zeros((dim, dim), device=next(iter(weights.values())).device,
                       dtype=torch.float32)
    for weight in weights.values():
        gram.add_(weight.T @ weight if side == "input" else weight @ weight.T)
    _, eigenvectors = torch.linalg.eigh(gram)
    basis = eigenvectors.flip(1).contiguous()
    output = {}
    for expert, weight in weights.items():
        coefficients = weight @ basis if side == "input" else basis.T @ weight
        prefix = _prefix_energies(coefficients, side)
        output[expert] = {rank: _rank_metric(weight, coefficients, basis, rank, side, prefix)
                          for rank in ranks}
    return output


def _group_rows(layer: int, projection: str, grouping: str, group_size: int,
                group_id: int, members: tuple[int, ...], weights: dict[int, torch.Tensor],
                ranks: list[int], individual_cache: dict[int, dict[int, dict]],
                reference_cache: dict[int, dict]) -> tuple[list[dict], list[dict]]:
    rows, cols = next(iter(weights.values())).shape
    member_count = len(members)
    shared = {}
    if member_count == 1:
        # One expert's optimal SVD spans the same input/output subspaces.
        for side in ("input", "output"):
            shared[side] = {members[0]: individual_cache[members[0]]}
    else:
        shared["input"] = shared_metrics(weights, ranks, "input")
        shared["output"] = shared_metrics(weights, ranks, "output")
    expert_rows, group_rows = [], []
    original_params = member_count * rows * cols
    qjl_n_bytes = member_count * rows * ((cols + 7) // 8 + 4)
    qjl_2n_bytes = member_count * rows * ((2 * cols + 7) // 8 + 4)
    baseline_energy = sum(reference_cache[e]["weight_squared_norm"] for e in members)
    baseline_residual = sum(reference_cache[e]["residual_squared_norm"] for e in members)
    baseline_bytes = sum(reference_cache[e]["base_bytes"] for e in members)
    baseline_rho = math.sqrt(baseline_residual / baseline_energy)
    for rank in ranks:
        for method in METHODS:
            by_expert = (individual_cache if method == "individual_svd" else
                         shared["input" if method == "shared_input" else "output"])
            params = parameter_count(method, member_count, rows, cols, rank)
            common = {"layer": layer, "projection": projection, "grouping": grouping,
                      "group_size": group_size, "group_id": group_id,
                      "members": ",".join(map(str, members)), "member_count": member_count,
                      "method": method, "rank": rank, "rows": rows, "cols": cols,
                      "base_parameters": params, "original_parameters": original_params,
                      "parameter_reduction_pct": 100 * (1 - params / original_params),
                      "fp16_base_bits_per_weight": 16 * params / original_params,
                      "projected_total_bits_per_weight_d_n": (
                          16 * params + 8 * qjl_n_bytes) / original_params,
                      "projected_total_bits_per_weight_d_2n": (
                          16 * params + 8 * qjl_2n_bytes) / original_params,
                      "baseline_rho": baseline_rho,
                      "baseline_base_bits_per_weight": 8 * baseline_bytes / original_params}
            selected = [by_expert[e][rank] for e in members]
            for expert, metrics in zip(members, selected):
                expert_rows.append({**common, "expert": expert, **metrics,
                                    "baseline_expert_rho": reference_cache[expert]["rho"],
                                    "baseline_expert_base_bytes": reference_cache[expert]["base_bytes"]})
            weight_energy = sum(item["weight_squared_norm"] for item in selected)
            oracle_residual = sum(item["oracle_residual_squared_norm"] for item in selected)
            fp16_residual = sum(item["fp16_residual_squared_norm"] for item in selected)
            group_rows.append({**common, "weight_squared_norm": weight_energy,
                               "oracle_residual_squared_norm": oracle_residual,
                               "fp16_residual_squared_norm": fp16_residual,
                               "oracle_rho": math.sqrt(oracle_residual / weight_energy),
                               "fp16_rho": math.sqrt(fp16_residual / weight_energy),
                               "oracle_eta": sum(item["oracle_preserved_squared_norm"]
                                                 for item in selected) / weight_energy,
                               "fp16_eta": sum(item["fp16_preserved_squared_norm"]
                                               for item in selected) / weight_energy,
                               "baseline_weight_squared_norm": baseline_energy,
                               "baseline_residual_squared_norm": baseline_residual,
                               "baseline_base_bytes": baseline_bytes,
                               "qjl_d_n_projected_bytes": qjl_n_bytes,
                               "qjl_d_2n_projected_bytes": qjl_2n_bytes})
    return expert_rows, group_rows


def aggregate_groups(group_rows: list[dict], target: float) -> list[dict]:
    buckets = defaultdict(list)
    for row in group_rows:
        key = (row["layer"], row["projection"], row["grouping"],
               row["group_size"], row["method"], row["rank"])
        buckets[key].append(row)
    result = []
    for key, rows in sorted(buckets.items()):
        energy = sum(row["weight_squared_norm"] for row in rows)
        oracle_residual = sum(row["oracle_residual_squared_norm"] for row in rows)
        fp16_residual = sum(row["fp16_residual_squared_norm"] for row in rows)
        original = sum(row["original_parameters"] for row in rows)
        params = sum(row["base_parameters"] for row in rows)
        qjl_n = sum(row["qjl_d_n_projected_bytes"] for row in rows)
        qjl_2n = sum(row["qjl_d_2n_projected_bytes"] for row in rows)
        baseline_residual = sum(row["baseline_residual_squared_norm"] for row in rows)
        result.append({"layer": key[0], "projection": key[1], "grouping": key[2],
                       "group_size": key[3], "method": key[4], "rank": key[5],
                       "group_count": len(rows), "expert_count": sum(r["member_count"] for r in rows),
                       "weight_squared_norm": energy,
                       "oracle_residual_squared_norm": oracle_residual,
                       "fp16_residual_squared_norm": fp16_residual,
                       "oracle_rho": math.sqrt(oracle_residual / energy),
                       "fp16_rho": math.sqrt(fp16_residual / energy),
                       "oracle_eta": sum(r["oracle_eta"] * r["weight_squared_norm"]
                                         for r in rows) / energy,
                       "fp16_eta": sum(r["fp16_eta"] * r["weight_squared_norm"]
                                       for r in rows) / energy,
                       "base_parameters": params, "original_parameters": original,
                       "parameter_reduction_pct": 100 * (1 - params / original),
                       "fp16_base_bits_per_weight": 16 * params / original,
                       "projected_total_bits_per_weight_d_n": (16 * params + 8 * qjl_n) / original,
                       "projected_total_bits_per_weight_d_2n": (16 * params + 8 * qjl_2n) / original,
                       "baseline_rho": math.sqrt(baseline_residual / energy),
                       "baseline_base_bits_per_weight": (
                           8 * sum(r["baseline_base_bytes"] for r in rows) / original),
                       "target_met_fp16": math.sqrt(fp16_residual / energy) <= target,
                       "target_met_oracle": math.sqrt(oracle_residual / energy) <= target})
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_plots(out: Path, summaries: list[dict], target: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {"individual_svd": "-", "shared_input": "--", "shared_output": ":"}
    colors = {size: color for size, color in zip(
        sorted({r["group_size"] for r in summaries}), ("C0", "C1", "C2", "C3", "C4", "C5"))}
    plots = (("rank_vs_residual.png", "rank", "fp16_rho", "FP16 residual norm ratio"),
             ("rank_vs_energy.png", "rank", "oracle_eta", "FP32 preserved energy fraction"),
             ("parameter_reduction_vs_residual.png", "parameter_reduction_pct", "fp16_rho",
              "FP16 residual norm ratio"))
    for filename, xkey, ykey, ylabel in plots:
        fig, axes = plt.subplots(3, 2, figsize=(15, 12), squeeze=False)
        for pi, projection in enumerate(PROJECTIONS):
            for gi, grouping in enumerate(("sequential", "random")):
                ax = axes[pi][gi]
                subset = [r for r in summaries if r["projection"] == projection
                          and r["grouping"] == grouping]
                for size in sorted(colors):
                    for method in METHODS:
                        curve = [r for r in subset if r["group_size"] == size
                                 and r["method"] == method]
                        if not curve:
                            continue
                        # Multiple layers are averaged by squared norms at each rank.
                        by_rank = defaultdict(list)
                        for row in curve:
                            by_rank[row["rank"]].append(row)
                        points = []
                        for rank, items in sorted(by_rank.items()):
                            w2 = sum(i["weight_squared_norm"] for i in items)
                            r2 = sum(i["fp16_residual_squared_norm"] for i in items)
                            eta = sum(i["oracle_eta"] * i["weight_squared_norm"]
                                      for i in items) / w2
                            reduction = sum(i["base_parameters"] for i in items)
                            original = sum(i["original_parameters"] for i in items)
                            points.append({"rank": rank, "fp16_rho": math.sqrt(r2 / w2),
                                           "oracle_eta": eta,
                                           "parameter_reduction_pct": 100 * (1 - reduction / original)})
                        points.sort(key=lambda item: item[xkey])
                        ax.plot([item[xkey] for item in points], [item[ykey] for item in points],
                                color=colors[size], linestyle=styles[method], marker="o",
                                markersize=2, linewidth=1, label=f"g={size} {method}")
                if ykey == "fp16_rho":
                    ax.axhline(target, color="black", linewidth=0.8, alpha=0.6)
                ax.set(title=f"{projection} / {grouping}", xlabel=xkey.replace("_", " "),
                       ylabel=ylabel)
                ax.grid(alpha=0.2)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=8)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)


def make_report(config: dict, summaries: list[dict], skipped: list[dict]) -> str:
    target = config["target_rho"]
    lines = ["# Qwen shared subspace ablation", "",
             "Weight-only FP32 oracle and FP16-factor reconstruction; no QJL encoding or inference.",
             f"Layers: {config['layers']}; projections: {config['projections']}; "
             f"experts: {len(config['experts'])}; target rho <= {target}.", "",
             "## Best practical rank in the requested sweep", "",
             "| Projection | Grouping | Group size | Method | Minimum rank meeting target | "
             "FP16 rho | Base reduction | Base bpw |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    scopes = sorted({(r["projection"], r["grouping"], r["group_size"], r["method"])
                     for r in summaries})
    for projection, grouping, size, method in scopes:
        subset = [r for r in summaries if (r["projection"], r["grouping"],
                                            r["group_size"], r["method"]) ==
                  (projection, grouping, size, method)]
        # Require every selected layer to meet the target at this rank.
        ranks = sorted({r["rank"] for r in subset})
        winning = next((rank for rank in ranks if all(
            r["target_met_fp16"] for r in subset if r["rank"] == rank)), None)
        chosen = winning if winning is not None else ranks[-1]
        chosen_rows = [r for r in subset if r["rank"] == chosen]
        w2 = sum(r["weight_squared_norm"] for r in chosen_rows)
        r2 = sum(r["fp16_residual_squared_norm"] for r in chosen_rows)
        params = sum(r["base_parameters"] for r in chosen_rows)
        original = sum(r["original_parameters"] for r in chosen_rows)
        lines.append(f"| {projection} | {grouping} | {size} | {method} | "
                     f"{winning if winning is not None else 'not met'} | "
                     f"{math.sqrt(r2 / w2):.6f} | {100 * (1 - params / original):.2f}% | "
                     f"{16 * params / original:.3f} |")
    # Pool layers by squared norms, and keep the all-layer target requirement.
    pooled = {}
    for projection, grouping, size, method in scopes:
        subset = [r for r in summaries if (r["projection"], r["grouping"],
                                            r["group_size"], r["method"]) ==
                  (projection, grouping, size, method)]
        for rank in sorted({r["rank"] for r in subset}):
            items = [r for r in subset if r["rank"] == rank]
            energy = sum(r["weight_squared_norm"] for r in items)
            original = sum(r["original_parameters"] for r in items)
            pooled[(projection, grouping, size, method, rank)] = {
                "rho": math.sqrt(sum(r["fp16_residual_squared_norm"] for r in items) / energy),
                "baseline_rho": math.sqrt(sum(
                    r["baseline_rho"] ** 2 * r["weight_squared_norm"] for r in items) / energy),
                "baseline_bpw": sum(r["baseline_base_bits_per_weight"] *
                                    r["original_parameters"] for r in items) / original,
                "reduction": 100 * (1 - sum(r["base_parameters"] for r in items) / original),
                "bpw": 16 * sum(r["base_parameters"] for r in items) / original,
                "all_layers_met": all(r["target_met_fp16"] for r in items),
            }
    lines += ["", "## Conclusions", ""]
    for projection in config["projections"]:
        ref = pooled.get((projection, config["groupings"][0],
                          config["group_sizes"][0], "individual_svd", 0))
        if ref is None:
            continue
        lines.append(f"### {projection}: measured exponent-mantissa reference")
        lines.append("")
        lines.append(f"The existing Base gives rho={ref['baseline_rho']:.6f} at "
                     f"{ref['baseline_bpw']:.3f} Base bpw on these weights.")
        lines.append("")
        for method, size in (("individual_svd", 1), ("shared_input", 4),
                             ("shared_output", 4), ("shared_input", 8),
                             ("shared_output", 8), ("shared_input", 60),
                             ("shared_output", 60)):
            key_prefix = (projection, config["groupings"][0], size, method)
            candidates = sorted((rank, data) for key, data in pooled.items()
                                if key[:4] == key_prefix for rank in (key[4],))
            if not candidates:
                continue
            at_budget = [item for item in candidates
                         if item[1]["bpw"] <= ref["baseline_bpw"]]
            budget_text = (f"best rho={at_budget[-1][1]['rho']:.6f} at "
                           f"rank {at_budget[-1][0]} within the reference Base bpw"
                           if at_budget else "no tested rank within the reference Base bpw")
            target_hit = next(((rank, data) for rank, data in candidates
                               if data["all_layers_met"]), None)
            target_text = (f"first all-layer target at rank {target_hit[0]} "
                           f"(rho={target_hit[1]['rho']:.6f}, "
                           f"reduction={target_hit[1]['reduction']:.2f}%, "
                           f"Base={target_hit[1]['bpw']:.3f} bpw)"
                           if target_hit else "target not reached in the sweep")
            lines.append(f"- {method}, group {size}: {budget_text}; {target_text}.")
        individual = [(rank, data) for key, data in pooled.items()
                      if key[:4] == (projection, config["groupings"][0], 1,
                                      "individual_svd")
                      for rank in (key[4],)]
        individual_budget = [item for item in individual
                             if item[1]["bpw"] <= ref["baseline_bpw"]]
        if individual_budget:
            best = max(individual_budget)
            lines.append(
                f"At the reference storage budget, individual SVD "
                f"{'does' if best[1]['rho'] <= target else 'does not'} reach "
                f"rho <= {target} (rank {best[0]}, rho={best[1]['rho']:.6f}).")
        shared_winners = [(key, data) for key, data in pooled.items()
                          if key[0] == projection and key[1] == config["groupings"][0]
                          and key[3] in ("shared_input", "shared_output")
                          and key[2] > 1 and data["all_layers_met"]
                          and data["reduction"] > 0]
        if shared_winners:
            key, data = min(shared_winners, key=lambda item: item[1]["bpw"])
            lines.append(f"A shared basis reaches the target with positive "
                         f"parameter reduction: {key[3]}, group {key[2]}, "
                         f"rank {key[4]}, reduction {data['reduction']:.2f}% "
                         f"and Base {data['bpw']:.3f} bpw.")
        else:
            lines.append("No tested shared group reaches the target with positive "
                         "Base parameter reduction across every selected layer.")
        probe = pooled.get((projection, config["groupings"][0], 1,
                            "individual_svd", 256))
        shared_input_probe = pooled.get((projection, config["groupings"][0], 4,
                                         "shared_input", 256))
        shared_output_probe = pooled.get((projection, config["groupings"][0], 4,
                                          "shared_output", 256))
        if probe and shared_input_probe and shared_output_probe:
            lines.append(f"At the same rank 256, group 4 has rho "
                         f"{shared_input_probe['rho']:.6f} (input) and "
                         f"{shared_output_probe['rho']:.6f} (output), versus "
                         f"{probe['rho']:.6f} for individual SVD. The shared "
                         "basis retains less energy at this rank.")
        lines.append("")
    shared_compressive = any(key[2] > 1 and key[3].startswith("shared_")
                             and data["all_layers_met"] and data["reduction"] > 0
                             for key, data in pooled.items())
    references = {projection: pooled.get((projection, config["groupings"][0],
                                          config["group_sizes"][0], "individual_svd", 0))
                  for projection in config["projections"]}
    individual_at_budget = any(
        key[3] == "individual_svd" and key[2] == 1 and data["all_layers_met"]
        and references[key[0]] and data["bpw"] <= references[key[0]]["baseline_bpw"]
        for key, data in pooled.items())
    shared_at_budget = any(
        key[2] > 1 and key[3].startswith("shared_") and data["all_layers_met"]
        and references[key[0]] and data["bpw"] <= references[key[0]]["baseline_bpw"]
        for key, data in pooled.items())
    lines += ["## Research answers", "",
              "1. **Are individual experts low rank at the reference storage budget?** "
              + ("At least one tested projection reaches the target." if
                 individual_at_budget else
                 "No tested projection reaches the target; see the individual SVD "
                 "residuals and ranks above."),
              "2. **Is there an effective shared low-rank subspace?** The "
              "rank-256 group-4 comparisons above show a larger residual than "
              "individual SVD. Shared directions exist, but the tested ranks do "
              "not preserve enough energy for the target with parameter savings.",
              "3. **Can a shared basis meet rho <= " + str(target) + " with "
              "parameter savings?** " + ("Yes; see the winning rows above." if
                                      shared_compressive else
                                      "No tested shared group does so across the selected layers."),
              "4. **What rank and reduction are needed?** The minimum-rank table "
              "gives the first FP16-factor rank meeting the target across every "
              "selected layer and its signed parameter reduction.",
              "5. **Does it beat the exponent-mantissa Base?** " +
              ("At least one shared configuration reaches the target within "
               "the measured reference Base budget." if shared_at_budget else
               "No tested shared configuration reaches the target within the "
               "measured reference Base budget."), "",
              "The individual SVD rows measure the best rank approximation per "
              "expert. The shared rows show the additional error "
              "from imposing one input or output basis across experts, and whether "
              "the target can be met with positive parameter reduction. All residual "
              "ratios above use squared-norm aggregation across groups and layers.", "",
              f"Skipped ranks beyond min(rows, cols): {len(skipped)} entries; "
              "see `summary.json`. FP16 factor bpw counts Base parameters only. "
              "Projected QJL D=n and D=2n storage in the CSV/JSON is a size estimate, "
              "not measured QJL behavior.", ""]
    return "\n".join(lines)


def run(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    model_path = args.model_path.expanduser().resolve()
    config = json.loads((model_path / "config.json").read_text())
    weight_map = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    layers = select(args.layers, config["num_hidden_layers"])
    experts = select(args.experts, config["num_experts"])
    projections = args.projections.split(",")
    if not projections or len(set(projections)) != len(projections) or any(
            p not in PROJECTIONS for p in projections):
        raise ValueError("invalid projections")
    group_sizes = [int(x) for x in args.group_sizes.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]
    groupings = args.groupings.split(",")
    if not group_sizes or len(group_sizes) != len(set(group_sizes)) or any(s < 1 for s in group_sizes):
        raise ValueError("invalid group sizes")
    if not ranks or len(ranks) != len(set(ranks)) or any(r < 0 for r in ranks):
        raise ValueError("invalid ranks")
    if not groupings or len(groupings) != len(set(groupings)) or any(
            g not in ("sequential", "random") for g in groupings):
        raise ValueError("invalid groupings")
    if not 0 < args.target_rho <= 1:
        raise ValueError("target rho must be in (0,1]")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory already contains files: {out}")
    out.mkdir(parents=True, exist_ok=True)
    settings = {"model_path": str(model_path), "layers": layers, "experts": experts,
                "projections": projections, "group_sizes": group_sizes,
                "groupings": groupings, "ranks": ranks, "device": device,
                "seed": args.seed, "target_rho": args.target_rho,
                "storage_note": "FP16 factors and theoretical QJL D=n,2n; no checkpoint or QJL run"}
    expert_rows, group_rows, skipped = [], [], []
    with ExitStack() as stack:
        handles = {}
        for layer in layers:
            for projection in projections:
                names = {expert: f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                         for expert in experts}
                def get_weight(expert):
                    name = names[expert]
                    shard = weight_map[name]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(safe_open(
                            str(model_path / shard), framework="pt", device="cpu"))
                    raw = handles[shard].get_tensor(name).to(device)
                    if raw.dtype != torch.bfloat16 or raw.ndim != 2:
                        raise ValueError(f"expected BF16 matrix: {name}")
                    return raw
                shape = tuple(get_weight(experts[0]).shape)
                valid_ranks = [r for r in ranks if r <= min(shape)]
                skipped += [{"layer": layer, "projection": projection, "rank": r,
                             "rows": shape[0], "cols": shape[1],
                             "reason": "rank exceeds min(rows, cols)"}
                            for r in ranks if r > min(shape)]
                if not valid_ranks:
                    continue
                individual_cache, reference_cache = {}, {}
                group_cache = {}
                for grouping in groupings:
                    for group_size in group_sizes:
                        seed = args.seed + 1009 * layer + 97 * PROJECTIONS.index(projection)
                        groups = groups_for(experts, group_size, grouping, seed)
                        for group_id, members in enumerate(groups):
                            key = members
                            if key not in group_cache:
                                weights = {}
                                for expert in members:
                                    raw = get_weight(expert)
                                    weights[expert] = raw.float()
                                    if expert not in reference_cache:
                                        reference_cache[expert] = baseline(raw)
                                    if expert not in individual_cache:
                                        individual_cache[expert] = individual_metrics(
                                            weights[expert], valid_ranks)
                                    del raw
                                group_cache[key] = _group_rows(
                                    layer, projection, grouping, group_size, group_id,
                                    members, weights, valid_ranks, individual_cache,
                                    reference_cache)
                                del weights
                            fresh_expert, fresh_group = group_cache[key]
                            # A cached group can occur with a different grouping or group ID.
                            expert_rows.extend({**row, "grouping": grouping,
                                                "group_size": group_size, "group_id": group_id}
                                               for row in fresh_expert)
                            group_rows.extend({**row, "grouping": grouping,
                                               "group_size": group_size, "group_id": group_id}
                                              for row in fresh_group)
                        print(f"[shared-subspace] L{layer} {projection} "
                              f"{grouping} g={group_size}: {len(groups)} groups", flush=True)
                del individual_cache, reference_cache, group_cache
                if device == "cuda":
                    torch.cuda.empty_cache()
    summaries = aggregate_groups(group_rows, args.target_rho)
    write_csv(out / "per_expert.csv", expert_rows)
    write_csv(out / "per_group.csv", group_rows)
    atomic_json(out / "summary.json", {"config": settings, "records": summaries,
                                       "skipped_ranks": skipped})
    save_plots(out, summaries, args.target_rho)
    (out / "REPORT.md").write_text(make_report(settings, summaries, skipped))
    print(f"[shared-subspace] wrote {len(expert_rows)} expert rows, "
          f"{len(group_rows)} group rows to {out}", flush=True)
    return {"records": summaries, "skipped_ranks": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="0,12,23")
    parser.add_argument("--experts", default="all")
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--group-sizes", default="1,4,8,60")
    parser.add_argument("--ranks", default="0,32,64,128,256,512,768,1024,1408,1792,2048")
    parser.add_argument("--groupings", default="sequential,random")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-rho", type=float, default=0.195)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/shared_subspace_qwen")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
