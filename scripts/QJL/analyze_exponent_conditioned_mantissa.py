#!/usr/bin/env python3
"""Weight-only comparison of row-wise and exponent-conditioned mantissa Bases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from safetensors import safe_open

from rescode.qjl.exponent_conditioned_mantissa import METHODS, analyze_matrix
from scripts.QJL.diagnose_codebook_base import (
    MODEL_PATH, PROJECTIONS, atomic_json, select,
)

DEFAULT_MODEL = Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat")
if not DEFAULT_MODEL.exists():
    DEFAULT_MODEL = MODEL_PATH


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records: list[dict]) -> list[dict]:
    """Aggregate squared norms and byte counts, never matrix-level ratios."""
    output = []
    for method in METHODS:
        selected = [r for r in records if r["method"] == method]
        if not selected:
            continue
        weight_sq = sum(r["weight_squared_norm"] for r in selected)
        residual_sq = sum(r["residual_squared_norm"] for r in selected)
        elements = sum(r["elements"] for r in selected)
        base_bytes = (sum(r["base_bytes"] for r in selected)
                      if method != "B1" else None)
        qjl_n = sum(r["qjl_d_n_projected_bytes"] for r in selected)
        qjl_2n = sum(r["qjl_d_2n_projected_bytes"] for r in selected)
        output.append({
            "method": method, "matrix_count": len(selected), "weight_squared_norm": weight_sq,
            "residual_squared_norm": residual_sq,
            "rho_global": math.sqrt(residual_sq / weight_sq) if weight_sq else None,
            "elements": elements, "base_bytes": base_bytes,
            "base_bits_per_weight": 8 * base_bytes / elements if base_bytes is not None else None,
            "qjl_d_n_projected_bytes": qjl_n, "qjl_d_2n_projected_bytes": qjl_2n,
            "projected_total_bits_per_weight_d_n": (
                8 * (base_bytes + qjl_n) / elements) if base_bytes is not None else None,
            "projected_total_bits_per_weight_d_2n": (
                8 * (base_bytes + qjl_2n) / elements) if base_bytes is not None else None,
            "projected_memory_reduction_vs_bf16_d_n": (
                1 - (base_bytes + qjl_n) / (2 * elements)) if base_bytes is not None else None,
            "projected_memory_reduction_vs_bf16_d_2n": (
                1 - (base_bytes + qjl_2n) / (2 * elements)) if base_bytes is not None else None,
            "fp16_worse_matrix_count_vs_A": (
                sum(bool(r["fp16_worse_matrix_vs_A"]) for r in selected) if method == "B2" else None),
            "fp16_worse_row_count_vs_A": (
                sum(r["fp16_worse_row_count_vs_A"] for r in selected) if method == "B2" else None),
        })
    baseline = next(row for row in output if row["method"] == "A")
    for row in output:
        a, b = baseline["rho_global"], row["rho_global"]
        row["absolute_rho_improvement_vs_A"] = a - b if a is not None else None
        row["relative_rho_improvement_vs_A"] = 1 - b / a if a else None
        row["residual_energy_ratio_vs_A"] = (
            row["residual_squared_norm"] / baseline["residual_squared_norm"]
            if baseline["residual_squared_norm"] else None)
    return output


def _number(value, digits=6):
    return "N/A" if value is None else f"{value:.{digits}f}"


def make_report(config: dict, overall: list[dict], projections: list[dict],
                layers: list[dict], statistics: list[dict], records: list[dict]) -> str:
    rows = {r["method"]: r for r in overall}
    b2_matrices = [r for r in records if r["method"] == "B2"]
    improved = sum(r["absolute_rho_improvement_vs_A"] > 1e-10 for r in b2_matrices)
    worsened = sum(r["absolute_rho_improvement_vs_A"] < -1e-10 for r in b2_matrices)
    unchanged = len(b2_matrices) - improved - worsened
    names = {"A": "A: existing FP16 row beta", "B1": "B1: exponent FP32 oracle",
             "B2": "B2: exponent FP16 beta"}
    lines = ["# Exponent-conditioned mantissa sharing", "",
             "Weight-only residual diagnostics. The existing encoder and QJL path were not changed.",
             f"Model: `{config['model_path']}`. Layers: {config['layers']}; "
             f"experts: {config['experts']}; projections: {config['projections']}; "
             f"minimum exponent count: {config['min_exponent_count']}.", "",
             "## Overall comparison", "",
             "| Method | Residual rho | Change vs A | Base bits/weight | D=n total bits | D=2n total bits |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        row = rows[method]
        change = row["absolute_rho_improvement_vs_A"]
        lines.append(f"| {names[method]} | {_number(row['rho_global'])} | "
                     f"{_number(change)} | {_number(row['base_bits_per_weight'], 3)} | "
                     f"{_number(row['projected_total_bits_per_weight_d_n'], 3)} | "
                     f"{_number(row['projected_total_bits_per_weight_d_2n'], 3)} |")
    lines += ["", "B1 uses FP32 beta only as an oracle; its storage fields are N/A. "
              "A Base bytes are measured from the existing packed payload. B2 Base "
              "bytes and all QJL totals are projected, not checkpoint file sizes.", "",
              "## By projection", "",
              "| Projection | A rho | B1 rho | B2 rho | B2 rho reduction | B2 residual energy / A |",
              "|---|---:|---:|---:|---:|---:|"]
    for projection in config["projections"]:
        subset = {r["method"]: r for r in projections if r["projection"] == projection}
        b2 = subset["B2"]
        lines.append(f"| {projection} | {_number(subset['A']['rho_global'])} | "
                     f"{_number(subset['B1']['rho_global'])} | {_number(b2['rho_global'])} | "
                     f"{_number(b2['absolute_rho_improvement_vs_A'])} | "
                     f"{_number(b2['residual_energy_ratio_vs_A'])} |")
    b2 = rows["B2"]
    extra = b2["base_bits_per_weight"] - rows["A"]["base_bits_per_weight"]
    oracle_gap = b2["rho_global"] - rows["B1"]["rho_global"]
    energy_reduction = 1 - b2["residual_energy_ratio_vs_A"]
    base_overhead_relative = extra / rows["A"]["base_bits_per_weight"]
    mean_within_row_std = sum(s["within_row_beta_std_mean"] for s in statistics) / len(statistics)
    mean_beta_difference = sum(s["beta_vs_row_weighted_mean_abs"] for s in statistics) / len(statistics)
    lines += ["", "## Findings", "",
              f"- B2 changes global rho by {_number(b2['absolute_rho_improvement_vs_A'])} "
              f"({_number(100 * b2['relative_rho_improvement_vs_A'], 3)}% relative) "
              f"and its residual energy ratio is {_number(b2['residual_energy_ratio_vs_A'])}.",
              f"- B2 lowers rho on {improved}/{len(b2_matrices)} matrices, leaves "
              f"{unchanged} effectively unchanged, and raises it on {worsened}. "
              f"FP16 storage worsens A on {b2['fp16_worse_matrix_count_vs_A']} "
              f"matrices and {b2['fp16_worse_row_count_vs_A']} rows.",
              f"- B2 minus B1 rho is {_number(oracle_gap, 8)}. "
              "This is the observed FP16 rounding penalty.",
              f"- Mean within-row exponent-beta standard deviation across matrices is "
              f"{_number(mean_within_row_std, 5)}; the mean count-weighted absolute "
              f"difference from row beta is {_number(mean_beta_difference, 5)}. "
              "The exponent-conditioned betas are measured separately from the row beta.",
              f"- B2 adds {_number(extra, 3)} Base bits per original weight. "
              f"Projected D=n and D=2n total bits are "
              f"{_number(b2['projected_total_bits_per_weight_d_n'], 3)} and "
              f"{_number(b2['projected_total_bits_per_weight_d_2n'], 3)}. "
              "These estimates include the same QJL sign bits and FP32 row norms; "
              "no QJL encoding was run.",
              f"- Projected memory reduction relative to 16-bit BF16 is "
              f"{_number(100 * b2['projected_memory_reduction_vs_bf16_d_n'], 2)}% "
              f"for D=n and {_number(100 * b2['projected_memory_reduction_vs_bf16_d_2n'], 2)}% "
              "for D=2n.",
              f"- In this weight-only comparison, residual energy falls by "
              f"{_number(100 * energy_reduction, 2)}% while estimated Base storage "
              f"rises by {_number(100 * base_overhead_relative, 2)}% "
              f"({_number(extra, 3)} bits/weight). "
              + ("This is a material residual reduction at the estimated overhead. "
                 if energy_reduction >= 0.05 else
                 "The measured residual reduction is small at the estimated overhead. ")
              + "Whether it improves QJL accuracy or runtime remains unmeasured.", ""]
    if len(config["layers"]) > 1:
        lines += ["## By layer", "",
                  "| Layer | A rho | B1 rho | B2 rho | B2 rho reduction |",
                  "|---:|---:|---:|---:|---:|"]
        for layer in config["layers"]:
            subset = {r["method"]: r for r in layers if r["layer"] == layer}
            lines.append(f"| {layer} | {_number(subset['A']['rho_global'])} | "
                         f"{_number(subset['B1']['rho_global'])} | "
                         f"{_number(subset['B2']['rho_global'])} | "
                         f"{_number(subset['B2']['absolute_rho_improvement_vs_A'])} |")
        lines.append("")
    lines += ["## Storage representation", "",
              "B2's estimate reuses the existing packed sign bytes, independently "
              "addressable lossless exponent blocks, and special-value table. "
              "Each present finite nonzero (row, original exponent) group stores "
              "one uint8 exponent ID and one FP16 beta. A uint32 row-offset array "
              "has rows+1 entries. Absent groups occupy no table entry. The dense "
              "[rows, 256] fitting array is temporary and is not counted as storage. "
              "Rare-group fallbacks still have one sparse entry with the row beta. "
              "B1 has no deployable storage claim.", "",
              "See `exponent_statistics.json` for group-size, beta-variation, "
              "fallback, and BF16 special-value counts. A full-model conclusion "
              "requires all 24 layers; this report describes only the selected sweep.", ""]
    return "\n".join(lines)


def run(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    model_path = args.model_path.expanduser().resolve()
    config = json.loads((model_path / "config.json").read_text())
    index = json.loads((model_path / "model.safetensors.index.json").read_text())
    layers = select(args.layers, config["num_hidden_layers"])
    experts = select(args.experts, config["num_experts"])
    projections = args.projections.split(",")
    if not projections or len(projections) != len(set(projections)) or any(
            value not in PROJECTIONS for value in projections):
        raise ValueError("--projections must be a unique subset of gate,up,down")
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory contains files: {out}; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    settings = {"model_path": str(model_path), "layers": layers, "experts": experts,
                "projections": projections, "device": device,
                "min_exponent_count": args.min_exponent_count,
                "storage_note": "B2 Base and QJL totals are projected; B1 is an FP32 oracle only"}
    atomic_json(out / "config.json", settings)
    records, statistics = [], []
    weight_map = index["weight_map"]
    with ExitStack() as stack:
        handles = {}
        for layer in layers:
            for expert in experts:
                for projection in projections:
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
                    shard = weight_map[name]
                    if shard not in handles:
                        handles[shard] = stack.enter_context(safe_open(
                            str(model_path / shard), framework="pt", device="cpu"))
                    weight = handles[shard].get_tensor(name).to(device)
                    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
                        raise ValueError(f"expected BF16 matrix: {name}")
                    methods, stats = analyze_matrix(weight, args.min_exponent_count)
                    records.extend({"layer": layer, "expert": expert,
                                    "projection": projection, **methods[method]}
                                   for method in METHODS)
                    statistics.append({"layer": layer, "expert": expert,
                                       "projection": projection, **stats})
                    print(f"[exponent-conditioned] L{layer} E{expert} {projection}: "
                          f"A={methods['A']['rho']:.6f} B1={methods['B1']['rho']:.6f} "
                          f"B2={methods['B2']['rho']:.6f}", flush=True)
                    del weight, methods, stats
    by_projection = [{"projection": p, **r}
                     for p in projections
                     for r in aggregate([item for item in records if item["projection"] == p])]
    by_layer = [{"layer": layer, **r}
                for layer in layers
                for r in aggregate([item for item in records if item["layer"] == layer])]
    overall = aggregate(records)
    write_csv(out / "per_matrix.csv", records)
    write_csv(out / "per_projection.csv", by_projection)
    write_csv(out / "per_layer.csv", by_layer)
    atomic_json(out / "summary.json", {"config": settings, "overall": overall,
                                       "by_projection": by_projection,
                                       "by_layer": by_layer})
    atomic_json(out / "exponent_statistics.json", {"records": statistics})
    (out / "REPORT.md").write_text(make_report(settings, overall, by_projection,
                                                by_layer, statistics, records))
    print(f"[exponent-conditioned] wrote {len(records)} method/matrix rows to {out}",
          flush=True)
    return {"overall": overall, "by_projection": by_projection, "by_layer": by_layer}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--layers", default="12")
    parser.add_argument("--experts", default="0")
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-exponent-count", type=int, choices=(1, 4, 16, 64), default=1)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/exponent_conditioned_mantissa")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
