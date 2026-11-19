#!/usr/bin/env python3
"""Full SVD of individual-QJL and majority-QJL effective residual weights.

This reads original BF16 safetensors and the saved majority checkpoint. The
randomized SVD factors from compression are not used to form either residual.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rescode.qjl.core import ProjectionBank, pack_residual
from rescode.qjl.bundled_model import dense_qjl_base
from scripts.QJL.analyze_bundle_residual_svd import weight_tensor


MODEL = ("/workspace/.hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
         "/snapshots/ec052fda178e241c7c443468d2fa1db6618996be")
GROUPS = "results/qjl_direct/shared/qwen/groups-0c0fb7aea36af7a8.pt"
MAJORITY = "results/qjl_bundle/full_majority_d2048_r128/bundle_qjl_state.pt"
PROJECTIONS = ("gate", "up", "down")
MODES = ("none", "majority")
RANKS = (64, 128, 256, 512, 768, 1024)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=Path(MODEL))
    p.add_argument("--groups-path", type=Path, default=Path(GROUPS))
    p.add_argument("--majority-checkpoint", type=Path, default=Path(MAJORITY))
    p.add_argument("--layers", default="all")
    p.add_argument("--qjl-d", type=int, default=2048)
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--bundle-mode", choices=("none", "majority", "both"), default="both")
    p.add_argument("--figure-layer", type=int, default=12)
    p.add_argument("--max-experts", type=int, default=0,
                   help="0 means every non-singleton expert; use 2 for a smoke test")
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/qjl_bundle/spectrum_comparison"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.qjl_d < 1 or args.max_experts < 0:
        p.error("qjl-d must be positive and max-experts nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    return args


def select_layers(spec, available):
    layers = sorted(available) if spec == "all" else sorted({int(s) for s in spec.split(",")})
    if not layers or any(layer not in available for layer in layers):
        raise ValueError(f"invalid layer selection: {spec}")
    return layers


def spectrum_statistics(singular, weight_frobenius, residual_frobenius):
    """Optimal full-spectrum quantities, separate from stored randomized SVD."""
    s = np.asarray(singular, dtype=np.float64)
    if s.ndim != 1 or len(s) == 0 or np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError("finite nonnegative singular values required")
    if np.any(np.diff(s) > 1e-4 * max(float(s[0]), 1.0)):
        raise AssertionError("singular values are not descending")
    energy = np.square(s)
    total = float(energy.sum())
    if total <= 0 or weight_frobenius <= 0:
        raise ValueError("zero-norm residual or original weight")
    normalized = energy / total
    cumulative = np.cumsum(normalized)
    if np.any(np.diff(cumulative) < -1e-10) or abs(cumulative[-1] - 1) > 1e-7:
        raise AssertionError("cumulative singular energy is invalid")
    ranks = {q: int(np.searchsorted(cumulative, q, side="left") + 1)
             for q in (0.9, 0.95, 0.99)}
    if not ranks[0.9] <= ranks[0.95] <= ranks[0.99]:
        raise AssertionError("effective ranks are not ordered")
    metrics = {"weight_frobenius": float(weight_frobenius),
               "residual_frobenius": float(residual_frobenius),
               "residual_norm_ratio": float(residual_frobenius / weight_frobenius),
               "r90": ranks[0.9], "r95": ranks[0.95], "r99": ranks[0.99]}
    for rank in RANKS:
        retained = float(cumulative[min(rank, len(s)) - 1])
        metrics[f"energy_r{rank}"] = retained
        # The exact optimal rank-r error is defined from the full singular tail.
        tail = float(energy[min(rank, len(s)):].sum())
        metrics[f"optimal_weight_error_r{rank}"] = math.sqrt(tail) / weight_frobenius
    return metrics, normalized.astype(np.float32), cumulative.astype(np.float32)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_fingerprint(checkpoint):
    root = checkpoint.parent
    result = {"checkpoint_size": checkpoint.stat().st_size,
              "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns}
    for relative in ("metrics.json", "summary.json", "eval_arc/metrics.json"):
        path = root / relative
        if path.is_file():
            result[relative] = _sha256(path)
    return result


def analyze_layer(layer, state, saved_groups, bank, snapshot, weight_map,
                  modes, device, max_experts):
    values, metadata, records = [], [], []
    layer_groups = state["layers"][layer]["groups"]
    if layer_groups != saved_groups[layer]:
        raise AssertionError(f"ResCode group membership differs at layer {layer}")
    flat = sorted(i for group in layer_groups for i in group)
    if flat != list(range(60)):
        raise AssertionError(f"layer {layer} does not partition all 60 routed experts")
    for group_id, members in enumerate(layer_groups):
        if len(members) == 1:
            continue
        cluster = state["layers"][layer]["clusters"][group_id]
        if sorted(cluster["experts"]) != sorted(members):
            raise AssertionError(f"checkpoint expert membership differs: layer={layer} group={group_id}")
        chosen = sorted(members)[:max_experts or None]
        for projection in PROJECTIONS:
            first = weight_tensor(snapshot, weight_map, layer, chosen[0], projection)
            rows, cols = first.shape
            shared_packed = cluster["sketches"][projection]["shared"].to(device)
            if shared_packed.shape != (rows, (bank.d + 7) // 8):
                raise AssertionError("shared sketch shape mismatch")
            for expert in chosen:
                original = first if expert == chosen[0] else weight_tensor(
                    snapshot, weight_map, layer, expert, projection)
                weight = original.to(device)
                if weight.dtype != torch.bfloat16 or weight.shape != (rows, cols):
                    raise AssertionError("original BF16 weight shape or dtype changed")
                individual_packed, individual_norms = pack_residual(weight, bank)
                stored_norms = cluster["experts"][expert][projection]["norms"].to(device)
                if not torch.equal(individual_norms, stored_norms):
                    raise AssertionError(f"row norms differ at {layer}/{expert}/{projection}")
                weight_fp32 = weight.float()
                weight_norm = torch.linalg.vector_norm(weight_fp32.double())
                for mode in modes:
                    packed = individual_packed if mode == "none" else shared_packed
                    base = torch.cat([part for _, part in dense_qjl_base(
                        packed, stored_norms, bank, cols)], dim=0)
                    if base.shape != weight.shape:
                        raise AssertionError("QJL base shape mismatch")
                    residual = weight_fp32 - base
                    singular = torch.linalg.svdvals(residual.float())
                    residual_norm = torch.linalg.vector_norm(residual.double())
                    reconstruction_error = float(
                        torch.linalg.vector_norm((base + residual - weight_fp32).double()) /
                        weight_norm.clamp_min(1e-30))
                    svd_energy_gap = float(abs(singular.double().square().sum() -
                        residual.double().square().sum()) /
                        residual.double().square().sum().clamp_min(1e-30))
                    if reconstruction_error > 1e-6 or svd_energy_gap > 1e-3:
                        raise AssertionError(f"full SVD or reconstruction failed at "
                                             f"{layer}/{expert}/{projection}/{mode}")
                    spectrum = singular.cpu().numpy().astype(np.float32)
                    metrics, _, _ = spectrum_statistics(
                        spectrum, float(weight_norm), float(residual_norm))
                    record = {"layer": layer, "group": group_id, "expert": expert,
                              "projection": projection, "bundle_mode": mode,
                              **metrics,
                              "svd_energy_relative_gap": svd_energy_gap,
                              "reconstruction_relative_error": reconstruction_error}
                    records.append(record)
                    values.append(spectrum)
                    metadata.append((layer, group_id, expert, projection, mode))
                    del residual, singular, base
                del weight, weight_fp32, individual_packed, individual_norms
            del first, shared_packed
    return np.stack(values), metadata, records


def load_saved_layers(output, layers):
    values, metadata, records = [], [], []
    for layer in layers:
        path = output / "layers" / f"layer_{layer:02d}.npz"
        with np.load(path) as saved:
            values.append(saved["singular_values"])
            metadata.extend(zip(saved["layer"].tolist(), saved["group"].tolist(),
                                saved["expert"].tolist(), saved["projection"].tolist(),
                                saved["bundle_mode"].tolist()))
            records.extend(json.loads(str(saved["metrics_json"])))
    return np.concatenate(values), metadata, records


def evaluation_metrics(output_root):
    rows = []
    for rank in (64, 128, 512):
        for mode, prefix in (("none", "individual"), ("majority", "full_majority")):
            path = output_root / f"{prefix}_d2048_r{rank}"
            metrics_path = path / "metrics.json"
            eval_path = path / "eval_arc" / "metrics.json"
            if not metrics_path.is_file() or not eval_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text())
            evaluation = json.loads(eval_path.read_text())
            for task in ("arc_easy", "arc_challenge"):
                accuracy = evaluation.get("task_results", {}).get(task, {}).get("acc,none")
                if accuracy is None:
                    continue
                rows.append({"bundle_mode": mode, "rank": rank, "task": task,
                             "accuracy": accuracy,
                             "checkpoint_bytes": metrics["checkpoint_bytes"],
                             "compression_ratio": metrics["routed_bf16_compression_ratio"],
                             "bits_per_routed_weight": metrics["effective_bits_per_routed_weight"],
                             "checkpoint_reload_probe_max_abs": metrics["checkpoint_reload_probe_max_abs"]})
    return rows


def actual_compression_metrics(output_root, records):
    """Keep stored randomized/BF16 errors separate from exact SVD bounds."""
    lookup = {(r["layer"], r["expert"], r["projection"], r["bundle_mode"]): r
              for r in records}
    output = []
    for rank in (64, 128, 512):
        for mode, prefix in (("none", "individual"), ("majority", "full_majority")):
            path = output_root / f"{prefix}_d2048_r{rank}" / "diagnostics.csv"
            if not path.is_file():
                continue
            with path.open(newline="") as source:
                for actual in csv.DictReader(source):
                    key = (int(actual["layer"]), int(actual["expert"]),
                           actual["projection"], mode)
                    optimum = lookup.get(key)
                    if optimum is None:
                        continue
                    output.append({"layer": key[0], "expert": key[1],
                        "projection": key[2], "bundle_mode": mode, "rank": rank,
                        "optimal_weight_error": optimum[f"optimal_weight_error_r{rank}"],
                        "actual_weight_relative_error": float(actual["weight_relative_error"]),
                        "actual_residual_energy_retention": float(actual["svd_energy_retention"])})
    return output


def save_actual_table(output, rows):
    path = output / "actual_vs_optimal.csv"
    columns = ("layer", "expert", "projection", "bundle_mode", "rank",
               "optimal_weight_error", "actual_weight_relative_error",
               "actual_residual_energy_retention")
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_actual(rows):
    grouped = {}
    for mode in MODES:
        for projection in (*PROJECTIONS, "all"):
            for rank in (64, 128, 512):
                chosen = [r for r in rows if r["bundle_mode"] == mode and
                          r["rank"] == rank and
                          (projection == "all" or r["projection"] == projection)]
                if not chosen:
                    continue
                gap = np.array([r["actual_weight_relative_error"] -
                                r["optimal_weight_error"] for r in chosen], dtype=float)
                grouped[f"{mode}/{projection}/r{rank}"] = {
                    "count": len(chosen), "actual_minus_optimal_mean": float(gap.mean()),
                    "actual_minus_optimal_median": float(np.median(gap)),
                    "fraction_actual_below_optimal_beyond_1e-3": float((gap < -1e-3).mean())}
    return grouped


def _save_plot(fig, output, name):
    directory = output / "plots"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(output, singular, metadata, records, cumulative, evaluation, figure_layer):
    """All plots use exact full spectra except the separately labeled ARC data."""
    layer = np.array([r[0] for r in metadata], dtype=np.int16)
    expert = np.array([r[2] for r in metadata], dtype=np.int16)
    projection = np.array([r[3] for r in metadata])
    mode = np.array([r[4] for r in metadata])
    normalized = singular / np.maximum(singular[:, :1], 1e-30)
    spectrum_log = np.log10(np.maximum(normalized, 1e-6))
    all_layers = sorted(set(layer.tolist()))
    x_max = singular.shape[1]
    for name in MODES:
        if not np.any(mode == name):
            continue
        for proj in PROJECTIONS:
            selection = (mode == name) & (projection == proj)
            if not np.any(selection):
                continue
            layer_average = np.stack([
                normalized[selection & (layer == index)].mean(axis=0)
                for index in all_layers if np.any(selection & (layer == index))])
            present_layers = [index for index in all_layers
                              if np.any(selection & (layer == index))]
            fig, ax = plt.subplots(figsize=(13, max(3.5, 0.25 * len(present_layers) + 1.5)))
            image = ax.imshow(np.log10(np.maximum(layer_average, 1e-6)),
                              aspect="auto", interpolation="nearest", origin="lower",
                              extent=(1, x_max, -.5, len(present_layers)-.5),
                              vmin=-4, vmax=0, cmap="viridis")
            ax.set(yticks=np.arange(len(present_layers)), yticklabels=present_layers,
                   xlabel="Singular-value index",
                   ylabel="MoE layer", title=f"{proj.capitalize()} / {name}: layer-mean normalized spectrum")
            fig.colorbar(image, ax=ax, label="log10(mean sigma_k / sigma_1)")
            _save_plot(fig, output, f"layer_spectrum_{proj}_{name}")
            chosen = selection & (layer == figure_layer)
            if np.any(chosen):
                grid = np.full((60, x_max), np.nan, dtype=np.float32)
                grid[expert[chosen]] = spectrum_log[chosen]
                fig, ax = plt.subplots(figsize=(13, 8))
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad("lightgray")
                image = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest",
                                  extent=(1, x_max, -.5, 59.5), cmap=cmap, vmin=-4, vmax=0)
                ax.set(xlabel="Singular-value index", ylabel="Original expert ID",
                       title=f"{proj.capitalize()} / {name} / layer {figure_layer}: expert spectra")
                fig.colorbar(image, ax=ax, label="log10(sigma_k / sigma_1); gray = singleton")
                _save_plot(fig, output, f"expert_spectrum_layer{figure_layer}_{proj}_{name}")
            for percentile in (90, 95, 99):
                grid = np.full((24, 60), np.nan, dtype=np.float32)
                for record in records:
                    if record["bundle_mode"] == name and record["projection"] == proj:
                        grid[record["layer"], record["expert"]] = record[f"r{percentile}"]
                fig, ax = plt.subplots(figsize=(13, 6))
                cmap = plt.get_cmap("plasma").copy()
                cmap.set_bad("lightgray")
                image = ax.imshow(grid, aspect="auto", origin="lower", interpolation="nearest",
                                  vmin=1, vmax=x_max, cmap=cmap)
                ax.set(xlabel="Original expert ID", ylabel="MoE layer",
                       title=f"{proj.capitalize()} / {name}: r{percentile} effective rank")
                fig.colorbar(image, ax=ax, label=f"r{percentile}; gray = singleton or unselected")
                _save_plot(fig, output, f"effective_r{percentile}_{proj}_{name}")
    for proj in (*PROJECTIONS, "all"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for name, color in (("none", "#1678ad"), ("majority", "#d26425")):
            selection = (mode == name) & ((projection == proj) if proj != "all" else True)
            if np.any(selection):
                ax.plot(np.arange(1, x_max + 1), cumulative[selection].mean(axis=0),
                        label=f"{name} (n={int(selection.sum())})", color=color)
        for rank in (64, 128, 512):
            ax.axvline(rank, color="gray", linestyle=":", linewidth=.8)
        ax.set(xlabel="Rank", ylabel="Mean cumulative residual energy retained",
               ylim=(0, 1.01), title=f"{proj.capitalize()}: exact full-SVD energy")
        ax.legend()
        _save_plot(fig, output, f"cumulative_energy_{proj}")
    keyed = {(r["layer"], r["expert"], r["projection"], r["bundle_mode"]): r
             for r in records}
    for proj in PROJECTIONS:
        pairs = []
        for record in records:
            if record["bundle_mode"] != "none" or record["projection"] != proj:
                continue
            other = keyed.get((record["layer"], record["expert"], proj, "majority"))
            if other is not None:
                pairs.append((record, other))
        if not pairs:
            continue
        for field in ("residual_norm_ratio", "r90", "r95",
                      "optimal_weight_error_r64", "optimal_weight_error_r128",
                      "optimal_weight_error_r512"):
            x = np.array([a[field] for a, _ in pairs], dtype=float)
            y = np.array([b[field] for _, b in pairs], dtype=float)
            lower, upper = min(x.min(), y.min()), max(x.max(), y.max())
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(x, y, s=7, alpha=.35)
            ax.plot([lower, upper], [lower, upper], "k--", linewidth=1)
            ax.set(xlabel=f"Individual {field}", ylabel=f"Majority {field}",
                   title=f"{proj.capitalize()}: paired experts, exact SVD (n={len(pairs)})")
            _save_plot(fig, output, f"compare_{field}_{proj}")
    if evaluation:
        for task in ("arc_easy", "arc_challenge"):
            fig, ax = plt.subplots(figsize=(8, 5))
            for name, shift, color in (("none", -.18, "#1678ad"),
                                       ("majority", .18, "#d26425")):
                rows = [r for r in evaluation if r["bundle_mode"] == name and r["task"] == task]
                if rows:
                    ax.bar(np.array([{64: 0, 128: 1, 512: 2}[r["rank"]]
                                     for r in rows], dtype=float) + shift,
                           [100 * r["accuracy"] for r in rows], width=.34,
                           label=name, color=color)
            ax.set(xticks=np.arange(3), xticklabels=["64", "128", "512"],
                   xlabel="Residual rank", ylabel="ARC accuracy (%)",
                   title=f"{task}: full zero-shot evaluation")
            ax.legend()
            _save_plot(fig, output, f"arc_accuracy_{task}")


def save_tables(output, singular, metadata, records, evaluation):
    with (output / "spectrum_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    with (output / "evaluation.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("bundle_mode", "rank", "task", "accuracy",
            "checkpoint_bytes", "compression_ratio", "bits_per_routed_weight",
            "checkpoint_reload_probe_max_abs"), lineterminator="\n")
        writer.writeheader(); writer.writerows(evaluation)
    normalized = np.square(singular.astype(np.float64))
    normalized /= normalized.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(normalized, axis=1)
    if np.any(np.diff(cumulative, axis=1) < -1e-10) or np.max(abs(cumulative[:, -1] - 1)) > 1e-7:
        raise AssertionError("saved cumulative energy is invalid")
    path = output / "singular_values.npz"
    tmp = output / "singular_values.tmp.npz"
    np.savez_compressed(tmp, singular_values=singular,
        normalized_squared=normalized.astype(np.float32),
        cumulative_energy=cumulative.astype(np.float32),
        layer=np.array([r[0] for r in metadata], dtype=np.int16),
        group=np.array([r[1] for r in metadata], dtype=np.int16),
        expert=np.array([r[2] for r in metadata], dtype=np.int16),
        projection=np.array([r[3] for r in metadata]),
        bundle_mode=np.array([r[4] for r in metadata]))
    os.replace(tmp, path)
    return cumulative.astype(np.float32)


def summarize(records, evaluation, fingerprint, config):
    grouped = {}
    fields = ("residual_norm_ratio", "r90", "r95", "r99",
              "energy_r64", "energy_r128", "energy_r256", "energy_r512",
              "optimal_weight_error_r64", "optimal_weight_error_r128",
              "optimal_weight_error_r512")
    for mode in MODES:
        grouped[mode] = {}
        for projection in (*PROJECTIONS, "all"):
            chosen = [r for r in records if r["bundle_mode"] == mode and
                      (projection == "all" or r["projection"] == projection)]
            if not chosen:
                continue
            grouped[mode][projection] = {"count": len(chosen)}
            for field in fields:
                vals = np.array([r[field] for r in chosen], dtype=np.float64)
                grouped[mode][projection][field] = {
                    "mean": float(vals.mean()), "median": float(np.median(vals)),
                    "p90": float(np.percentile(vals, 90))}
    keyed = {(r["layer"], r["expert"], r["projection"], r["bundle_mode"]): r
             for r in records}
    paired = {}
    for projection in (*PROJECTIONS, "all"):
        keys = sorted({(r["layer"], r["expert"], r["projection"]) for r in records
                       if projection == "all" or r["projection"] == projection})
        pairs = [(keyed[(*key, "none")], keyed[(*key, "majority")])
                 for key in keys if (*key, "none") in keyed and (*key, "majority") in keyed]
        if not pairs:
            continue
        paired[projection] = {"count": len(pairs)}
        for field in ("residual_norm_ratio", "r90", "r95", "r99",
                      "optimal_weight_error_r64", "optimal_weight_error_r128",
                      "optimal_weight_error_r512"):
            difference = np.array([b[field] - a[field] for a, b in pairs], dtype=float)
            paired[projection][field] = {"majority_minus_individual_mean": float(difference.mean()),
                "majority_minus_individual_median": float(np.median(difference)),
                "fraction_majority_higher": float((difference > 0).mean())}
    return {"config": config, "matrix_count": len(records),
            "per_mode_projection": grouped, "evaluation": evaluation,
            "paired_comparison": paired,
            "majority_input_fingerprint": fingerprint,
            "max_svd_energy_relative_gap": max(r["svd_energy_relative_gap"] for r in records),
            "max_reconstruction_relative_error": max(r["reconstruction_relative_error"] for r in records)}


def main():
    args = arguments()
    os.chdir(ROOT)
    snapshot = args.model_path.resolve()
    checkpoint = args.majority_checkpoint.resolve()
    groups_path = args.groups_path.resolve()
    output = args.output_dir.resolve()
    if not (snapshot / "model.safetensors.index.json").is_file() or not checkpoint.is_file() or not groups_path.is_file():
        raise FileNotFoundError("original model, ResCode grouping, or majority checkpoint missing")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state.get("format") != "qjl-majority-lowrank-v1":
        raise ValueError("invalid majority checkpoint format")
    if state["config"]["bundle_mode"] != "majority" or state["config"]["qjl_d"] != args.qjl_d or state["config"]["projection_seed"] != args.projection_seed:
        raise ValueError("majority checkpoint projection configuration differs")
    saved_groups = torch.load(groups_path, map_location="cpu", weights_only=False)["groups"]
    layers = select_layers(args.layers, state["layers"])
    modes = MODES if args.bundle_mode == "both" else (args.bundle_mode,)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    bank = ProjectionBank(args.qjl_d, args.projection_seed)
    fingerprint = baseline_fingerprint(checkpoint)
    config = {"model_path": str(snapshot), "groups_path": str(groups_path),
              "majority_checkpoint": str(checkpoint), "layers": layers,
              "qjl_d": args.qjl_d, "projection_seed": args.projection_seed,
              "bundle_mode": args.bundle_mode, "max_experts": args.max_experts,
              "figure_layer": args.figure_layer,
              "svd_method": "torch.linalg.svdvals FP32", "device": args.device}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "analysis_config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError(f"output directory has different analysis settings: {output}")
    if not manifest.exists() and (output / "layers").exists() and any((output / "layers").glob("*.npz")):
        raise ValueError("existing layer files without configuration manifest")
    manifest.write_text(json.dumps(config, indent=2))
    (output / "layers").mkdir(exist_ok=True)
    device = torch.device(args.device)
    started = time.perf_counter()
    prior_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for layer in layers:
            path = output / "layers" / f"layer_{layer:02d}.npz"
            if path.exists():
                print(f"[spectrum] reused layer={layer}", flush=True)
                continue
            tick = time.perf_counter()
            values, metadata, records = analyze_layer(layer, state, saved_groups,
                bank, snapshot, index, modes, device, args.max_experts)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, singular_values=values,
                layer=np.array([x[0] for x in metadata], dtype=np.int16),
                group=np.array([x[1] for x in metadata], dtype=np.int16),
                expert=np.array([x[2] for x in metadata], dtype=np.int16),
                projection=np.array([x[3] for x in metadata]),
                bundle_mode=np.array([x[4] for x in metadata]),
                metrics_json=json.dumps(records))
            os.replace(tmp, path)
            print(f"[spectrum] layer={layer} matrices={len(records)} seconds={time.perf_counter()-tick:.1f}", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32
    singular, metadata, records = load_saved_layers(output, layers)
    evaluation = evaluation_metrics(Path("results/qjl_bundle"))
    cumulative = save_tables(output, singular, metadata, records, evaluation)
    actual = actual_compression_metrics(Path("results/qjl_bundle"), records)
    save_actual_table(output, actual)
    make_plots(output, singular, metadata, records, cumulative,
               evaluation, args.figure_layer)
    summary = summarize(records, evaluation, fingerprint, config)
    summary["singleton_counts"] = {str(layer): sum(len(group) == 1 for group in saved_groups[layer])
                                   for layer in layers}
    summary["non_singleton_counts"] = {str(layer): sum(len(group) for group in saved_groups[layer]
                                                       if len(group) > 1) for layer in layers}
    summary["actual_randomized_bf16_rows"] = len(actual)
    summary["actual_vs_optimal"] = summarize_actual(actual)
    summary["output_generation_seconds"] = time.perf_counter() - started
    if fingerprint != baseline_fingerprint(checkpoint):
        raise AssertionError("saved majority checkpoint or metrics changed during analysis")
    (output / "spectrum_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[spectrum] wrote {len(records)} spectra to {output}", flush=True)


if __name__ == "__main__":
    main()
