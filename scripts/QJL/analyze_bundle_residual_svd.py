#!/usr/bin/env python3
"""Exact singular spectra of W - the stored majority-QJL effective base.

The low-rank factors in the checkpoint are deliberately not used to construct
the analyzed residual. Original BF16 weights come from local safetensors.
"""

import argparse
import base64
import csv
import html
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from safetensors import safe_open

from rescode.qjl.core import ProjectionBank, unpack_signs


DEFAULT_MODEL = ("/workspace/.hf_home/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
                 "/snapshots/ec052fda178e241c7c443468d2fa1db6618996be")
DEFAULT_CHECKPOINT = "results/qjl_bundle/full_majority_d2048_r128/bundle_qjl_state.pt"
PROJECTIONS = ("gate", "up", "down")
PALETTE = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140),
                    (94, 201, 97), (253, 231, 37)], dtype=np.float32)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/qjl_bundle/full_majority_d2048_r128/residual_svd"))
    parser.add_argument("--layers", default="all", help="all or comma-separated layer indices")
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--max-experts", type=int, default=0,
                        help="0 means all non-singleton experts; useful for a smoke test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_experts < 0:
        parser.error("max-experts must be nonnegative")
    args.projections = tuple(x.strip() for x in args.projections.split(","))
    if not args.projections or len(set(args.projections)) != len(args.projections) or any(
            x not in PROJECTIONS for x in args.projections):
        parser.error("projections must be a unique comma-separated subset of gate,up,down")
    return args


def select_layers(spec, available):
    layers = sorted(available) if spec == "all" else sorted({int(x) for x in spec.split(",")})
    if not layers or any(x not in available for x in layers):
        raise ValueError(f"invalid layers: {spec}")
    return layers


def png_bytes(rgb):
    """Write an RGB PNG using the standard library; no plotting dependency."""
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("RGB image required")
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(height))
    def chunk(kind, payload):
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    return (b"\x89PNG\r\n\x1a\n" +
            chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def colorize(log_values, vmin, vmax):
    scaled = np.clip((log_values - vmin) / (vmax - vmin), 0, 1)
    indices = scaled * (len(PALETTE) - 1)
    left = np.floor(indices).astype(np.int32)
    right = np.minimum(left + 1, len(PALETTE) - 1)
    fraction = (indices - left)[..., None]
    return ((1 - fraction) * PALETTE[left] + fraction * PALETTE[right]).astype(np.uint8)


def save_heatmap(figures, name, values, layers, vmin, vmax, row_scale=1):
    log_values = np.log10(np.maximum(values, 1e-30))
    rgb = colorize(log_values, vmin, vmax)
    if row_scale > 1:
        rgb = np.repeat(rgb, row_scale, axis=0)
    png = png_bytes(rgb)
    png_path = figures / f"{name}.png"
    png_path.write_bytes(png)
    image_width, image_height = rgb.shape[1], rgb.shape[0]
    ticks = [(1, 0), (128, 127), (256, 255), (512, 511), (1024, 1023),
             (1408, 1407)]
    ticks = [(label, col) for label, col in ticks if col < image_width]
    x0, y0 = 120, 65
    width, height = x0 + image_width + 110, y0 + image_height + 95
    layer_positions = {}
    for row, layer in enumerate(layers):
        layer_positions.setdefault(int(layer), []).append(row)
    line = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{x0}" y="30" font-size="20" font-family="sans-serif">'
            f'{html.escape(name.replace("_", " "))}: log10 singular value</text>',
            f'<image x="{x0}" y="{y0}" width="{image_width}" height="{image_height}" '
            f'href="data:image/png;base64,{base64.b64encode(png).decode()}"/>',
            f'<text x="{x0 + image_width // 2}" y="{y0 + image_height + 55}" '
            'font-size="15" text-anchor="middle" font-family="sans-serif">'
            'Singular-value index (descending)</text>',
            f'<text x="20" y="{y0 + image_height // 2}" font-size="15" '
            f'transform="rotate(-90 20 {y0 + image_height // 2})" '
            'text-anchor="middle" font-family="sans-serif">Layer / expert rows</text>']
    for label, col in ticks:
        x = x0 + col
        line.append(f'<line x1="{x}" x2="{x}" y1="{y0 + image_height}" '
                    f'y2="{y0 + image_height + 6}" stroke="black"/>')
        line.append(f'<text x="{x}" y="{y0 + image_height + 24}" '
                    f'font-size="12" text-anchor="middle" font-family="sans-serif">{label}</text>')
    for layer, positions in layer_positions.items():
        center = y0 + (min(positions) + max(positions) + 1) * row_scale / 2
        line.append(f'<text x="{x0 - 12}" y="{center + 4}" font-size="11" '
                    f'text-anchor="end" font-family="sans-serif">{layer}</text>')
    for fraction in (0, 0.5, 1):
        y = y0 + image_height * (1 - fraction)
        color = colorize(np.array([[vmin + fraction * (vmax - vmin)]]), vmin, vmax)[0, 0]
        line.append(f'<rect x="{x0 + image_width + 20}" y="{y - 8}" width="16" '
                    f'height="16" fill="rgb({color[0]},{color[1]},{color[2]})"/>')
        line.append(f'<text x="{x0 + image_width + 43}" y="{y + 4}" '
                    f'font-size="11" font-family="sans-serif">{vmin + fraction * (vmax - vmin):.2f}</text>')
    line.append('</svg>')
    (figures / f"{name}.svg").write_text("\n".join(line))


def weight_tensor(snapshot, weight_map, layer, expert, projection):
    key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"original BF16 weight missing: {key}")
    with safe_open(str(snapshot / shard), framework="pt", device="cpu") as source:
        weight = source.get_tensor(key)
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError(f"expected BF16 matrix for {key}, got {weight.dtype} {weight.shape}")
    return weight


def analyze_layer(layer, state, bank, snapshot, weight_map, projections,
                  device, max_experts):
    spectra, metadata, summary = [], [], []
    clusters = state["layers"][layer]["clusters"]
    for group_id, group in sorted(clusters.items(), key=lambda item: int(item[0])):
        experts = sorted(int(x) for x in group["experts"])
        if max_experts:
            experts = experts[:max_experts]
        for projection in projections:
            packed = group["sketches"][projection]["shared"].to(device)
            first = weight_tensor(snapshot, weight_map, layer, experts[0], projection)
            rows, cols = first.shape
            p = bank.get(cols, device).float()
            signs = unpack_signs(packed, bank.d).to(torch.float32).mul_(2).sub_(1)
            if signs.shape != (rows, bank.d):
                raise ValueError(f"sketch shape mismatch at {layer}/{group_id}/{projection}")
            shared_basis = (signs @ p.T) * (math.sqrt(math.pi / 2) / bank.d)
            del signs
            for expert in experts:
                weight = first if expert == experts[0] else weight_tensor(
                    snapshot, weight_map, layer, expert, projection)
                weight = weight.to(device)
                if weight.shape != shared_basis.shape:
                    raise ValueError(f"weight shape mismatch at {layer}/{expert}/{projection}")
                norms = group["experts"][expert][projection]["norms"].to(device)
                if norms.dtype != torch.float32 or norms.shape != (rows,):
                    raise ValueError(f"row-norm mismatch at {layer}/{expert}/{projection}")
                base = shared_basis * norms[:, None]
                residual = weight.float() - base
                singular = torch.linalg.svdvals(residual)
                squared = singular.double().square().sum()
                residual_squared = residual.double().square().sum()
                squared_value = float(squared)
                residual_squared_value = float(residual_squared)
                energy_gap = float(abs(squared - residual_squared) /
                                   residual_squared.clamp_min(1e-30))
                reconstruction = float(torch.linalg.vector_norm(base + residual - weight.float()) /
                                       torch.linalg.vector_norm(weight.float()).clamp_min(1e-30))
                if energy_gap > 1e-3 or reconstruction > 1e-6:
                    raise AssertionError(f"SVD/reconstruction check failed at "
                                         f"{layer}/{expert}/{projection}: "
                                         f"{energy_gap=} {reconstruction=}")
                s = singular.cpu().numpy().astype(np.float32)
                spectra.append(s)
                metadata.append((layer, int(group_id), expert, projection))
                row = {"layer": layer, "group": int(group_id), "expert": expert,
                       "projection": projection, "rows": rows, "cols": cols,
                       "sv_max": float(s[0]), "sv_128": float(s[min(127, len(s)-1)]),
                       "sv_512": float(s[min(511, len(s)-1)]),
                       "sv_min": float(s[-1]),
                       "energy_top_128": float(np.square(s[:128].astype(np.float64)).sum() / squared_value),
                       "energy_top_512": float(np.square(s[:512].astype(np.float64)).sum() / squared_value),
                       "residual_frobenius": math.sqrt(residual_squared_value),
                       "svd_energy_relative_gap": energy_gap,
                       "reconstruction_relative_error": reconstruction}
                summary.append(row)
                del weight, base, residual, singular
            del first, shared_basis, packed, p
    return np.stack(spectra), metadata, summary


def write_outputs(output, layers, projections, config):
    files = [output / "layers" / f"layer_{layer:02d}.npz" for layer in layers]
    spectra, metadata, rows = [], [], []
    for path in files:
        with np.load(path) as saved:
            spectra.append(saved["values"])
            metadata.extend(zip(saved["layer"].tolist(), saved["group"].tolist(),
                                saved["expert"].tolist(), saved["projection"].tolist()))
            rows.extend(json.loads(str(saved["summary_json"])))
    values = np.concatenate(spectra)
    with (output / "per_expert.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "singular_values.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["layer", "group", "expert", "projection"] +
                        [f"sv_{i+1}" for i in range(values.shape[1])])
        for key, singular in zip(metadata, values):
            writer.writerow([*key, *(f"{float(x):.8g}" for x in singular)])
    with (output / "heatmap_row_index.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["projection", "heatmap_row_zero_based", "layer", "group", "expert"])
        for projection in projections:
            for row_index, item in enumerate(x for x in metadata if x[3] == projection):
                writer.writerow([projection, row_index, item[0], item[1], item[2]])
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    vmin = float(np.percentile(np.log10(np.maximum(values, 1e-30)), 1))
    vmax = float(np.percentile(np.log10(np.maximum(values, 1e-30)), 99))
    projection_summary = {}
    for projection in projections:
        mask = np.array([item[3] == projection for item in metadata])
        part = values[mask]
        part_layers = np.array([item[0] for item, selected in zip(metadata, mask) if selected])
        save_heatmap(figures, f"{projection}_expert_singular_values", part,
                     part_layers, vmin, vmax)
        unique_layers = sorted(set(part_layers))
        median = np.stack([np.median(part[part_layers == layer], axis=0)
                           for layer in unique_layers])
        save_heatmap(figures, f"{projection}_layer_median_singular_values",
                     median, unique_layers, vmin, vmax, row_scale=18)
        projection_summary[projection] = {
            "expert_matrices": int(mask.sum()),
            "median_sv_max": float(np.median(part[:, 0])),
            "median_sv_128": float(np.median(part[:, min(127, part.shape[1]-1)])),
            "median_sv_512": float(np.median(part[:, min(511, part.shape[1]-1)])),
            "median_energy_top_128": float(np.median([row["energy_top_128"] for row in rows
                                                      if row["projection"] == projection])),
            "median_energy_top_512": float(np.median([row["energy_top_512"] for row in rows
                                                      if row["projection"] == projection]))}
    report = {"config": config, "n_matrices": len(metadata), "n_singular_values": int(values.size),
              "global_log10_color_scale": [vmin, vmax], "projection_summary": projection_summary,
              "max_svd_energy_relative_gap": max(row["svd_energy_relative_gap"] for row in rows),
              "max_reconstruction_relative_error": max(row["reconstruction_relative_error"] for row in rows)}
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    (output / "README.md").write_text(
        "# Exact bundled-QJL residual singular values\n\n"
        "Each row in `singular_values.csv` is one non-singleton expert/projection. "
        "`sv_1` through `sv_1408` are descending FP32 singular values of "
        "`R_i = FP32(W_i) - FP32(B_tilde_gi)`, where `B_tilde_gi` is built from "
        "the checkpoint's packed majority signs, FP32 expert row norms, and "
        "runtime FP16 Gaussian projection. The low-rank checkpoint factors "
        "are not used to compute this residual. `layers/layer_XX.npz` contains "
        "the same full values and identity metadata. Heatmaps use a shared "
        "log10 color scale across Gate/Up/Down; rows are sorted by layer, "
        "group, expert. `heatmap_row_index.csv` maps each image row to an expert. "
        "Singular index is descending.\n")
    return report


def main():
    args = arguments()
    checkpoint = args.checkpoint.resolve()
    snapshot = args.model_path.resolve()
    output = args.output_dir.resolve()
    if not checkpoint.is_file() or not (snapshot / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("checkpoint or original BF16 model snapshot missing")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state.get("format") != "qjl-majority-lowrank-v1":
        raise ValueError("not a majority-bundled QJL checkpoint")
    config = state["config"]
    if config["bundle_mode"] != "majority" or config["residual_mode"] != "lowrank":
        raise ValueError("analysis requires majority + lowrank checkpoint")
    layers = select_layers(args.layers, state["layers"])
    bank = ProjectionBank(config["qjl_d"], config["projection_seed"])
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    (output / "layers").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    computed_layers = 0
    prior_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for layer in layers:
            layer_path = output / "layers" / f"layer_{layer:02d}.npz"
            if layer_path.exists():
                print(f"[svd] reused layer {layer}: {layer_path}", flush=True)
                continue
            tick = time.perf_counter()
            values, metadata, rows = analyze_layer(
                layer, state, bank, snapshot, weight_map, args.projections,
                device, args.max_experts)
            tmp = layer_path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, values=values,
                layer=np.array([x[0] for x in metadata], dtype=np.int16),
                group=np.array([x[1] for x in metadata], dtype=np.int16),
                expert=np.array([x[2] for x in metadata], dtype=np.int16),
                projection=np.array([x[3] for x in metadata]),
                summary_json=json.dumps(rows))
            os.replace(tmp, layer_path)
            computed_layers += 1
            print(f"[svd] layer={layer} matrices={len(rows)} seconds={time.perf_counter()-tick:.1f}",
                  flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prior_tf32
    report = write_outputs(output, layers, args.projections, {
        "checkpoint": str(checkpoint), "original_model": str(snapshot),
        "layers": layers, "projections": args.projections,
        "max_experts_per_group": args.max_experts, "qjl_d": config["qjl_d"],
        "rank_in_checkpoint": config["rank"], "projection_seed": config["projection_seed"],
        "computed_layers_this_run": computed_layers,
        "reused_layers_this_run": len(layers) - computed_layers,
        "layer_processing_seconds_this_run": time.perf_counter() - started})
    print(f"[svd] wrote {report['n_matrices']} matrices to {output}", flush=True)


if __name__ == "__main__":
    main()
