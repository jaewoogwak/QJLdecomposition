#!/usr/bin/env python3
"""Compare exact residual spectra for exponent Base and three QJL Bases.

The original BF16 safetensors and an existing ResCode grouping cache are the
only model inputs. No compressed checkpoint is created or modified.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rescode.qjl.exponent_mantissa_base import (
    PackedExponentMantissaBase, encode_exponent_mantissa_base,
    fit_row_beta, reconstruct_base_from_weight)
from rescode.qjl.core import ProjectionBank, pack_residual, unpack_signs
from rescode.qjl.bundled_model import dense_qjl_base, majority_signs
from scripts.QJL.analyze_bundle_residual_svd import weight_tensor
from scripts.QJL.analyze_residual_spectrum import MODEL, GROUPS, PROJECTIONS, select_layers, spectrum_statistics

CASES = ("A", "B", "C", "D")
CASE_NAMES = {"A": "Exponent Base only", "B": "Original Individual QJL",
              "C": "Exponent Individual QJL", "D": "Exponent Majority QJL"}
RANKS = (128, 512)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path(MODEL))
    parser.add_argument("--groups-path", type=Path, default=Path(GROUPS))
    parser.add_argument("--layers", default="12")
    parser.add_argument("--max-experts", type=int, default=0)
    parser.add_argument("--qjl-d", type=int, default=2048)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/qjl_exponent_bundle_ablation"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.max_experts < 0 or args.qjl_d < 1:
        parser.error("max-experts must be nonnegative and qjl-d positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_config(args, snapshot, groups_path, layers, weight_map):
    shards = sorted(set(weight_map.values()))
    return {"analysis_version": 2,
            "model_path": str(snapshot), "model_index_sha256": sha256(
                snapshot / "model.safetensors.index.json"),
            "model_shards": {name: [int((snapshot / name).stat().st_size),
                                    int((snapshot / name).stat().st_mtime_ns)] for name in shards},
            "groups_path": str(groups_path), "groups_sha256": sha256(groups_path),
            "layers": layers, "max_experts": args.max_experts, "qjl_d": args.qjl_d,
            "projection_seed": args.projection_seed, "random_seed": args.random_seed,
            "projections": list(PROJECTIONS), "ranks": list(RANKS), "device": args.device,
            "projection_dtype": "FP16 Gaussian", "svd": "torch.linalg.svdvals FP32; TF32 off",
            "case_names": CASE_NAMES}


def prepare_output(output, config):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError(f"existing output has different model, groups, or settings: {output}")
    if not manifest.exists() and (output / "layers").exists() and any((output / "layers").glob("*.npz")):
        raise ValueError("layer cache exists without configuration manifest")
    manifest.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    (output / "layers").mkdir(exist_ok=True)


def exponent_base(weight):
    """Match the existing k=0 encoder's FP16 beta storage convention."""
    beta = fit_row_beta(weight).to(torch.float16)
    if beta.dtype != torch.float16 or not bool(torch.isfinite(beta).all()):
        raise ValueError("invalid stored FP16 row beta")
    base = reconstruct_base_from_weight(weight, beta)
    if base.shape != weight.shape or base.dtype != torch.float32:
        raise AssertionError("exponent base shape or dtype mismatch")
    if not bool(torch.isfinite(base).all()):
        raise ValueError("non-finite exponent Base")
    return base, beta


def sign_agreement(first, second, d):
    if first.shape != second.shape:
        raise ValueError("sketch shapes differ")
    return float((unpack_signs(first, d) == unpack_signs(second, d)).float().mean())


def pairwise_scores(packed_by_expert, d):
    """Each expert's mean same-row agreement with all other group members."""
    ids = sorted(packed_by_expert)
    if len(ids) < 2:
        raise ValueError("non-singleton group required")
    bits = {expert: unpack_signs(packed_by_expert[expert], d) for expert in ids}
    counts = sum((bits[expert].to(torch.int16) for expert in ids))
    scores = {expert: float(torch.where(bits[expert], counts - 1,
                        len(ids) - counts - 1).float().mean() / (len(ids) - 1))
              for expert in ids}
    return scores


def independent_pairwise_chance(packed_by_expert, d):
    probabilities = [float(unpack_signs(packed_by_expert[i], d).float().mean())
                     for i in sorted(packed_by_expert)]
    return float(np.mean([a * b + (1 - a) * (1 - b)
                          for j, a in enumerate(probabilities)
                          for b in probabilities[j + 1:]]))


def random_row_pairing(packed_by_expert, layer, group, projection, seed, d):
    """Deterministic control retaining every expert's bit balance."""
    projection_id = PROJECTIONS.index(projection)
    shuffled = {}
    for expert, packed in packed_by_expert.items():
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + layer * 1000003 + group * 10007 + projection_id * 1009 + expert)
        order = torch.randperm(packed.shape[0], generator=generator).to(packed.device)
        shuffled[expert] = packed[order]
    random_vote, agreement = majority_signs(shuffled, d, True)
    majority_each = {expert: sign_agreement(packed, random_vote, d)
                     for expert, packed in shuffled.items()}
    return float(agreement), pairwise_scores(shuffled, d), majority_each


def qjl_base(packed, norms, bank, input_dim):
    return torch.cat([part for _, part in dense_qjl_base(
        packed, norms, bank, input_dim)], dim=0)


def case_statistics(original, base, case, layer, group, expert, projection):
    weight = original.float()
    if base.shape != weight.shape or base.dtype != torch.float32:
        raise AssertionError("base must match original shape and be FP32")
    residual = weight - base
    weight_norm = float(torch.linalg.vector_norm(weight.double()))
    residual_norm = float(torch.linalg.vector_norm(residual.double()))
    identity = float(torch.linalg.vector_norm((base + residual - weight).double()) /
                     max(weight_norm, 1e-30))
    sigma = torch.linalg.svdvals(residual.float())
    energy_gap = float(abs(sigma.double().square().sum() -
                           residual.double().square().sum()) /
                       residual.double().square().sum().clamp_min(1e-30))
    if identity > 1e-6 or energy_gap > 1e-3:
        raise AssertionError(f"reconstruction/SVD energy failed: {layer}/{expert}/{projection}/{case}")
    singular = sigma.cpu().numpy().astype(np.float32)
    metrics, _, _ = spectrum_statistics(singular, weight_norm, residual_norm)
    selected = {key: metrics[key] for key in ("residual_norm_ratio", "r90", "r95", "r99",
                "energy_r128", "energy_r512", "optimal_weight_error_r128",
                "optimal_weight_error_r512")}
    if not all(np.isfinite(float(value)) for value in selected.values()):
        raise AssertionError("non-finite spectrum result")
    return {"layer": layer, "group": group, "expert": expert, "projection": projection,
            "case": case, "weight_frobenius": weight_norm,
            "residual_frobenius": residual_norm, **selected,
            "identity_relative_error": identity, "svd_energy_relative_gap": energy_gap}, singular


@torch.no_grad()
def analyze_layer(layer, groups, snapshot, weight_map, bank, device, max_experts, random_seed):
    layer_groups = groups[layer]
    if sorted(int(i) for members in layer_groups for i in members) != list(range(60)):
        raise AssertionError(f"layer {layer} groups do not partition 60 routed experts")
    spectra, records, sign_rows, group_rows = [], [], [], []
    for group_id, members in enumerate(layer_groups):
        if len(members) == 1:
            continue
        all_experts = sorted(int(i) for i in members)
        chosen = all_experts[:max_experts or None]
        for projection in PROJECTIONS:
            signed_base, signed_weight, unsigned_scale, base_norms, weight_norms = {}, {}, {}, {}, {}
            rows = cols = None
            encoder_check_max_abs = None
            # All group members vote, including those excluded by --max-experts.
            for expert in all_experts:
                weight = weight_tensor(snapshot, weight_map, layer, expert, projection).to(device)
                if rows is None:
                    rows, cols = weight.shape
                if weight.dtype != torch.bfloat16 or weight.shape != (rows, cols):
                    raise AssertionError("original BF16 group matrix mismatch")
                base, beta = exponent_base(weight)
                if expert == chosen[0]:
                    encoded = encode_exponent_mantissa_base(weight)
                    if not torch.equal(encoded["beta"], beta.cpu()):
                        raise AssertionError("encoder FP16 beta differs from fitted beta")
                    decoded = PackedExponentMantissaBase(encoded, device).decode()
                    encoder_check_max_abs = float((decoded - base).abs().max())
                    if not torch.allclose(decoded, base, rtol=1e-6, atol=1e-6):
                        raise AssertionError("stored Exponent Base decode differs")
                    del decoded, encoded
                signed_weight[expert], weight_norms[expert] = pack_residual(weight, bank)
                signed_base[expert], base_norms[expert] = pack_residual(base, bank)
                # The sign-free control is positive exponent scale times row beta.
                unsigned_scale[expert], _ = pack_residual(base.abs(), bank)
                if signed_weight[expert].shape != signed_base[expert].shape:
                    raise AssertionError("W/Base sketch shape mismatch")
                del weight, base, beta
            majority, base_pairwise = majority_signs(signed_base, bank.d, True)
            _, weight_pairwise = majority_signs(signed_weight, bank.d, True)
            _, unsigned_pairwise = majority_signs(unsigned_scale, bank.d, True)
            base_each = pairwise_scores(signed_base, bank.d)
            weight_each = pairwise_scores(signed_weight, bank.d)
            random_base, random_each, random_majority_each = random_row_pairing(
                signed_base, layer, group_id, projection, random_seed, bank.d)
            random_weight, _, _ = random_row_pairing(
                signed_weight, layer, group_id, projection, random_seed + 100000000,
                bank.d)
            if abs(np.mean(list(base_each.values())) - base_pairwise) > 1e-5:
                raise AssertionError("signed Base pairwise count mismatch")
            group_rows.append({"layer": layer, "group": group_id, "projection": projection,
                "group_size": len(all_experts), "signed_base_pairwise_agreement": base_pairwise,
                "weight_pairwise_agreement": weight_pairwise,
                "unsigned_exponent_beta_pairwise_agreement": unsigned_pairwise,
                "random_row_base_pairwise_agreement": random_base,
                "random_row_weight_pairwise_agreement": random_weight,
                "random_row_base_vs_majority_agreement": float(np.mean(
                    list(random_majority_each.values()))),
                "independent_base_chance_agreement": independent_pairwise_chance(signed_base, bank.d),
                "independent_weight_chance_agreement": independent_pairwise_chance(signed_weight, bank.d),
                "encoder_decode_max_abs": encoder_check_max_abs})
            for expert in chosen:
                weight = weight_tensor(snapshot, weight_map, layer, expert, projection).to(device)
                base_a, beta = exponent_base(weight)
                if not torch.equal(base_norms[expert], torch.linalg.vector_norm(base_a, dim=1)):
                    raise AssertionError("C/D row norms changed after reconstruction")
                if not torch.equal(weight_norms[expert], torch.linalg.vector_norm(weight.float(), dim=1)):
                    raise AssertionError("B row norms changed")
                sign_rows.append({"layer": layer, "group": group_id, "expert": expert,
                    "projection": projection, "group_size": len(all_experts),
                    "weight_vs_signed_base_agreement": sign_agreement(
                        signed_weight[expert], signed_base[expert], bank.d),
                    "signed_base_pairwise_agreement": base_each[expert],
                    "weight_pairwise_agreement": weight_each[expert],
                    "signed_base_random_row_agreement": random_each[expert],
                    "signed_base_random_row_vs_majority_agreement": random_majority_each[expert],
                    "signed_base_vs_majority_agreement": sign_agreement(
                        signed_base[expert], majority, bank.d)})
                cases = {"A": base_a,
                         "B": qjl_base(signed_weight[expert], weight_norms[expert], bank, cols),
                         "C": qjl_base(signed_base[expert], base_norms[expert], bank, cols),
                         "D": qjl_base(majority, base_norms[expert], bank, cols)}
                for case in CASES:
                    row, sigma = case_statistics(weight, cases[case], case,
                                                 layer, group_id, expert, projection)
                    records.append(row)
                    spectra.append(sigma)
                del weight, base_a, beta, cases
            del majority, signed_base, signed_weight, unsigned_scale, base_norms, weight_norms
    return np.stack(spectra), records, sign_rows, group_rows


def save_layer(path, spectra, records, sign_rows, group_rows):
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, singular_values=spectra,
        records_json=json.dumps(records, sort_keys=True),
        sign_rows_json=json.dumps(sign_rows, sort_keys=True),
        group_rows_json=json.dumps(group_rows, sort_keys=True))
    os.replace(tmp, path)


def load_layers(output, layers):
    spectra, records, sign_rows, group_rows = [], [], [], []
    for layer in layers:
        with np.load(output / "layers" / f"layer_{layer:02d}.npz") as item:
            spectra.append(item["singular_values"])
            records.extend(json.loads(str(item["records_json"])))
            sign_rows.extend(json.loads(str(item["sign_rows_json"])))
            group_rows.extend(json.loads(str(item["group_rows_json"])))
    return np.concatenate(spectra), records, sign_rows, group_rows


def write_csv(path, rows):
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_statistics(records, sign_rows, group_rows, config, crosscheck):
    metrics = ("residual_norm_ratio", "energy_r128", "energy_r512",
               "optimal_weight_error_r128", "optimal_weight_error_r512", "r90", "r95", "r99")
    per_case = {}
    for projection in (*PROJECTIONS, "all"):
        per_case[projection] = {}
        for case in CASES:
            chosen = [row for row in records if row["case"] == case and
                      (projection == "all" or row["projection"] == projection)]
            if not chosen:
                continue
            per_case[projection][case] = {"count": len(chosen)}
            for metric in metrics:
                values = np.array([row[metric] for row in chosen], dtype=float)
                per_case[projection][case][metric] = {"mean": float(values.mean()),
                    "median": float(np.median(values)), "p90": float(np.percentile(values, 90))}
    paired = {}
    by_key = {(r["layer"], r["group"], r["expert"], r["projection"], r["case"]): r
              for r in records}
    if len(by_key) != len(records):
        raise AssertionError("duplicate expert/projection/case result")
    for projection in (*PROJECTIONS, "all"):
        paired[projection] = {}
        keys = sorted({key[:4] for key in by_key
                       if projection == "all" or key[3] == projection})
        for left, right in (("A", "B"), ("B", "C"), ("A", "C"), ("C", "D")):
            result = {"count": len(keys)}
            for metric in metrics:
                changes = np.array([by_key[(*key, right)][metric] -
                                    by_key[(*key, left)][metric] for key in keys], dtype=float)
                favorable = changes > 0 if metric.startswith("energy_") else changes < 0
                result[metric] = {"right_minus_left_mean": float(changes.mean()),
                    "right_minus_left_median": float(np.median(changes)),
                    "fraction_right_improved": float(favorable.mean())}
            paired[projection][f"{left}_to_{right}"] = result
    sign_summary = {}
    for projection in (*PROJECTIONS, "all"):
        sign_summary[projection] = {}
        for field in ("weight_vs_signed_base_agreement", "signed_base_pairwise_agreement",
                      "weight_pairwise_agreement", "signed_base_random_row_agreement",
                      "signed_base_random_row_vs_majority_agreement",
                      "signed_base_vs_majority_agreement"):
            values = [row[field] for row in sign_rows
                      if projection == "all" or row["projection"] == projection]
            sign_summary[projection][field] = {"mean": float(np.mean(values)),
                "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90))}
        for field in ("signed_base_pairwise_agreement", "weight_pairwise_agreement",
                      "unsigned_exponent_beta_pairwise_agreement",
                      "random_row_base_pairwise_agreement", "independent_base_chance_agreement"):
            values = [row[field] for row in group_rows
                      if projection == "all" or row["projection"] == projection]
            sign_summary[projection][f"group_{field}"] = {"mean": float(np.mean(values)),
                "median": float(np.median(values))}
        values = [row["random_row_base_vs_majority_agreement"] for row in group_rows
                  if projection == "all" or row["projection"] == projection]
        sign_summary[projection]["group_random_row_base_vs_majority_agreement"] = {
            "mean": float(np.mean(values)), "median": float(np.median(values))}
    return {"config": config, "matrix_count": len(records),
            "expert_projection_count": len(sign_rows), "group_projection_count": len(group_rows),
            "per_case": per_case, "paired_comparison": paired, "sign_agreement": sign_summary,
            "case_b_existing_spectrum_crosscheck": crosscheck,
            "max_identity_relative_error": max(r["identity_relative_error"] for r in records),
            "max_svd_energy_relative_gap": max(r["svd_energy_relative_gap"] for r in records)}


def crosscheck_case_b(records, prior_path, qjl_d, projection_seed):
    if qjl_d != 2048 or projection_seed != 0 or not prior_path.is_file():
        return {"status": "not_applicable_or_missing", "count": 0}
    with prior_path.open(newline="") as source:
        prior = {(int(row["layer"]), int(row["expert"]), row["projection"]): row
                 for row in csv.DictReader(source) if row["bundle_mode"] == "none"}
    differences = {key: [] for key in ("residual_norm_ratio", "energy_r128", "energy_r512",
                 "optimal_weight_error_r128", "optimal_weight_error_r512", "r90", "r95", "r99")}
    count = 0
    for record in records:
        if record["case"] != "B":
            continue
        old = prior.get((record["layer"], record["expert"], record["projection"]))
        if old is None:
            continue
        count += 1
        for key in differences:
            differences[key].append(abs(float(record[key]) - float(old[key])))
    if count:
        maximum = {key: max(values) for key, values in differences.items()}
        if max(maximum.values()) > 1e-4:
            raise AssertionError(f"Case B differs from prior individual QJL spectrum: {maximum}")
        return {"status": "matched", "count": count, "max_absolute_difference": maximum,
                "prior_csv_sha256": sha256(prior_path)}
    return {"status": "no_common_experts", "count": 0}


def save_figure(fig, output, stem):
    directory = output / "plots"
    directory.mkdir(exist_ok=True)
    fig.savefig(directory / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight",
                metadata={"Creator": "QJL exponent bundle ablation", "CreationDate": None})
    plt.close(fig)


def make_plots(output, spectra, records, sign_rows, group_rows, config):
    ranks = np.arange(1, spectra.shape[1] + 1)
    norms = np.square(spectra.astype(np.float64))
    cumulative = np.cumsum(norms / norms.sum(axis=1, keepdims=True), axis=1)
    normalized_spectrum = spectra / np.maximum(spectra[:, :1], 1e-30)
    conditions = f"D={config['qjl_d']}, layers={','.join(map(str, config['layers']))}, FP16 G"
    labels = [f"{case}: {CASE_NAMES[case]}" for case in CASES]
    for projection in PROJECTIONS:
        selected = np.array([r["projection"] == projection for r in records])
        case_mask = {case: selected & np.array([r["case"] == case for r in records])
                     for case in CASES}
        for metric, ylabel, stem in (("residual_norm_ratio", "||R||F / ||W||F", "residual_ratio"),
                                    ("optimal_weight_error_r128", "Optimal rank-128 weight error", "optimal_error_r128"),
                                    ("optimal_weight_error_r512", "Optimal rank-512 weight error", "optimal_error_r512")):
            fig, ax = plt.subplots(figsize=(10, 5))
            values = [[r[metric] for r in records if r["projection"] == projection and r["case"] == case]
                      for case in CASES]
            ax.boxplot(values, tick_labels=list(CASES), showfliers=False)
            ax.set(xlabel="Reconstruction case", ylabel=ylabel,
                   title=f"{projection.capitalize()}: {stem.replace('_', ' ')}; {conditions}")
            ax.grid(axis="y", alpha=.25)
            save_figure(fig, output, f"{stem}_{projection}")
        fig, ax = plt.subplots(figsize=(9, 5))
        for case, label in zip(CASES, labels):
            ax.plot(ranks, cumulative[case_mask[case]].mean(axis=0), label=label)
        for rank in RANKS:
            ax.axvline(rank, color="gray", linewidth=.7, linestyle=":")
        ax.set(xlabel="Residual SVD rank", ylabel="Mean cumulative residual energy retained",
               ylim=(0, 1.01), title=f"{projection.capitalize()}: full FP32 SVD; {conditions}")
        ax.legend(fontsize=8)
        save_figure(fig, output, f"cumulative_energy_{projection}")
        fig, ax = plt.subplots(figsize=(9, 5))
        for case, label in zip(CASES, labels):
            ax.plot(ranks, np.maximum(normalized_spectrum[case_mask[case]].mean(axis=0), 1e-8),
                    label=label)
        ax.set(xlabel="Singular-value index", ylabel="Mean sigma_k / sigma_1",
               yscale="log", title=f"{projection.capitalize()}: normalized spectra; {conditions}")
        ax.legend(fontsize=8)
        save_figure(fig, output, f"normalized_spectrum_{projection}")
        chosen = [r for r in sign_rows if r["projection"] == projection]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist([r["weight_vs_signed_base_agreement"] * 100 for r in chosen], bins=20,
                color="#4c78a8", edgecolor="white")
        ax.set(xlabel="sign(WG) versus sign(BG) agreement (%)", ylabel="Experts",
               title=f"{projection.capitalize()}: same-expert signed sketches; {conditions}")
        save_figure(fig, output, f"weight_vs_base_sign_agreement_{projection}")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = [r["signed_base_pairwise_agreement"] * 100 for r in chosen]
        y = [r["signed_base_random_row_agreement"] * 100 for r in chosen]
        ax.scatter(x, y, s=18, alpha=.65)
        ax.plot([0, 100], [0, 100], "k--", linewidth=1)
        ax.set(xlabel="Aligned-row signed Base pairwise agreement (%)",
               ylabel="Random-row pairing agreement (%)",
               title=f"{projection.capitalize()}: within-group control; {conditions}")
        save_figure(fig, output, f"pairwise_signed_vs_random_{projection}")
        fig, ax = plt.subplots(figsize=(8, 5))
        x = [r["signed_base_random_row_vs_majority_agreement"] * 100 for r in chosen]
        y = [r["signed_base_vs_majority_agreement"] * 100 for r in chosen]
        ax.scatter(x, y, s=18, alpha=.65)
        ax.plot([0, 100], [0, 100], "k--", linewidth=1)
        ax.set(xlabel="Random-row individual versus majority agreement (%)",
               ylabel="Aligned-row individual versus majority agreement (%)",
               title=f"{projection.capitalize()}: signed sketch sharing vs control; {conditions}")
        save_figure(fig, output, f"individual_vs_majority_sign_{projection}")


def write_report(output, summary):
    case = summary["per_case"]["all"]
    signs = summary["sign_agreement"]["all"]
    paired = summary["paired_comparison"]["all"]
    mean = lambda name, field: case[name][field]["mean"]
    rows = [
        "# Exponent Base and QJL sketch ablation",
        "",
        "A: existing exponent-mantissa k=0 Base (stored FP16 row beta), no QJL. "
        "B: QJL of original BF16 W. C: QJL of the signed Exponent Base, with its "
        "own row norms. D: group majority vote of C's sketches, retaining each "
        "expert's Exponent Base row norms. Every residual is original W minus "
        "the case's effective Base; no model inference was performed.",
        "",
        f"Layers: {summary['config']['layers']}; D={summary['config']['qjl_d']}; "
        f"FP16 Gaussian seed {summary['config']['projection_seed']}; "
        f"non-singleton expert/projection pairs: {summary['expert_projection_count']}. "
        "All original weights are BF16. Full FP32 SVD uses TF32 disabled.",
        "",
        "| Case | Mean residual rho | Mean E128 | Mean E512 | Mean optimal error128 | "
        "Mean optimal error512 | Mean r90 | Mean r95 | Mean r99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CASES:
        values = [mean(name, field) for field in ("residual_norm_ratio", "energy_r128",
                  "energy_r512", "optimal_weight_error_r128", "optimal_weight_error_r512",
                  "r90", "r95", "r99")]
        rows.append("| " + name + " | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    rows += ["", "## Paired changes", "",
             "Changes below are right minus left for the same expert and projection. "
             "A smaller rho/error/rank or larger retained energy is favorable.", "",
             "| Comparison | Mean Δrho | Fraction with smaller rho | Mean ΔE128 | Mean ΔE512 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for comparison in ("A_to_B", "B_to_C", "A_to_C", "C_to_D"):
        item = paired[comparison]
        norm = item["residual_norm_ratio"]
        rows.append(f"| {comparison} | {norm['right_minus_left_mean']:+.4f} | "
                    f"{norm['fraction_right_improved']:.1%} | "
                    f"{item['energy_r128']['right_minus_left_mean']:+.4f} | "
                    f"{item['energy_r512']['right_minus_left_mean']:+.4f} |")
    rows += ["", "## Signed binary sketch agreement", "",
             "These are `sign(BG)` sketches of the signed Exponent Base. The "
             "unsigned control uses `abs(B)` and is reported separately; it "
             "must not be interpreted as signed-sketch agreement.", "",
             "| Metric | Mean agreement |",
             "| --- | ---: |"]
    for field, label in (("weight_vs_signed_base_agreement", "Same expert sign(WG) vs sign(BG)"),
                         ("group_signed_base_pairwise_agreement", "Signed Base expert pairs"),
                         ("group_weight_pairwise_agreement", "Original W expert pairs"),
                         ("group_unsigned_exponent_beta_pairwise_agreement", "Unsigned abs(Base) expert pairs"),
                         ("group_random_row_base_pairwise_agreement", "Random neuron pairing of signed Base"),
                         ("group_random_row_base_vs_majority_agreement", "Random-row Base vs random majority"),
                         ("group_independent_base_chance_agreement", "Independent sign chance"),
                         ("signed_base_vs_majority_agreement", "Individual signed Base vs majority")):
        rows.append(f"| {label} | {100 * signs[field]['mean']:.2f}% |")
    rows += ["", "## Projection-specific means", "",
             "| Projection | A rho | B rho | C rho | D rho | Signed Base pairs | Random-row pairs |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for projection in PROJECTIONS:
        part = summary["per_case"][projection]
        sign_part = summary["sign_agreement"][projection]
        rows.append("| " + projection.capitalize() + " | " + " | ".join(
            f"{part[name]['residual_norm_ratio']['mean']:.4f}" for name in CASES) +
            " | " + f"{100 * sign_part['group_signed_base_pairwise_agreement']['mean']:.2f}%" +
            " | " + f"{100 * sign_part['group_random_row_base_pairwise_agreement']['mean']:.2f}%" + " |")
    rows += ["", "## Questions answered by this diagnostic", "",
             f"Q1: A retains mean residual rho {mean('A', 'residual_norm_ratio'):.4f}.",
             f"Q2: C changes rho from B by {paired['B_to_C']['residual_norm_ratio']['right_minus_left_mean']:+.4f} "
             f"on average, improving {paired['B_to_C']['residual_norm_ratio']['fraction_right_improved']:.1%} of paired matrices.",
             f"Q3: QJL of the exponent Base raises rho over A by "
             f"{paired['A_to_C']['residual_norm_ratio']['right_minus_left_mean']:+.4f}.",
             f"Q4: Majority voting raises rho over C by "
             f"{paired['C_to_D']['residual_norm_ratio']['right_minus_left_mean']:+.4f}.",
             f"Q5: C minus B mean retained energy is "
             f"{paired['B_to_C']['energy_r128']['right_minus_left_mean']:+.4f} at rank 128 and "
             f"{paired['B_to_C']['energy_r512']['right_minus_left_mean']:+.4f} at rank 512. "
             f"D minus C is {paired['C_to_D']['energy_r128']['right_minus_left_mean']:+.4f} "
             f"and {paired['C_to_D']['energy_r512']['right_minus_left_mean']:+.4f}.",
             f"Q6: Signed Base pairwise agreement is "
             f"{100 * signs['group_signed_base_pairwise_agreement']['mean']:.3f}%, "
             f"versus {100 * signs['group_random_row_base_pairwise_agreement']['mean']:.3f}% "
             f"under random row pairing. Actual individual-vs-majority agreement is "
             f"{100 * signs['signed_base_vs_majority_agreement']['mean']:.3f}%, "
             f"versus {100 * signs['group_random_row_base_vs_majority_agreement']['mean']:.3f}% "
             f"under the random-row control. The unsigned abs(Base) value of "
             f"{100 * signs['group_unsigned_exponent_beta_pairwise_agreement']['mean']:.3f}% "
             "does not indicate signed sketch sharing.", ""]
    rows += ["", "## Validation and limits", "",
             f"Maximum `W - (Base + Residual)` relative error: "
             f"{summary['max_identity_relative_error']:.3g}. Maximum FP32 SVD energy "
             f"relative gap: {summary['max_svd_energy_relative_gap']:.3g}. "
             f"Existing Case B spectrum cross-check: "
             f"{summary['case_b_existing_spectrum_crosscheck']['status']} "
             f"({summary['case_b_existing_spectrum_crosscheck']['count']} matrices).",
             "",
             "The random-row control permutes neuron rows independently for each "
             "expert while preserving its sketch bits and bit balance. The "
             "independent-sign chance baseline uses each expert's observed 1-bit "
             "fraction. Neither control is a compressed-model accuracy result.",
             "",
             "Optimal rank errors are full-SVD lower bounds. This experiment does "
             "not store low-rank factors, create a compressed checkpoint, or run ARC "
             "evaluation. A favorable residual spectrum does not by itself prove "
             "end-to-end inference gains.",
             "",
             "Raw full singular values, expert IDs, cases, per-group sign controls, "
             "and per-expert metrics are in `layers/layer_XX.npz`, "
             "`group_sign_stats.csv`, `sign_agreement.csv`, and `per_expert.csv`.", ""]
    (output / "REPORT.md").write_text("\n".join(rows))


def main():
    args = arguments()
    os.chdir(ROOT)
    snapshot = args.model_path.resolve()
    groups_path = args.groups_path.resolve()
    output = args.output_dir.resolve()
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file() or not groups_path.is_file():
        raise FileNotFoundError("original BF16 model or saved ResCode grouping cache missing")
    index = json.loads(index_path.read_text())["weight_map"]
    groups = torch.load(groups_path, map_location="cpu", weights_only=False)["groups"]
    layers = select_layers(args.layers, groups)
    config = input_config(args, snapshot, groups_path, layers, index)
    prepare_output(output, config)
    bank = ProjectionBank(args.qjl_d, args.projection_seed)
    device = torch.device(args.device)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for layer in layers:
            path = output / "layers" / f"layer_{layer:02d}.npz"
            if path.exists():
                print(f"[exponent-bundle] reused layer={layer}", flush=True)
                continue
            spectra, records, sign_rows, group_rows = analyze_layer(
                layer, groups, snapshot, index, bank, device,
                args.max_experts, args.random_seed)
            save_layer(path, spectra, records, sign_rows, group_rows)
            print(f"[exponent-bundle] layer={layer} matrices={len(records)} "
                  f"group-projections={len(group_rows)}", flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    spectra, records, sign_rows, group_rows = load_layers(output, layers)
    if len(records) != len(spectra) or len(records) != 4 * len(sign_rows):
        raise AssertionError("A-D results do not align with selected expert/projections")
    write_csv(output / "per_expert.csv", records)
    write_csv(output / "sign_agreement.csv", sign_rows)
    write_csv(output / "group_sign_stats.csv", group_rows)
    prior = ROOT / "results/qjl_bundle/spectrum_comparison/spectrum_metrics.csv"
    crosscheck = crosscheck_case_b(records, prior, args.qjl_d, args.projection_seed)
    summary = summary_statistics(records, sign_rows, group_rows, config, crosscheck)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_plots(output, spectra, records, sign_rows, group_rows, config)
    write_report(output, summary)
    print(f"[exponent-bundle] finished: {len(records)} matrices at {output}", flush=True)


if __name__ == "__main__":
    main()
