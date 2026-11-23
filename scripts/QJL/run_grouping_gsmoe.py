#!/usr/bin/env python3
"""GS-MoE gain grouping + shared projections + expert residual SVD for Qwen1.5-MoE.

This is an independent decomposition experiment. It uses no QJL kernels, pruning,
or training. The reconstruction-gain objective and fixed-size medoid assignment
follow GS-MoE; the checkpoint and evaluation interface follow this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.QJL.run_shared_lowrank_qjl import (
    DEFAULT_C4, DEFAULT_TASKS, MODEL_ID, PROJECTIONS, WeightReader,
    _load_hf_model, ensure_c4, selected, svd_factors,
)

FORMAT = "qwen15-gsmoe-shared-projection-residual-svd-v1"


def _sym(matrix: torch.Tensor) -> torch.Tensor:
    return (matrix + matrix.T) * 0.5


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"nonfinite {name}")


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def _norm2(x: torch.Tensor) -> float:
    return float(x.double().square().sum())


def validate_groups(groups: list[list[int]], experts: list[int], num_groups: int) -> None:
    if len(groups) != num_groups or any(not group for group in groups):
        raise ValueError("group count or empty group")
    if sorted(i for group in groups for i in group) != sorted(experts):
        raise ValueError("grouping has missing or duplicated experts")
    sizes = [len(group) for group in groups]
    if max(sizes) - min(sizes) > 1:
        raise ValueError("group sizes are not balanced")


def balanced_kmedoids(distance: np.ndarray, n_groups: int, seed: int,
                      restarts: int = 8, iterations: int = 25) -> list[list[int]]:
    """Fixed-size Hungarian assignment and medoid update from GS-MoE.

    The distance can be asymmetric: D[i,j] is the loss for expert i when using
    expert j's probe projection, as in GS-MoE's reconstruction gain.
    """
    e = distance.shape[0]
    if distance.shape != (e, e) or not np.isfinite(distance).all():
        raise ValueError("distance must be a finite square matrix")
    if not 1 <= n_groups <= e or restarts < 1 or iterations < 1:
        raise ValueError("invalid medoid settings")
    sizes = [e // n_groups + (j < e % n_groups) for j in range(n_groups)]
    seats = np.repeat(np.arange(n_groups), sizes)
    rng = np.random.default_rng(seed)
    best_cost, best_labels = math.inf, None
    for _ in range(restarts):
        medoids = np.empty(n_groups, dtype=np.int64)
        medoids[0] = rng.integers(e)
        for j in range(1, n_groups):
            d = np.square(np.clip(distance[:, medoids[:j]].min(axis=1), 0, None))
            d[medoids[:j]] = 0
            remaining = np.setdiff1d(np.arange(e), medoids[:j])
            medoids[j] = rng.choice(e, p=d / d.sum()) if d.sum() > 0 else rng.choice(remaining)
        for _ in range(iterations):
            cost = distance[:, medoids][:, seats]
            row, col = linear_sum_assignment(cost)
            labels = np.empty(e, dtype=np.int64)
            labels[row] = seats[col]
            updated = medoids.copy()
            for j in range(n_groups):
                members = np.flatnonzero(labels == j)
                updated[j] = members[distance[np.ix_(members, members)].sum(axis=0).argmin()]
            if np.array_equal(updated, medoids):
                break
            medoids = updated
        cost = distance[:, medoids][:, seats]
        row, col = linear_sum_assignment(cost)
        labels = np.empty(e, dtype=np.int64)
        labels[row] = seats[col]
        total = float(distance[np.arange(e), medoids[labels]].sum())
        if total < best_cost:
            best_cost, best_labels = total, labels.copy()
    groups = [np.flatnonzero(best_labels == j).tolist() for j in range(n_groups)]
    validate_groups(groups, list(range(e)), n_groups)
    return groups


def input_projection_gain(x: torch.Tensor, y: torch.Tensor, p: torch.Tensor,
                          ridge: float = 1e-6) -> torch.Tensor:
    """GS-MoE J=tr((PᵀCP)⁻¹ PᵀC WᵀW CP), using routed samples.

    x has rows of inputs and y=x Wᵀ. Both may be sample-weighted before calling.
    The ridge only stabilizes a singular sample Gram and is relative to its trace.
    """
    z = x @ p
    gram = _sym(z.T @ z)
    scale = gram.trace().clamp_min(1e-30) / max(1, p.shape[1])
    gram = gram + ridge * scale * torch.eye(p.shape[1], device=p.device, dtype=p.dtype)
    cross = z.T @ y
    return (cross * torch.linalg.solve(gram, cross)).sum() / max(1, x.shape[0])


def down_projection_gain(y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """GS-MoE down term tr(Pᵀ A P), A=YᵀY/N and PᵀP=I."""
    return (y @ p).square().sum() / max(1, y.shape[0])


def reconstruction_gain(x: torch.Tensor, outputs: dict[str, torch.Tensor],
                        projections: dict[str, torch.Tensor], frequency: float,
                        down_beta: float = 1.0, ridge: float = 1e-6) -> torch.Tensor:
    return frequency * (
        input_projection_gain(x, outputs["up"], projections["up"], ridge)
        + input_projection_gain(x, outputs["gate"], projections["gate"], ridge)
        + down_beta * down_projection_gain(outputs["down"], projections["down"]))


def affinity_from_gains(gains: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    if gains.ndim != 2 or gains.shape[0] != gains.shape[1]:
        raise ValueError("gain matrix must be square")
    _finite(gains, "reconstruction gains")
    own = gains.diagonal()
    if bool((own <= 0).any()):
        raise ValueError("nonpositive self reconstruction gain")
    affinity = (gains / own[:, None]).clamp(0, 1)
    distance = (1 - affinity).double().cpu().numpy()
    np.fill_diagonal(distance, 0)
    return distance, own.double().cpu().numpy()


def _cov(x: torch.Tensor) -> torch.Tensor:
    return _sym(x.T @ x / max(1, x.shape[0]))


def _ridge_cov(cov: torch.Tensor, ridge: float) -> torch.Tensor:
    d = cov.shape[0]
    scale = cov.trace().clamp_min(1e-20) / d
    return _sym(cov) + ridge * scale * torch.eye(d, device=cov.device, dtype=cov.dtype)


def shared_projection(cov_x: torch.Tensor, gram: torch.Tensor, rank: int,
                      direction: str, ridge: float) -> torch.Tensor:
    """GS-MoE find_shared_proj eigenproblem, adapted to Qwen1.5 dimensions.

    Gate/Up: top eigenvectors of sqrt(Cx) (Σ WᵀW) sqrt(Cx), mapped back by
    Cx^{-1/2} and QR. Down: top eigenvectors of Σ Y_downᵀY_down.
    """
    d = gram.shape[0]
    if not 0 < rank <= d:
        raise ValueError("projection rank exceeds hidden dimension")
    if direction == "down":
        _, vectors = torch.linalg.eigh(_sym(gram))
        p = vectors[:, -rank:].flip(-1).contiguous()
    elif direction in ("gate", "up"):
        eig, vectors = torch.linalg.eigh(_ridge_cov(cov_x, ridge))
        eig = eig.clamp_min(eig.max().clamp_min(1e-20) * 1e-8)
        sqrt = (vectors * eig.sqrt()) @ vectors.T
        invsqrt = (vectors * eig.rsqrt()) @ vectors.T
        _, basis = torch.linalg.eigh(_sym(sqrt @ gram @ sqrt))
        p, _ = torch.linalg.qr(invsqrt @ basis[:, -rank:].flip(-1), mode="reduced")
    else:
        raise ValueError(f"unknown direction: {direction}")
    _finite(p, "shared projection")
    return p.contiguous()


def group_projections(members: list[int], samples: dict, frequency: dict,
                      reader: WeightReader, layer: int, rank: int,
                      ridge: float, device: torch.device) -> dict[str, torch.Tensor]:
    """Refit P_g from pooled routed covariance and group weight/output Grams."""
    hidden = next(iter(samples.values()))["gate"].shape[1]
    cov_x = torch.zeros(hidden, hidden, device=device, dtype=torch.float32)
    grams = {key: torch.zeros_like(cov_x) for key in PROJECTIONS}
    total = sum(max(float(frequency[i]), 1.0) for i in members)
    for expert in members:
        weight = max(float(frequency[expert]), 1.0) / total
        x = samples[expert]["gate"].float().to(device)
        middle = samples[expert]["down"].float().to(device)
        cov_x.add_(_cov(x), alpha=weight)
        for key in ("gate", "up"):
            w = reader.get(layer, expert, key).float().to(device)
            grams[key].add_(w.T @ w, alpha=weight)
        w_down = reader.get(layer, expert, "down").float().to(device)
        y_down = middle @ w_down.T
        grams["down"].add_(_cov(y_down), alpha=weight)
    return {key: shared_projection(cov_x, grams[key], rank, key, ridge).cpu()
            for key in PROJECTIONS}


def calibration_coefficient(weight: torch.Tensor, p: torch.Tensor,
                            cov_x: torch.Tensor, direction: str,
                            ridge: float) -> torch.Tensor:
    """GS-MoE accumulate_expert_coefficients for input/output projections."""
    if direction == "down":
        return p.T @ weight
    c = _ridge_cov(cov_x, ridge)
    gram = _sym(p.T @ c @ p)
    gram = gram + ridge * gram.trace().clamp_min(1e-20) / p.shape[1] * torch.eye(
        p.shape[1], device=p.device, dtype=p.dtype)
    return torch.linalg.solve(gram, p.T @ c @ weight.T).T


def reconstruct_shared(coeff: torch.Tensor, p: torch.Tensor,
                       direction: str) -> torch.Tensor:
    return p @ coeff if direction == "down" else coeff @ p.T


def reconstruct_part(part: dict, p: torch.Tensor, direction: str,
                     device: torch.device) -> torch.Tensor:
    coeff = part["coeff"].to(device).float()
    p = p.to(device).float()
    w = reconstruct_shared(coeff, p, direction)
    if "u" in part:
        w = w + part["u"].to(device).float() @ part["v"].to(device).float()
    return w


def _layer_value(store: dict, layer: int):
    return store[layer] if layer in store else store[str(layer)]


def load_calibration_layer(path: Path, layer: int, experts: list[int]) -> tuple[dict, dict, str]:
    """Load this experiment's layer file or an existing QJL routed-sample cache."""
    if path.is_dir():
        entry = torch.load(path / f"layer_{layer}.pt", map_location="cpu", weights_only=False)
        raw = entry["samples"]
        frequency = {int(k): float(v) for k, v in entry["frequency"].items()}
        source = entry.get("frequency_source", "full_routing_count")
    else:
        entry = torch.load(path, map_location="cpu", weights_only=False)
        raw = _layer_value(entry.get("routed_samples", entry).get("train",
                            entry.get("routed_samples", entry)), layer)
        frequency, source = {}, "capped_sample_count_proxy"
    result = {}
    for i in experts:
        gate_entry = raw["gate"].get(i, raw["gate"].get(str(i)))
        down_entry = raw["down"].get(i, raw["down"].get(str(i)))
        if gate_entry is None or down_entry is None:
            raise ValueError(f"missing routed calibration samples for layer {layer} expert {i}")
        x = gate_entry[0] if isinstance(gate_entry, (tuple, list)) else gate_entry
        middle = down_entry[0] if isinstance(down_entry, (tuple, list)) else down_entry
        if x.ndim != 2 or middle.ndim != 2 or x.shape[0] < 2 or middle.shape[0] < 2:
            raise ValueError(f"insufficient routed samples for layer {layer} expert {i}")
        result[i] = {"gate": x.contiguous(), "up": x.contiguous(),
                     "down": middle.contiguous()}
        if source == "capped_sample_count_proxy":
            frequency[i] = float(x.shape[0])
        if frequency.get(i, 0) <= 0:
            raise ValueError(f"nonpositive routing frequency for layer {layer} expert {i}")
    return result, frequency, source


def collect_calibration(args, model_dir: Path, layers: list[int], out: Path) -> Path:
    """One model pass; hooks reuse QJL's routed input collector and count all routes."""
    from transformers import AutoTokenizer
    from rescode.evaluation import get_calib_dataloder
    from rescode.merging.calib_stats import iter_moe_blocks, route, routing_config
    from rescode.qjl.calibration import ProjectionSampleCollector

    calibration_data = ensure_c4(args.calibration_data)
    model = _load_hf_model(model_dir, args.gpu_memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataloader = get_calib_dataloder(
        dataset="c4", tokenizer=tokenizer, max_block_size=args.calibration_seqlen,
        n_blocks_for_stat=args.calibration_blocks, batch_size=args.calibration_batch_size,
        num_workers=0, seed=args.seed, data_path=str(calibration_data))
    collector = ProjectionSampleCollector("qwen", layers,
                                          max_train=args.calibration_samples,
                                          max_heldout=0)
    counts = defaultdict(Counter)
    top_k, norm = routing_config(model, "qwen")
    handles = []
    for layer_idx, block in iter_moe_blocks(model, "qwen"):
        if layer_idx not in layers:
            continue

        def hook(module, inputs, _output, layer=layer_idx):
            hidden = inputs[0]
            selected, weights = route(module, hidden.reshape(-1, hidden.shape[-1]),
                                      "qwen", top_k, norm)
            for eid, count in enumerate(torch.bincount(
                    selected.flatten(), minlength=len(module.experts)).tolist()):
                counts[layer][eid] += count
            collector.capture(layer, module, hidden, selected, weights)

        handles.append(block.register_forward_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                device = next(model.parameters()).device
                model(**{k: v.to(device) for k, v in batch.items() if k != "labels"})
    finally:
        for handle in handles:
            handle.remove()
    train, _, _ = collector.result()
    out.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        torch.save({"samples": _layer_value(train, layer),
                    "frequency": dict(counts[layer]),
                    "frequency_source": "full_routing_count",
                    "calibration_blocks": args.calibration_blocks,
                    "sample_cap": args.calibration_samples}, out / f"layer_{layer}.pt")
    del model, collector, train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def get_calibration_path(args, model_dir: Path, layers: list[int]) -> Path:
    if args.calibration_cache:
        path = args.calibration_cache.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    path = args.output_dir.expanduser().resolve() / "calibration"
    if all((path / f"layer_{layer}.pt").is_file() for layer in layers):
        return path
    return collect_calibration(args, model_dir, layers, path)


def make_grouping(layer: int, experts: list[int], samples: dict,
                  frequency: dict, reader: WeightReader, args,
                  device: torch.device) -> tuple[list[list[int]], np.ndarray, dict]:
    """Singleton probes, GS-MoE J_i(P_j), normalized affinity, balanced medoids."""
    probes = {}
    for i in experts:
        probes[i] = group_projections([i], samples, frequency, reader, layer,
                                      args.probe_rank, args.covariance_ridge, device)
    gains = torch.empty(len(experts), len(experts), dtype=torch.float64)
    for row, i in enumerate(experts):
        x = samples[i]["gate"].float().to(device)
        middle = samples[i]["down"].float().to(device)
        outputs = {}
        for key in PROJECTIONS:
            w = reader.get(layer, i, key).float().to(device)
            outputs[key] = (middle if key == "down" else x) @ w.T
        for col, j in enumerate(experts):
            p = {key: probes[j][key].to(device) for key in PROJECTIONS}
            gains[row, col] = reconstruction_gain(
                x, outputs, p, frequency[i], args.down_beta,
                args.covariance_ridge).double().cpu()
    distance, own = affinity_from_gains(gains)
    local_groups = balanced_kmedoids(distance, args.num_groups, args.seed + layer,
                                      args.grouping_restarts, args.grouping_iterations)
    groups = [[experts[j] for j in group] for group in local_groups]
    validate_groups(groups, experts, args.num_groups)
    diagnostics = {"self_gain": {str(i): float(own[j]) for j, i in enumerate(experts)},
                   "frequency": {str(i): float(frequency[i]) for i in experts},
                   "offdiagonal_affinity_mean": float((1 - distance)[~np.eye(
                       len(experts), dtype=bool)].mean()) if len(experts) > 1 else None}
    return groups, distance, diagnostics


def grouping_and_bases(layer: int, experts: list[int], samples: dict,
                       frequency: dict, reader: WeightReader, args,
                       device: torch.device) -> tuple[list[list[int]], dict, dict]:
    """Reuse cached grouping and maximum-rank bases across analyze/compress runs."""
    source = args.grouping_cache_dir or args.output_dir / "grouping"
    source = source.expanduser().resolve()
    path = source / f"layer_{layer}.pt"
    required = {"layer": layer, "experts": experts, "num_groups": args.num_groups,
                "probe_rank": args.probe_rank, "shared_rank": max(args.shared_ranks),
                "seed": args.seed, "covariance_ridge": args.covariance_ridge,
                "down_beta": args.down_beta}
    if path.is_file():
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if cached["settings"] != required:
            raise ValueError(f"grouping cache settings mismatch: {path}")
        groups = cached["groups"]
        validate_groups(groups, experts, args.num_groups)
        return groups, cached["bases"], cached["diagnostics"]
    if args.grouping_cache_dir:
        raise FileNotFoundError(path)
    groups, distance, diagnostics = make_grouping(
        layer, experts, samples, frequency, reader, args, device)
    bases = {j: group_projections(group, samples, frequency, reader, layer,
                                  max(args.shared_ranks), args.covariance_ridge,
                                  device) for j, group in enumerate(groups)}
    source.mkdir(parents=True, exist_ok=True)
    torch.save({"settings": required, "groups": groups, "bases": bases,
                "diagnostics": diagnostics, "distance": distance}, path)
    return groups, bases, diagnostics


class SharedProjectionGroup(nn.Module):
    def __init__(self, projections: dict, device: torch.device):
        super().__init__()
        self.p = nn.ParameterDict({key: nn.Parameter(value.to(device), requires_grad=False)
                                   for key, value in projections.items()})


class ProjectedExpert(nn.Module):
    def __init__(self, base: SharedProjectionGroup, parts: dict,
                 device: torch.device):
        super().__init__()
        self.base = base
        self.coeff = nn.ParameterDict()
        self.u = nn.ParameterDict()
        self.v = nn.ParameterDict()
        for key, part in parts.items():
            self.coeff[key] = nn.Parameter(part["coeff"].to(device), requires_grad=False)
            if "u" in part:
                self.u[key] = nn.Parameter(part["u"].to(device), requires_grad=False)
                self.v[key] = nn.Parameter(part["v"].to(device), requires_grad=False)

    def project(self, key: str, x: torch.Tensor) -> torch.Tensor:
        p = self.base.p[key]
        if key == "down":
            y = F.linear(F.linear(x, self.coeff[key]), p)
        else:
            y = F.linear(F.linear(x, p.T), self.coeff[key])
        if key in self.u:
            y = y + F.linear(F.linear(x, self.v[key]), self.u[key])
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.project("gate", x)
        up = self.project("up", x)
        return self.project("down", F.silu(gate) * up)


def install_checkpoint(model: nn.Module, manifest: Path) -> dict:
    metadata = json.loads(manifest.read_text())
    if metadata.get("format") != FORMAT:
        raise ValueError("invalid grouping-gsmoe checkpoint format")
    for filename in metadata["layer_files"]:
        data = torch.load(manifest.parent / filename, map_location="cpu", weights_only=False)
        layer = int(data["layer"])
        block = model.model.layers[layer].mlp
        router = block.gate
        shared = getattr(block, "shared_expert", None)
        shared_gate = getattr(block, "shared_expert_gate", None)
        top_k = block.top_k
        ids = sorted(int(i) for i in data["experts"])
        validate_groups(data["groups"], ids, len(data["groups"]))
        device = next(block.parameters()).device
        bases = {int(g): SharedProjectionGroup(p, device)
                 for g, p in data["bases"].items()}
        owner = {int(i): g for g, members in enumerate(data["groups"]) for i in members}
        for i, parts in data["experts"].items():
            block.experts[int(i)] = ProjectedExpert(bases[owner[int(i)]], parts, device)
        if (block.gate is not router or getattr(block, "shared_expert", None) is not shared
                or getattr(block, "shared_expert_gate", None) is not shared_gate
                or block.top_k != top_k):
            raise AssertionError("router or shared expert changed")
    return metadata


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(x) for x in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(x) for x in value)
    return 0


def checkpoint_components(payload: dict) -> dict:
    shared = _tensor_bytes(payload["bases"])
    coeff = sum(_tensor_bytes(part["coeff"])
                for expert in payload["experts"].values() for part in expert.values())
    residual = sum(_tensor_bytes(part.get("u")) + _tensor_bytes(part.get("v"))
                   for expert in payload["experts"].values() for part in expert.values())
    return {"shared_projection_bytes": shared,
            "expert_coefficient_bytes": coeff,
            "residual_factor_bytes": residual}


def save_layer(path: Path, payload: dict) -> tuple[int, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, temp)
    os.replace(temp, path)
    return path.stat().st_size, checkpoint_components(payload)


def _metrics(actual: torch.Tensor, target: torch.Tensor) -> dict:
    a, b = actual.float(), target.float()
    if a.shape != b.shape:
        raise ValueError(f"diagnostic output shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    _finite(a, "diagnostic output")
    _finite(b, "reference output")
    error2 = _norm2(a - b)
    ref2 = _norm2(b)
    a2 = _norm2(a)
    dot = float((a.double() * b.double()).sum())
    return {"relative_l2": math.sqrt(error2 / ref2) if ref2 else None,
            "cosine": dot / math.sqrt(a2 * ref2) if a2 and ref2 else None,
            "output_norm_squared_ratio": _ratio(a2, ref2),
            "error_squared_norm": error2, "reference_squared_norm": ref2}


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _part_for_rank(residual: torch.Tensor, left: torch.Tensor,
                   right: torch.Tensor, rank: int, coeff: torch.Tensor) -> tuple[dict, dict]:
    oracle = left[:, :rank] @ right[:rank]
    oracle_tail = residual - oracle
    r2 = _norm2(residual)
    l2 = _norm2(oracle)
    e2 = _norm2(oracle_tail)
    if r2 and abs((l2 + e2) / r2 - 1) > 3e-4:
        raise AssertionError("FP32 SVD energy decomposition failed")
    part = {"coeff": coeff}
    if rank:
        part["u"] = left[:, :rank].to(torch.bfloat16).cpu().contiguous()
        part["v"] = right[:rank].to(torch.bfloat16).cpu().contiguous()
    stored = (part["u"].to(residual.device).float() @
              part["v"].to(residual.device).float()) if rank else torch.zeros_like(residual)
    stored_tail2 = _norm2(residual - stored)
    return part, {"oracle_uv_energy_over_residual": _ratio(l2, r2),
                  "oracle_tail_energy_over_residual": _ratio(e2, r2),
                  "oracle_tail_squared_norm": e2,
                  "stored_tail_squared_norm": stored_tail2}


def _projected_output(x: torch.Tensor, part: dict, p: torch.Tensor,
                      direction: str) -> torch.Tensor:
    """Match ProjectedExpert's BF16 checkpoint arithmetic on routed activations."""
    x = x.to(torch.bfloat16)
    coeff = part["coeff"].to(x.device)
    p = p.to(x.device)
    if direction == "down":
        y = F.linear(F.linear(x, coeff), p)
    else:
        y = F.linear(F.linear(x, p.T), coeff)
    if "u" in part:
        y = y + F.linear(F.linear(x, part["v"].to(x.device)),
                         part["u"].to(x.device))
    return y


def _summarize_projections(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["projection"], row["shared_rank"], row["residual_rank"])].append(row)
    result = []
    for (projection, sr, rr), items in sorted(buckets.items()):
        w2 = sum(r["weight_squared_norm"] for r in items)
        r2 = sum(r["shared_residual_squared_norm"] for r in items)
        e2 = sum(r["oracle_tail_squared_norm"] for r in items)
        stored_e2 = sum(r["stored_tail_squared_norm"] for r in items)
        output_ref2 = sum(r["output_reference_squared_norm"] for r in items)
        output_err2 = sum(r["output_error_squared_norm"] for r in items)
        result.append({
            "projection": projection, "shared_rank": sr, "residual_rank": rr,
            "expert_count": len(items),
            "mean_shared_rho": sum(r["shared_residual_rho"] for r in items) / len(items),
            "pooled_shared_rho": math.sqrt(r2 / w2) if w2 else None,
            "mean_oracle_tail_rho": sum(r["oracle_tail_rho"] for r in items) / len(items),
            "pooled_oracle_tail_rho": math.sqrt(e2 / w2) if w2 else None,
            "pooled_stored_tail_rho": math.sqrt(stored_e2 / w2) if w2 else None,
            "pooled_oracle_uv_energy_over_residual": _ratio(r2 - e2, r2),
            "mean_output_relative_l2": sum(r["output_relative_l2"] for r in items) / len(items),
            "pooled_output_relative_l2": math.sqrt(output_err2 / output_ref2)
            if output_ref2 else None,
            "mean_output_cosine": sum(r["output_cosine"] for r in items) / len(items),
        })
    return result


def _summarize_swiglu(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["shared_rank"], row["residual_rank"])].append(row)
    result = []
    for (sr, rr), items in sorted(buckets.items()):
        ref2 = sum(r["reference_squared_norm"] for r in items)
        err2 = sum(r["error_squared_norm"] for r in items)
        result.append({"shared_rank": sr, "residual_rank": rr,
                       "expert_count": len(items),
                       "mean_relative_l2": sum(r["relative_l2"] for r in items) / len(items),
                       "pooled_relative_l2": math.sqrt(err2 / ref2) if ref2 else None,
                       "mean_cosine": sum(r["cosine"] for r in items) / len(items)})
    return result


def _plot(out: Path, projections: list[dict], swiglu: list[dict], groups: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, PROJECTIONS):
        for sr in sorted({r["shared_rank"] for r in projections}):
            rows = sorted((r for r in projections if r["projection"] == key and
                           r["shared_rank"] == sr), key=lambda x: x["residual_rank"])
            ax.plot([r["residual_rank"] for r in rows],
                    [r["pooled_stored_tail_rho"] for r in rows], marker="o",
                    label=f"shared {sr}")
        ax.set(title=key, xlabel="residual rank", ylabel="pooled stored Tail / W")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis / "rank_vs_tail_norm.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for ax, key in zip(axes, (*PROJECTIONS, "SwiGLU")):
        rows_all = swiglu if key == "SwiGLU" else [r for r in projections if r["projection"] == key]
        field = "pooled_relative_l2" if key == "SwiGLU" else "pooled_output_relative_l2"
        for sr in sorted({r["shared_rank"] for r in rows_all}):
            rows = sorted((r for r in rows_all if r["shared_rank"] == sr),
                          key=lambda x: x["residual_rank"])
            ax.plot([r["residual_rank"] for r in rows], [r[field] for r in rows],
                    marker="o", label=f"shared {sr}")
        ax.set(title=key, xlabel="residual rank", ylabel="pooled output relative L2")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(analysis / "rank_vs_output_error.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = Counter(r["group_size"] for r in groups)
    ax.bar(list(sorted(counts)), [counts[k] for k in sorted(counts)])
    ax.set(xlabel="experts per group", ylabel="number of groups", title="Group size distribution")
    fig.tight_layout()
    fig.savefig(analysis / "group_size_distribution.png", dpi=150)
    plt.close(fig)


def _report(config: dict, projection_summary: list[dict],
            swiglu_summary: list[dict], checkpoints: list[dict], groups: list[dict]) -> str:
    lines = ["# GS-MoE gain grouping + shared projection + residual SVD", "",
             "This experiment adapts GS-MoE's reconstruction gain and fixed-size "
             "balanced K-medoids to Qwen1.5-MoE. It does not run QJL, pruning, or training.", "",
             f"Layers: {config['layers']}; experts: {config['experts']}; "
             f"shared ranks: {config['shared_ranks']}; residual ranks: {config['residual_ranks']}.",
             f"Calibration: {config['calibration_cache']} "
             f"(routing frequency: {config['frequency_source']}).", "",
             "Shared projection coefficients and bases are stored as BF16. The FP32 "
             "residual is defined after reconstructing those stored BF16 factors; "
             "the residual SVD oracle is measured before its factors are rounded to BF16. "
             "Pooled norm ratios use summed squared norms, while mean metrics average experts. "
             "Routed output errors run BF16 checkpoint arithmetic and are separate "
             "from Frobenius/SVD energy.", "",
             "## Projection diagnostics", "",
             "| Projection | Shared rank | Residual rank | Mean R/W | Pooled R/W | "
             "Pooled stored Tail/W | Pooled output L2 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in projection_summary:
        lines.append(f"| {r['projection']} | {r['shared_rank']} | {r['residual_rank']} | "
                     f"{r['mean_shared_rho']:.5f} | {r['pooled_shared_rho']:.5f} | "
                     f"{r['pooled_stored_tail_rho']:.5f} | {r['pooled_output_relative_l2']:.5f} |")
    lines += ["", "## Full SwiGLU expert output", "",
              "| Shared rank | Residual rank | Mean relative L2 | Pooled relative L2 | Mean cosine |",
              "|---:|---:|---:|---:|---:|"]
    for r in swiglu_summary:
        lines.append(f"| {r['shared_rank']} | {r['residual_rank']} | "
                     f"{r['mean_relative_l2']:.5f} | {r['pooled_relative_l2']:.5f} | "
                     f"{r['mean_cosine']:.5f} |")
    lines += ["", f"Group sizes: {dict(sorted(Counter(g['group_size'] for g in groups).items()))}.",
              "", "## Checkpoint storage", ""]
    if checkpoints:
        lines += ["| Shared rank | Residual rank | Bytes | Effective bpw | BF16 ratio |",
                  "|---:|---:|---:|---:|---:|"]
        for r in checkpoints:
            lines.append(f"| {r['shared_rank']} | {r['residual_rank']} | "
                         f"{r['checkpoint_bytes']} | {r['effective_bpw']:.3f} | "
                         f"{r['bf16_compression_ratio']:.3f}x |")
    else:
        lines.append("Analyze mode does not write checkpoints; exact bytes are measured in compress mode.")
    lines += ["", "The original HF backbone, router, and shared expert are external "
              "dependencies of these expert-only checkpoints. Full evaluation was not "
              "run as part of this report.", ""]
    return "\n".join(lines)


@torch.no_grad()
def analyze_or_compress(args) -> dict:
    model_dir = args.model_path.expanduser().resolve()
    model_config = json.loads((model_dir / "config.json").read_text())
    layers = selected(args.layers, int(model_config["num_hidden_layers"]))
    experts = list(range(int(model_config["num_experts"])))
    if args.max_experts is not None:
        experts = experts[:args.max_experts]
    if not experts or not 1 <= args.num_groups <= len(experts):
        raise ValueError("invalid expert or group count")
    hidden = int(model_config["hidden_size"])
    intermediate = int(model_config["moe_intermediate_size"])
    if any(not 0 < r <= hidden for r in args.shared_ranks):
        raise ValueError("shared rank outside hidden dimension")
    if any(not 0 <= r <= min(hidden, intermediate) for r in args.residual_ranks):
        raise ValueError("residual rank outside weight dimensions")
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    calibration = get_calibration_path(args, model_dir, layers)
    projection_rows, swiglu_rows, group_rows, routing_rows = [], [], [], []
    file_names = defaultdict(list)
    byte_totals = defaultdict(lambda: defaultdict(int))
    original_weight_count = 0
    frequency_sources = set()
    reader = WeightReader(model_dir)
    try:
        for layer in layers:
            samples, frequency, source = load_calibration_layer(calibration, layer, experts)
            frequency_sources.add(source)
            routing_rows.extend({"layer": layer, "expert": i,
                                 "routing_frequency": frequency[i],
                                 "calibration_samples": samples[i]["gate"].shape[0],
                                 "frequency_source": source} for i in experts)
            if samples[experts[0]]["gate"].shape[1] != hidden or \
                    samples[experts[0]]["down"].shape[1] != intermediate:
                raise ValueError("calibration shape does not match Qwen1.5-MoE weights")
            groups, bases, grouping_diagnostics = grouping_and_bases(
                layer, experts, samples, frequency, reader, args, device)
            for group_id, members in enumerate(groups):
                group_rows.append({"layer": layer, "group_id": group_id,
                                   "group_size": len(members),
                                   "experts": ",".join(map(str, members)),
                                   "routing_frequency": sum(frequency[i] for i in members),
                                   "frequency_source": source,
                                   "offdiagonal_affinity_mean": grouping_diagnostics[
                                       "offdiagonal_affinity_mean"]})
            for sr in args.shared_ranks:
                payloads = {}
                if args.mode == "compress":
                    payloads = {rr: {"layer": layer, "groups": groups,
                                     "bases": {}, "experts": {}}
                                for rr in args.residual_ranks}
                for group_id, members in enumerate(groups):
                    stored_p = {key: bases[group_id][key][:, :sr].to(torch.bfloat16)
                                .cpu().contiguous() for key in PROJECTIONS}
                    for payload in payloads.values():
                        payload["bases"][group_id] = stored_p
                    for expert in members:
                        x = samples[expert]["gate"].to(device=device, dtype=torch.bfloat16)
                        middle = samples[expert]["down"].to(device=device, dtype=torch.bfloat16)
                        cov_x = _cov(x.float())
                        weights = {key: reader.get(layer, expert, key).to(device)
                                   for key in PROJECTIONS}
                        if sr == args.shared_ranks[0]:
                            original_weight_count += sum(w.numel() for w in weights.values())
                        original_gate = F.linear(x, weights["gate"])
                        original_up = F.linear(x, weights["up"])
                        original_full = F.linear(
                            F.silu(original_gate) * original_up, weights["down"])
                        parts_by_rank = {rr: {} for rr in args.residual_ranks}
                        for key in PROJECTIONS:
                            w = weights[key].float()
                            p = bases[group_id][key][:, :sr].to(device)
                            coeff_oracle = calibration_coefficient(
                                w, p, cov_x, key, args.covariance_ridge)
                            oracle_shared = reconstruct_shared(coeff_oracle, p, key)
                            coeff = coeff_oracle.to(torch.bfloat16).cpu().contiguous()
                            shared_stored = reconstruct_shared(
                                coeff.to(device).float(), stored_p[key].to(device).float(), key)
                            residual = w - shared_stored
                            left, right = svd_factors(residual, max(args.residual_ranks))
                            routed = middle if key == "down" else x
                            y_original = F.linear(routed, weights[key])
                            shared_metrics = _metrics(_projected_output(
                                routed, {"coeff": coeff}, stored_p[key], key), y_original)
                            w2, r2 = _norm2(w), _norm2(residual)
                            oracle_r2 = _norm2(w - oracle_shared)
                            for rr in args.residual_ranks:
                                part, svd = _part_for_rank(residual, left, right, rr, coeff)
                                parts_by_rank[rr][key] = part
                                if payloads:
                                    payloads[rr]["experts"].setdefault(expert, {})[key] = part
                                error = _metrics(_projected_output(
                                    routed, part, stored_p[key], key), y_original)
                                e2 = svd["oracle_tail_squared_norm"]
                                projection_rows.append({
                                    "layer": layer, "group_id": group_id,
                                    "group_size": len(members), "expert": expert,
                                    "projection": key, "shared_rank": sr,
                                    "residual_rank": rr, "input_source": "routed",
                                    "frequency_source": source,
                                    "routing_frequency": frequency[expert],
                                    "calibration_samples": routed.shape[0],
                                    "weight_squared_norm": w2,
                                    "oracle_shared_residual_rho": math.sqrt(oracle_r2 / w2),
                                    "shared_residual_squared_norm": r2,
                                    "shared_residual_rho": math.sqrt(r2 / w2),
                                    "shared_residual_energy_ratio": r2 / w2,
                                    "oracle_uv_energy_over_residual": svd[
                                        "oracle_uv_energy_over_residual"],
                                    "oracle_tail_energy_over_residual": svd[
                                        "oracle_tail_energy_over_residual"],
                                    "oracle_tail_squared_norm": e2,
                                    "oracle_tail_rho": math.sqrt(e2 / w2),
                                    "stored_tail_squared_norm": svd[
                                        "stored_tail_squared_norm"],
                                    "stored_tail_rho": math.sqrt(
                                        svd["stored_tail_squared_norm"] / w2),
                                    "shared_output_relative_l2": shared_metrics["relative_l2"],
                                    "output_relative_l2": error["relative_l2"],
                                    "output_cosine": error["cosine"],
                                    "output_norm_squared_ratio": error[
                                        "output_norm_squared_ratio"],
                                    "output_error_squared_norm": error["error_squared_norm"],
                                    "output_reference_squared_norm": error[
                                        "reference_squared_norm"],
                                })
                        for rr, parts in parts_by_rank.items():
                            gate = _projected_output(x, parts["gate"], stored_p["gate"], "gate")
                            up = _projected_output(x, parts["up"], stored_p["up"], "up")
                            whole = _projected_output(F.silu(gate) * up, parts["down"],
                                                      stored_p["down"], "down")
                            swiglu_rows.append({"layer": layer, "group_id": group_id,
                                                "expert": expert, "shared_rank": sr,
                                                "residual_rank": rr,
                                                "routing_frequency": frequency[expert],
                                                **_metrics(whole, original_full)})
                for rr, payload in payloads.items():
                    folder = out / "checkpoints" / f"shared_{sr}" / f"residual_{rr}"
                    filename = f"layer_{layer}.pt"
                    size, components = save_layer(folder / filename, payload)
                    file_names[(sr, rr)].append(filename)
                    byte_totals[(sr, rr)]["layer_file_bytes"] += size
                    for key, value in components.items():
                        byte_totals[(sr, rr)][key] += value
            print(f"[grouping-gsmoe] layer {layer}: {len(groups)} balanced groups; "
                  f"{len(args.shared_ranks)} shared ranks, {len(args.residual_ranks)} residual ranks",
                  flush=True)
            del samples, bases
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        reader.close()
    config = {"model": MODEL_ID, "model_path": str(model_dir), "mode": args.mode,
              "layers": layers, "experts": experts, "num_groups": args.num_groups,
              "shared_ranks": args.shared_ranks, "residual_ranks": args.residual_ranks,
              "probe_rank": args.probe_rank, "seed": args.seed,
              "down_beta": args.down_beta, "covariance_ridge": args.covariance_ridge,
              "calibration_cache": str(calibration),
              "frequency_source": ",".join(sorted(frequency_sources)),
              "grouping_cache_dir": str(args.grouping_cache_dir or out / "grouping")}
    checkpoints = []
    if args.mode == "compress":
        for (sr, rr), files in sorted(file_names.items()):
            folder = out / "checkpoints" / f"shared_{sr}" / f"residual_{rr}"
            manifest = folder / "manifest.json"
            metadata = {"format": FORMAT, "model": MODEL_ID,
                        "shared_rank": sr, "residual_rank": rr,
                        "model_path": str(model_dir), "layers": layers,
                        "experts": experts, "layer_files": files,
                        "num_groups": args.num_groups, "seed": args.seed}
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
            totals = dict(byte_totals[(sr, rr)])
            size = totals["layer_file_bytes"] + manifest.stat().st_size
            tensors = sum(value for key, value in totals.items() if key != "layer_file_bytes")
            if size < tensors:
                raise AssertionError("checkpoint smaller than stored factors")
            checkpoints.append({"shared_rank": sr, "residual_rank": rr,
                                "manifest": str(manifest), "checkpoint_bytes": size,
                                "shared_projection_bytes": totals["shared_projection_bytes"],
                                "expert_coefficient_bytes": totals["expert_coefficient_bytes"],
                                "residual_factor_bytes": totals["residual_factor_bytes"],
                                "metadata_bytes": size - tensors,
                                "original_bf16_expert_bytes": 2 * original_weight_count,
                                "effective_bpw": 8 * size / original_weight_count,
                                "bf16_compression_ratio": 2 * original_weight_count / size})
    projection_summary = _summarize_projections(projection_rows)
    swiglu_summary = _summarize_swiglu(swiglu_rows)
    analysis = out / "analysis"
    _write_csv(analysis / "per_projection.csv", projection_rows)
    _write_csv(analysis / "per_expert_swiglu.csv", swiglu_rows)
    _write_csv(analysis / "groups.csv", group_rows)
    _write_csv(analysis / "routing_frequency.csv", routing_rows)
    _write_csv(analysis / "projection_summary.csv", projection_summary)
    _write_csv(analysis / "swiglu_summary.csv", swiglu_summary)
    if checkpoints:
        _write_csv(analysis / "checkpoint_storage.csv", checkpoints)
    else:
        (analysis / "checkpoint_storage.csv").unlink(missing_ok=True)
    summary = {"config": config, "projection_summary": projection_summary,
               "swiglu_summary": swiglu_summary, "checkpoints": checkpoints,
               "group_size_distribution": dict(sorted(Counter(
                   row["group_size"] for row in group_rows).items()))}
    (analysis / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (analysis / "REPORT.md").write_text(_report(
        config, projection_summary, swiglu_summary, checkpoints, group_rows))
    _plot(out, projection_summary, swiglu_summary, group_rows)
    return summary


def evaluate(args) -> dict:
    """User-invoked full lm-eval of one saved shared/residual rank checkpoint."""
    if args.checkpoint is None:
        raise ValueError("eval mode requires --checkpoint")
    if args.eval_limit < 0:
        raise ValueError("eval-limit must be nonnegative; zero means full evaluation")
    from transformers import AutoTokenizer
    from rescode.evaluation import evaluate_fewshot
    from rescode.merging.eval_summary import report_eval_summary

    manifest = args.checkpoint.expanduser().resolve()
    metadata = json.loads(manifest.read_text())
    if metadata.get("model") != MODEL_ID or metadata.get("format") != FORMAT:
        raise ValueError("checkpoint model or format mismatch")
    model_dir = args.model_path.expanduser().resolve()
    model = _load_hf_model(model_dir, args.gpu_memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    install_checkpoint(model, manifest)
    out = (args.output_dir.expanduser().resolve() / "eval" /
           f"shared_{metadata['shared_rank']}" / f"residual_{metadata['residual_rank']}")
    out.mkdir(parents=True, exist_ok=True)
    log = out / "lm_eval.txt"
    result = evaluate_fewshot(model, tokenizer, args.tasks, 0,
                              eval_batch_size=args.eval_batch_size,
                              output_path=str(log), limit=args.eval_limit or None)
    task_summary = report_eval_summary(
        result, args.tasks, str(log),
        header=(f"GS-MoE grouping shared {metadata['shared_rank']} "
                f"residual {metadata['residual_rank']}"))
    metrics = {"checkpoint": str(manifest), "shared_rank": metadata["shared_rank"],
               "residual_rank": metadata["residual_rank"], "tasks": args.tasks,
               "eval_limit": args.eval_limit, "task_summary": task_summary,
               "task_results": result.get("results", {})}
    (out / "eval_metrics.json").write_text(json.dumps(metrics, indent=2,
                                                     default=str) + "\n")
    return metrics


def _rank_list(value: str, allow_zero: bool) -> list[int]:
    ranks = sorted({int(item.strip()) for item in value.split(",")})
    if not ranks or any(r < (0 if allow_zero else 1) for r in ranks):
        raise ValueError(f"invalid rank list: {value}")
    return ranks


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analyze", "compress", "eval"), required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat"))
    parser.add_argument("--layers", default="all")
    parser.add_argument("--max-experts", type=int)
    parser.add_argument("--num-groups", type=int, default=30)
    parser.add_argument("--probe-rank", type=int, default=64)
    parser.add_argument("--shared-ranks", default="64,128,256")
    parser.add_argument("--residual-ranks", default="0,32,64,128,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grouping-restarts", type=int, default=8)
    parser.add_argument("--grouping-iterations", type=int, default=25)
    parser.add_argument("--down-beta", type=float, default=1.0)
    parser.add_argument("--covariance-ridge", type=float, default=1e-4)
    parser.add_argument("--calibration-cache", type=Path,
                        help="layer-wise cache directory or QJL routed_samples .pt cache")
    parser.add_argument("--grouping-cache-dir", type=Path,
                        help="reuse analyze-mode grouping and maximum-rank bases")
    parser.add_argument("--calibration-data", type=Path, default=DEFAULT_C4)
    parser.add_argument("--calibration-blocks", type=int, default=32)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--calibration-batch-size", type=int, default=2)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/grouping_gsmoe")
    args = parser.parse_args(argv)
    args.shared_ranks = _rank_list(args.shared_ranks, allow_zero=False)
    args.residual_ranks = _rank_list(args.residual_ranks, allow_zero=True)
    if args.probe_rank < 1 or args.covariance_ridge <= 0 or args.down_beta < 0:
        parser.error("probe-rank and covariance-ridge must be positive; down-beta nonnegative")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.mode == "eval":
        return evaluate(args)
    return analyze_or_compress(args)


if __name__ == "__main__":
    main()
