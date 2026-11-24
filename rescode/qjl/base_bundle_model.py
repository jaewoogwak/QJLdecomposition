"""Pairwise QJL base with optional expert-specific low-rank residuals.

The base shards contain only shared packed sketches and individual FP32 row
norms. Rank factors live in separate shards and are never read for case A.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bundled_model import dense_qjl_base
from .codebook_model import bank_for
from .core import PackedResidual, pack_residual, qjl_triton_fused_gate_up, unpack_signs
from .exponent_model import PROJECTIONS, _install
from .residual_bundle_model import (KEYS, _tie_generator, bundle_signs,
                                    make_pairs, validate_pairs)

FORMAT = "pairwise-qjl-base-lowrank-v1"
RANKS = (32, 64, 128, 256, 384, 512, 768, 1024)


class LowRankCorrection(nn.Module):
    def __init__(self, u: torch.Tensor, v: torch.Tensor):
        super().__init__()
        self.register_buffer("u", u.contiguous())
        self.register_buffer("v", v.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x.to(self.v.dtype), self.v)
        return F.linear(hidden, self.u).to(x.dtype)


class BundledBaseExpert(nn.Module):
    def __init__(self, bases: dict, corrections: dict, backend: str):
        super().__init__()
        self.bases = nn.ModuleDict(bases)
        self.corrections = nn.ModuleDict(corrections)
        self.backend = backend

    def _projection(self, key: str, x: torch.Tensor, z=None):
        base = self.bases[key]
        y = (base.forward_projected(z, x.dtype, x.shape[:-1])
             if z is not None else base(x))
        if key in self.corrections:
            y = y + self.corrections[key](x)
        return y

    def forward(self, x: torch.Tensor):
        gate, up = self.bases["gate"], self.bases["up"]
        p = gate.bank.get(x.shape[-1], x.device)
        z = x.reshape(-1, x.shape[-1]).to(p.dtype) @ p
        if self.backend == "triton":
            g, u = qjl_triton_fused_gate_up(z, gate, up, x.dtype)
            g = g.reshape(*x.shape[:-1], -1)
            u = u.reshape(*x.shape[:-1], -1)
            if "gate" in self.corrections:
                g = g + self.corrections["gate"](x)
                u = u + self.corrections["up"](x)
        else:
            g = self._projection("gate", x, z)
            u = self._projection("up", x, z)
        return self._projection("down", F.silu(g) * u)


def _fraction(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float(numerator / denominator.clamp_min(1e-30))


def _rank_needed(cumulative: torch.Tensor, threshold: float) -> int:
    return int(torch.searchsorted(cumulative, threshold).item()) + 1


def spectrum_metrics(s: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                     base: torch.Tensor, u: torch.Tensor, v: torch.Tensor, ranks=RANKS):
    """Exact FP32 spectrum, optimal tail error, and actual BF16 factor error."""
    energy = s.square()
    total = energy.sum().clamp_min(1e-30)
    cumulative = energy.cumsum(0) / total
    result = {"residual_frobenius_ratio": _fraction(
                  torch.linalg.vector_norm(residual), torch.linalg.vector_norm(weight.float())),
              "rank_for_90pct": _rank_needed(cumulative, .90),
              "rank_for_95pct": _rank_needed(cumulative, .95),
              "rank_for_99pct": _rank_needed(cumulative, .99)}
    for rank in ranks:
        k = min(rank, s.numel())
        retention = float(cumulative[k - 1])
        result[f"energy_rank_{rank}"] = retention
        result[f"optimal_residual_error_rank_{rank}"] = math.sqrt(max(0., 1. - retention))
        result[f"optimal_weight_error_rank_{rank}"] = (
            result["residual_frobenius_ratio"] * result[f"optimal_residual_error_rank_{rank}"])
    approximation = u.float() @ v.float()
    result["bf16_factor_residual_error"] = _fraction(
        torch.linalg.vector_norm(residual - approximation), torch.linalg.vector_norm(residual))
    result["bf16_factor_weight_error"] = _fraction(
        torch.linalg.vector_norm(base + approximation - weight.float()),
        torch.linalg.vector_norm(weight.float()))
    return result


@torch.no_grad()
def compress_layer(block, layer: int, config: dict, banks: dict):
    """Return one layer's CPU base/factor shards and full spectrum records."""
    if len(block.experts) != 60:
        raise ValueError(f"layer {layer}: expected 60 routed experts")
    base_state = {"format": FORMAT, "layer": layer, "projections": {}}
    factor_state = {"format": FORMAT, "layer": layer, "projections": {}}
    records, spectra = [], {}
    for key, attr in PROJECTIONS.items():
        weights = [getattr(expert, attr).weight.detach() for expert in block.experts]
        rows, cols = weights[0].shape
        bank = bank_for(banks, cols, config["qjl_d_factor"], config["projection_seed"])
        individual, norms = {}, {}
        for i, weight in enumerate(weights):
            packed, norm = pack_residual(weight, bank, config["row_chunk"])
            individual[i], norms[i] = packed.cpu(), norm.cpu()
        pairs = make_pairs(individual, bank.d, "similarity",
                           config["pairing_seed"] + layer * 1009 + KEYS.index(key) * 9176)
        validate_pairs(pairs, 60)
        sketches, sketch_ids, factors = {}, {}, {}
        for pair_id, (i, j) in enumerate(pairs):
            shared = bundle_signs(individual[i], individual[j], bank.d, "majority",
                                  _tie_generator(config["bundle_seed"], layer, key,
                                                 pair_id).initial_seed(),
                                  row_chunk=config["row_chunk"])
            sketches[pair_id] = shared
            sketch_ids[i] = sketch_ids[j] = pair_id
            bits_i = unpack_signs(individual[i], bank.d)
            bits_j = unpack_signs(individual[j], bank.d)
            bits_g = unpack_signs(shared, bank.d)
            agreement = float((bits_i == bits_j).float().mean())
            for expert_id, bits in ((i, bits_i), (j, bits_j)):
                weight = weights[expert_id]
                packed_gpu = shared.to(weight.device)
                norm_gpu = norms[expert_id].to(weight.device)
                base = torch.cat([chunk for _, chunk in dense_qjl_base(
                    packed_gpu, norm_gpu, bank, cols, config["row_chunk"])], dim=0)
                residual = weight.float() - base
                identity_error = _fraction(
                    torch.linalg.vector_norm(base + residual - weight.float()),
                    torch.linalg.vector_norm(weight.float()))
                if identity_error > 1e-6:
                    raise AssertionError(f"exact reconstruction failed: {layer}/{key}/{expert_id}")
                # Full reduced SVD gives the complete nonzero spectrum and exact
                # optimal rank tails; factors are quantized only after SVD.
                u_full, s, vh = torch.linalg.svd(residual, full_matrices=False)
                rank = min(config["rank"], *residual.shape)
                u = (u_full[:, :rank] * s[:rank]).to(torch.bfloat16)
                v = vh[:rank].to(torch.bfloat16).contiguous()
                factors[expert_id] = {"u": u.cpu(), "v": v.cpu()}
                row = {"layer": layer, "projection": key, "expert": expert_id,
                       "pair_id": pair_id, "partner": j if expert_id == i else i,
                       "rows": rows, "cols": cols, "qjl_d": bank.d,
                       "rank": rank, "identity_error": identity_error,
                       "pre_bundle_bit_agreement": agreement,
                       "post_bundle_bit_preservation": float((bits == bits_g).float().mean())}
                row.update(spectrum_metrics(s, residual, weight, base, u, v))
                records.append(row)
                spectra[f"{key}_{expert_id}"] = s.cpu().numpy()
                del residual, base, u_full, s, vh, u, v
        base_state["projections"][key] = {"shape": (rows, cols), "d": bank.d,
                "pairs": pairs, "sketches": sketches, "sketch_id": sketch_ids,
                "norms": norms}
        factor_state["projections"][key] = factors
        del individual, weights
    return base_state, factor_state, records, spectra


def install_layer(block, base_state: dict, factor_state: dict | None,
                  config: dict, banks: dict, backend: str):
    if base_state.get("format") != FORMAT:
        raise ValueError("wrong pairwise QJL base checkpoint format")
    if factor_state is not None and (factor_state.get("format") != FORMAT or
                                     factor_state["layer"] != base_state["layer"]):
        raise ValueError("mismatched factor shard")
    device = next(block.parameters()).device
    projections = base_state["projections"]
    shared = {}
    for key in KEYS:
        part = projections[key]
        validate_pairs(part["pairs"], 60)
        if len(part["sketches"]) != 30 or len(part["norms"]) != 60:
            raise ValueError("incomplete pairwise sketch or norm state")
        shared[key] = {int(k): packed.to(device) for k, packed in part["sketches"].items()}
    experts = []
    for expert_id in range(60):
        bases, corrections = {}, {}
        for key in KEYS:
            part = projections[key]
            cols = part["shape"][1]
            bank = bank_for(banks, cols, config["qjl_d_factor"], config["projection_seed"])
            bases[key] = PackedResidual(shared[key][part["sketch_id"][expert_id]],
                       part["norms"][expert_id].to(device), cols, bank.d, bank, backend)
            if factor_state is not None:
                factors = factor_state["projections"][key][expert_id]
                corrections[key] = LowRankCorrection(factors["u"].to(device),
                                                      factors["v"].to(device))
        experts.append(BundledBaseExpert(bases, corrections, backend))
    router = block.gate
    shared_expert = getattr(block, "shared_expert", None)
    shared_gate = getattr(block, "shared_expert_gate", None)
    top_k = block.top_k
    _install(block, experts)
    if (block.gate is not router or getattr(block, "shared_expert", None) is not shared_expert
            or getattr(block, "shared_expert_gate", None) is not shared_gate
            or block.top_k != top_k):
        raise AssertionError("router or shared expert changed")


def load_checkpoint(model, checkpoint_dir, case: str, backend: str):
    """Load A from base shards only; B additionally reads BF16 factor shards."""
    import json
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "manifest.json").read_text())
    if config["format"] != FORMAT or case not in ("A", "B"):
        raise ValueError("invalid checkpoint format or case")
    banks = {}
    model._base_bundle_banks = banks
    for layer in config["layers"]:
        base = torch.load(checkpoint_dir / f"base_layer_{layer}.pt",
                          map_location="cpu", weights_only=False)
        factors = (torch.load(checkpoint_dir / f"rank_layer_{layer}.pt",
                              map_location="cpu", weights_only=False) if case == "B" else None)
        install_layer(model.model.layers[layer].mlp, base, factors, config, banks, backend)
        del base, factors
    return config


def resident_bytes(model):
    sketches, norms, factors = {}, {}, {}
    for layer in model.model.layers:
        for expert in getattr(layer.mlp, "_compact_experts", ()):
            for base in expert.bases.values():
                for source, destination in ((base.packed, sketches), (base.norms, norms)):
                    if source.is_cuda:
                        storage = source.untyped_storage()
                        destination[storage.data_ptr()] = storage.nbytes()
            for correction in expert.corrections.values():
                for source in (correction.u, correction.v):
                    if source.is_cuda:
                        storage = source.untyped_storage()
                        factors[storage.data_ptr()] = storage.nbytes()
    projection = sum(bank.resident_bytes("cuda") for bank in
                     getattr(model, "_base_bundle_banks", {}).values())
    return {"gpu_shared_sketch_bytes": sum(sketches.values()),
            "gpu_row_norm_bytes": sum(norms.values()),
            "gpu_lowrank_bytes": sum(factors.values()),
            "gpu_projection_bytes": projection,
            "gpu_compact_total_bytes": sum(sketches.values()) + sum(norms.values())
                                       + sum(factors.values()) + projection}


def checkpoint_sizes(checkpoint_dir, config, case: str):
    checkpoint_dir = Path(checkpoint_dir)
    sketch = norms = factors = weights = 0
    disk = (checkpoint_dir / "manifest.json").stat().st_size
    for layer in config["layers"]:
        base_path = checkpoint_dir / f"base_layer_{layer}.pt"
        base = torch.load(base_path, map_location="cpu", weights_only=False)
        disk += base_path.stat().st_size
        for part in base["projections"].values():
            rows, cols = part["shape"]
            weights += 60 * rows * cols
            sketch += sum(t.numel() * t.element_size() for t in part["sketches"].values())
            norms += sum(t.numel() * t.element_size() for t in part["norms"].values())
        if case == "B":
            rank_path = checkpoint_dir / f"rank_layer_{layer}.pt"
            rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
            disk += rank_path.stat().st_size
            factors += sum(t.numel() * t.element_size() for parts in
                           rank_state["projections"].values() for expert in parts.values()
                           for t in expert.values())
    return {"shared_sketch_bytes": sketch, "row_norm_bytes": norms,
            "lowrank_residual_bytes": factors, "actual_checkpoint_bytes": disk,
            "metadata_and_container_bytes": disk - sketch - norms - factors,
            "routed_weight_count": weights, "effective_bpw": 8 * disk / weights,
            "bf16_storage_reduction": 1 - disk / (2 * weights)}
