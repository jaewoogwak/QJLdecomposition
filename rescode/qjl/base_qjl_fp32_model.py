"""Packed QJL Base with an exact, checkpointed FP32 residual."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_qjl_model import PROJECTIONS, _pair
from .core import ExactResidual, PackedResidual, ProjectionBank
from .exponent_model import _install

FORMAT = "rescode-qjl-base-fp32-residual-v1"


class FP32Residual(ExactResidual):
    """Compute the dense residual in FP32, optionally on CPU to fit one GPU."""

    def __init__(self, residual: torch.Tensor):
        if residual.ndim != 2 or residual.dtype != torch.float32:
            raise ValueError("dense residual must be a 2D FP32 tensor")
        super().__init__(residual)

    def forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        x_local = x.to(device=self.weight.device, dtype=torch.float32)
        return F.linear(x_local, self.weight).to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_fp32(x).to(x.dtype)


class QJLBaseFP32Expert(nn.Module):
    """The original SwiGLU order with QJL Base and dense FP32 residual."""

    def __init__(self, bases: dict, residuals: dict):
        super().__init__()
        self.bases = nn.ModuleDict(bases)
        self.residuals = nn.ModuleDict(residuals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bg, bu = _pair(self.bases["gate"], self.bases["up"], x)
        g = bg + self.residuals["gate"](x)
        u = bu + self.residuals["up"](x)
        middle = F.silu(g) * u
        return self.bases["down"](middle) + self.residuals["down"](middle)


def save_checkpoint(payload: dict, path: Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    base_packed = base_norms = residual_bytes = 0
    for experts in payload["layers"].values():
        for parts in experts.values():
            for part in parts.values():
                base = part["base"]
                residual = part["residual"]
                if residual.dtype != torch.float32 or residual.ndim != 2:
                    raise ValueError("checkpoint residual must be a dense FP32 matrix")
                base_packed += base["packed"].numel() * base["packed"].element_size()
                base_norms += base["norms"].numel() * base["norms"].element_size()
                residual_bytes += residual.numel() * residual.element_size()
    total = path.stat().st_size
    return {"checkpoint_bytes": total, "base_packed_bytes": base_packed,
            "base_norm_bytes": base_norms, "fp32_residual_bytes": residual_bytes,
            "metadata_bytes": total - base_packed - base_norms - residual_bytes}


def _residual_location(requested: str, payload: dict, model) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("FP32 residual CUDA placement requested without CUDA")
        return torch.device("cuda:0")
    if requested != "auto":
        raise ValueError(f"unknown residual device: {requested}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        return torch.device("cpu")
    dense_bytes = sum(part["residual"].numel() * 4
                      for experts in payload["layers"].values() for parts in experts.values()
                      for part in parts.values())
    free, _ = torch.cuda.mem_get_info(0)
    # Keep room for packed Base, projection banks, and inference activations.
    return torch.device("cuda:0" if dense_bytes + 2 * 2**30 < free else "cpu")


def load_checkpoint(model, path: Path, backend="triton", residual_device="auto") -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("wrong QJL-Base/FP32-Residual checkpoint format")
    config = payload["config"]
    location = _residual_location(residual_device, payload, model)
    banks = {}
    model._base_qjl_projection_banks = banks
    model._fp32_residual_location = str(location)
    resident_devices = set()
    for layer, experts_data in payload["layers"].items():
        block = model.model.layers[int(layer)].mlp
        if len(block.experts) != 60:
            raise ValueError("expected original 60-expert Qwen MoE block")
        device = next(block.parameters()).device
        if backend == "triton" and device.type != "cuda":
            raise RuntimeError("Triton Base QJL requires the routed expert block on CUDA")
        resident_devices.add(device)
        old_router, old_shared, old_top_k = (block.gate,
            getattr(block, "shared_expert", None), block.top_k)
        converted = []
        for expert_id in range(60):
            if expert_id not in experts_data:
                converted.append(None)
                continue
            bases, residuals = {}, {}
            for key in PROJECTIONS:
                part = experts_data[expert_id][key]
                base, residual = part["base"], part["residual"]
                expected_shape = (
                    config.get("output_dims", {"gate": 1408, "up": 1408,
                                               "down": 2048})[key],
                    config.get("input_dims", {"gate": 2048, "up": 2048,
                                              "down": 1408})[key])
                input_dim = expected_shape[1]
                if (tuple(residual.shape) != expected_shape or residual.dtype != torch.float32
                        or not residual.isfinite().all()):
                    raise ValueError(f"invalid FP32 residual at {layer}/{expert_id}/{key}")
                if (base["input_dim"] != input_dim or base["d"] != input_dim * config["base_factor"]
                        or base["packed"].shape != (expected_shape[0], (base["d"] + 7) // 8)
                        or base["packed"].dtype != torch.uint8
                        or base["norms"].shape != (expected_shape[0],)
                        or base["norms"].dtype != torch.float32):
                    raise ValueError(f"invalid QJL Base at {layer}/{expert_id}/{key}")
                if input_dim not in banks:
                    banks[input_dim] = ProjectionBank(base["d"], config["base_seed"])
                bank = banks[input_dim]
                bases[key] = PackedResidual(base["packed"].to(device),
                                            base["norms"].to(device), input_dim,
                                            bank.d, bank, backend)
                residuals[key] = FP32Residual(residual.to(location))
            converted.append(QJLBaseFP32Expert(bases, residuals))
        if all(expert is not None for expert in converted):
            _install(block, converted)
        else:
            for expert_id, expert in enumerate(converted):
                if expert is not None:
                    block.experts[expert_id] = expert
        if (block.gate is not old_router or getattr(block, "shared_expert", None) is not old_shared
                or block.top_k != old_top_k):
            raise AssertionError("router, shared expert, or top-k changed")
    for device in resident_devices:
        for input_dim, bank in banks.items():
            bank.get(input_dim, device)
    return payload


def resident_breakdown(model) -> dict:
    base_bits = base_norms = dense_gpu = dense_cpu = 0
    for layer in model.model.layers:
        block = layer.mlp
        experts = getattr(block, "_compact_experts", None) or block.experts
        for expert in experts:
            if not isinstance(expert, QJLBaseFP32Expert):
                continue
            for base in expert.bases.values():
                if base.packed.is_cuda:
                    base_bits += base.packed.numel() * base.packed.element_size()
                    base_norms += base.norms.numel() * base.norms.element_size()
            for residual in expert.residuals.values():
                size = residual.weight.numel() * residual.weight.element_size()
                if residual.weight.is_cuda:
                    dense_gpu += size
                else:
                    dense_cpu += size
    projection = sum(bank.resident_bytes("cuda") for bank in
                     getattr(model, "_base_qjl_projection_banks", {}).values())
    return {"base_packed_gpu_bytes": base_bits, "base_norm_gpu_bytes": base_norms,
            "fp32_residual_gpu_bytes": dense_gpu, "fp32_residual_cpu_bytes": dense_cpu,
            "base_projection_gpu_bytes": projection, "metadata_gpu_bytes": 0,
            "gpu_resident_compact_total_bytes": base_bits + base_norms + dense_gpu + projection,
            "fp32_residual_location": getattr(model, "_fp32_residual_location", None)}
