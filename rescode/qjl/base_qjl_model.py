"""Independent QJL Base plus unchanged QJL residual for Qwen routed experts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook_model import FORMAT as BASELINE_FORMAT
from .codebook_model import CodebookExpert, bank_for, restore_base
from .core import PackedResidual, ProjectionBank, pack_residual, qjl_triton_fused_gate_up
from .exponent_model import _install

FORMAT = "rescode-dual-qjl-base-residual-v1"
PROJECTIONS = ("gate", "up", "down")


def _pair(gate: PackedResidual, up: PackedResidual, x: torch.Tensor):
    if (gate.backend == up.backend == "triton" and gate.d == up.d
            and gate.packed.shape == up.packed.shape and gate.bank is up.bank):
        p = gate.bank.get(gate.input_dim, x.device)
        z = x.reshape(-1, gate.input_dim).to(p.dtype) @ p
        return qjl_triton_fused_gate_up(z, gate, up, x.dtype)
    return gate(x), up(x)


class DualQJLExpert(nn.Module):
    """The same SwiGLU path as CodebookExpert, with two packed QJL branches."""

    def __init__(self, bases: dict, residuals: dict):
        super().__init__()
        self.bases = nn.ModuleDict(bases)
        self.residuals = nn.ModuleDict(residuals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bg, bu = _pair(self.bases["gate"], self.bases["up"], x)
        rg, ru = _pair(self.residuals["gate"], self.residuals["up"], x)
        g, u = bg + rg, bu + ru
        middle = F.silu(g) * u
        return self.bases["down"](middle) + self.residuals["down"](middle)


def baseline_payload(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != BASELINE_FORMAT:
        raise ValueError("baseline checkpoint has the wrong format")
    config = payload["config"]
    if config.get("base_method") != "exponent-mantissa":
        raise ValueError("baseline must use the existing exponent-mantissa Base")
    if config.get("qjl_d_factor") != 2:
        raise ValueError("baseline Residual QJL must have D_R = 2 x input_dim")
    for layer, experts in payload["layers"].items():
        if sorted(experts) != list(range(60)):
            raise ValueError(f"baseline layer {layer} must contain expert IDs 0..59")
        for expert_id, parts in experts.items():
            if set(parts) != set(PROJECTIONS):
                raise ValueError(f"baseline layer {layer} expert {expert_id} lacks projections")
            for key, part in parts.items():
                rows, input_dim = part["base"]["shape"]
                if (rows, input_dim) != ((2048, 1408) if key == "down" else (1408, 2048)):
                    raise ValueError(f"wrong {key} weight shape in baseline")
                if part["packed"].shape != (rows, (2 * input_dim + 7) // 8):
                    raise ValueError("baseline residual packed shape mismatch")
                if part["norms"].shape != (rows,) or part["norms"].dtype != torch.float32:
                    raise ValueError("baseline residual row norms must be FP32")
    return payload


def residual_digest(payload: dict) -> str:
    """Hash every packed residual byte and FP32 row norm in stable order."""
    digest = hashlib.sha256()
    for layer in sorted(payload["layers"]):
        for expert in sorted(payload["layers"][layer]):
            for key in PROJECTIONS:
                part = payload["layers"][layer][expert][key]
                digest.update(part["packed"].contiguous().numpy().tobytes())
                digest.update(part["norms"].contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def encode_qjl_base(base_state: dict, bank: ProjectionBank, device,
                    row_chunk: int = 128) -> tuple[dict, PackedResidual]:
    base = restore_base(base_state, device, "exponent-mantissa")
    decoded = base.decode()
    rows, input_dim = decoded.shape
    if decoded.dtype != torch.float32 or not torch.isfinite(decoded).all():
        raise ValueError("decoded Base must be finite FP32")
    packed, norms = pack_residual(decoded, bank, row_chunk=row_chunk)
    if packed.shape != (rows, (bank.d + 7) // 8) or norms.shape != (rows,):
        raise AssertionError("Base QJL packed shape mismatch")
    reference_norms = torch.linalg.vector_norm(decoded, dim=1)
    if not torch.equal(norms, reference_norms):
        raise AssertionError("Base QJL row norms changed")
    state = {"packed": packed.cpu(), "norms": norms.cpu(),
             "input_dim": input_dim, "d": bank.d}
    module = PackedResidual(packed, norms, input_dim, bank.d, bank, "triton")
    return state, module


def packed_bytes(part: dict) -> int:
    return sum(part[key].numel() * part[key].element_size() for key in ("packed", "norms"))


def save_checkpoint(payload: dict, path: Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    projections = [part for experts in payload["layers"].values() for parts in experts.values()
                   for part in parts.values()]
    def tensor_bytes(branch, key):
        return sum(part[branch][key].numel() * part[branch][key].element_size()
                   for part in projections)
    base_packed = tensor_bytes("base", "packed")
    base_norms = tensor_bytes("base", "norms")
    residual_packed = tensor_bytes("residual", "packed")
    residual_norms = tensor_bytes("residual", "norms")
    base_bytes = base_packed + base_norms
    residual_bytes = residual_packed + residual_norms
    total = path.stat().st_size
    return {"checkpoint_bytes": total, "base_packed_and_norm_bytes": base_bytes,
            "residual_packed_and_norm_bytes": residual_bytes,
            "base_packed_bytes": base_packed, "base_norm_bytes": base_norms,
            "residual_packed_bytes": residual_packed,
            "residual_norm_bytes": residual_norms,
            "metadata_bytes": total - base_bytes - residual_bytes}


def load_checkpoint(model, path: Path, backend="triton") -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("wrong dual-QJL checkpoint format")
    config = payload["config"]
    if config["base_seed"] == config["residual_seed"]:
        raise ValueError("Base and residual projection seeds must be independent")
    base_banks, residual_banks = {}, {}
    resident_devices = set()
    model._base_qjl_projection_banks = base_banks
    model._residual_qjl_projection_banks = residual_banks
    for layer, experts_data in payload["layers"].items():
        block = model.model.layers[int(layer)].mlp
        if len(block.experts) != 60:
            raise ValueError("expected original 60-expert Qwen MoE block")
        device = next(block.parameters()).device
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
                b, r = part["base"], part["residual"]
                input_dim = config.get("input_dims", {"gate": 2048, "up": 2048,
                                                       "down": 1408})[key]
                if b["input_dim"] != input_dim or b["d"] != input_dim * config["base_factor"]:
                    raise ValueError("Base QJL dimension mismatch")
                if r["packed"].shape != (b["packed"].shape[0], (2 * input_dim + 7) // 8):
                    raise ValueError("residual QJL dimension mismatch")
                if (b["packed"].dtype != torch.uint8 or r["packed"].dtype != torch.uint8
                        or b["norms"].dtype != torch.float32 or r["norms"].dtype != torch.float32
                        or b["norms"].shape != r["norms"].shape):
                    raise ValueError("QJL packed bits or FP32 row norms mismatch")
                b_bank = bank_for(base_banks, input_dim, config["base_factor"], config["base_seed"])
                r_bank = bank_for(residual_banks, input_dim, 2, config["residual_seed"])
                bases[key] = PackedResidual(b["packed"].to(device), b["norms"].to(device),
                                            input_dim, b_bank.d, b_bank, backend)
                residuals[key] = PackedResidual(r["packed"].to(device), r["norms"].to(device),
                                                input_dim, r_bank.d, r_bank, backend)
            converted.append(DualQJLExpert(bases, residuals))
        if all(expert is not None for expert in converted):
            _install(block, converted)
        else:
            for expert_id, expert in enumerate(converted):
                if expert is not None:
                    block.experts[expert_id] = expert
        if block.gate is not old_router or getattr(block, "shared_expert", None) is not old_shared or block.top_k != old_top_k:
            raise AssertionError("router, shared expert, or top-k changed")
    for device in resident_devices:
        for input_dim, bank in base_banks.items():
            bank.get(input_dim, device)
        for input_dim, bank in residual_banks.items():
            bank.get(input_dim, device)
    return payload


def gpu_resident_breakdown(model) -> dict:
    base_bits = base_norms = residual_bits = residual_norms = 0
    for layer in model.model.layers:
        block = layer.mlp
        experts = getattr(block, "_compact_experts", None) or block.experts
        for expert in experts:
            if not isinstance(expert, DualQJLExpert):
                continue
            for modules, name in ((expert.bases, "base"), (expert.residuals, "residual")):
                for part in modules.values():
                    if part.packed.is_cuda:
                        n = part.packed.numel() * part.packed.element_size()
                        m = part.norms.numel() * part.norms.element_size()
                        if name == "base":
                            base_bits += n; base_norms += m
                        else:
                            residual_bits += n; residual_norms += m
    base_proj = sum(bank.resident_bytes("cuda") for bank in
                    getattr(model, "_base_qjl_projection_banks", {}).values())
    residual_proj = sum(bank.resident_bytes("cuda") for bank in
                        getattr(model, "_residual_qjl_projection_banks", {}).values())
    return {"base_packed_gpu_bytes": base_bits, "base_norm_gpu_bytes": base_norms,
            "residual_packed_gpu_bytes": residual_bits,
            "residual_norm_gpu_bytes": residual_norms,
            "base_projection_gpu_bytes": base_proj,
            "residual_projection_gpu_bytes": residual_proj,
            "gpu_resident_compact_total_bytes": (base_bits + base_norms + residual_bits
                                                 + residual_norms + base_proj + residual_proj)}
