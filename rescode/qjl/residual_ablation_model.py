"""Runtime modules and sharded checkpoint loading for the residual A–E study."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook_model import restore_base
from .base_qjl_model import _pair
from .core import ExactResidual, PackedResidual, ProjectionBank
from .exponent_model import _install
from .residual_ablation_methods import PackedINT2Residual, SharedLowRankResidual


PROJECTIONS = ("gate", "up", "down")
FORMAT = "qjl-decomposition-residual-ablation-v1"


class ZeroResidual(nn.Module):
    def __init__(self, rows: int):
        super().__init__()
        self.rows = rows

    def forward(self, x):
        return x.new_zeros((*x.shape[:-1], self.rows))


class FP32Residual(ExactResidual):
    def forward(self, x):
        return F.linear(x.to(self.weight.device, torch.float32), self.weight).to(
            device=x.device, dtype=x.dtype)


class ResidualAblationExpert(nn.Module):
    def __init__(self, bases: dict, residuals: dict):
        super().__init__()
        self.bases = nn.ModuleDict(bases)
        self.residuals = nn.ModuleDict(residuals)

    def projection(self, key: str, x: torch.Tensor) -> torch.Tensor:
        return self.bases[key](x) + self.residuals[key](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.residuals["gate"], PackedResidual) and isinstance(
                self.residuals["up"], PackedResidual):
            rg, ru = _pair(self.residuals["gate"], self.residuals["up"], x)
            gate = self.bases["gate"](x) + rg
            up = self.bases["up"](x) + ru
        else:
            gate = self.projection("gate", x)
            up = self.projection("up", x)
        return self.projection("down", F.silu(gate) * up)


def build_expert(parts: dict, groups: dict, case: str, factor: int, seed: int,
                 device: torch.device, backend: str, fp32_device: torch.device,
                 bank_cache: dict, shared_cache: dict | None = None) -> ResidualAblationExpert:
    bases, residuals = {}, {}
    for key in PROJECTIONS:
        part = parts[key]
        state = part["base"]
        bases[key] = restore_base(state, device, "exponent-mantissa")
        rows, cols = state["shape"]
        item = part.get("residual")
        if case == "A":
            residuals[key] = ZeroResidual(rows)
        elif case == "B":
            residuals[key] = FP32Residual(item.to(fp32_device))
        elif case == "C":
            cache_key = (cols, factor)
            if cache_key not in bank_cache:
                bank_cache[cache_key] = ProjectionBank(cols * factor, seed)
            bank = bank_cache[cache_key]
            residuals[key] = PackedResidual(item["packed"].to(device), item["norms"].to(device),
                                            cols, cols * factor, bank, backend)
        elif case == "D":
            group_id = int(item["group_id"])
            group = groups[group_id][key]
            cache_key = (group_id, key)
            if shared_cache is not None and cache_key in shared_cache:
                a, b = shared_cache[cache_key]
            else:
                a, b = group["a"].to(device), group["b"].to(device)
                if shared_cache is not None:
                    shared_cache[cache_key] = (a, b)
            residuals[key] = SharedLowRankResidual(
                a, b, group["codes"][int(item["code_index"])].to(device),
                float(group["alpha"]))
        elif case == "E":
            residuals[key] = PackedINT2Residual(item, device)
        else:
            raise ValueError(f"unknown case: {case}")
    return ResidualAblationExpert(bases, residuals)


def _state_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_state_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_state_bytes(v) for v in obj)
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    return 0


def tensor_bytes(payload: dict) -> dict:
    base = sum(_state_bytes(p["base"]) for ex in payload["experts"].values()
               for p in ex.values())
    residual = sum(_state_bytes(p.get("residual")) for ex in payload["experts"].values()
                   for p in ex.values())
    shared = _state_bytes(payload.get("groups", {}))
    return {"base_bytes": base, "residual_bytes": residual + shared,
            "shared_factor_bytes": shared}


def load_variant(model, folder: Path, config: dict, backend="triton",
                 fp32_residual_device="auto") -> dict:
    """Load original nonexpert weights and replace routed experts only."""
    folder = Path(folder)
    case, factor, seed = config["case"], config["factor"], config["qjl_seed"]
    layer_files = sorted(folder.glob("layer_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not layer_files:
        raise FileNotFoundError(f"no layer checkpoints in {folder}")
    if len(layer_files) != len(config["layers"]):
        raise ValueError("incomplete layer checkpoint set")
    bank_cache, size = {}, {"base_bytes": 0, "residual_bytes": 0,
                            "shared_factor_bytes": 0, "checkpoint_bytes": 0}
    model._residual_ablation_banks = bank_cache
    # B is an exact FP32 upper bound.  On a single GPU it may not fit, so
    # auto placement follows the existing Base-QJL/FP32-residual experiment.
    if case == "B" and fp32_residual_device == "auto":
        count = 60 * len(layer_files)
        needed = count * 3 * 1408 * 2048 * 4
        free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        fp32_device = torch.device("cuda:0" if needed + 2 * 2**30 < free else "cpu")
    else:
        fp32_device = torch.device("cpu" if fp32_residual_device == "auto" else fp32_residual_device)
    for path in layer_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta_path = path.with_name(path.stem + ".meta.json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"missing checkpoint source metadata: {meta_path}")
        meta = json.loads(meta_path.read_text())
        if (payload.get("format") != FORMAT or payload.get("config_digest") != config["digest"]
                or payload.get("case") != case or payload.get("factor") != factor
                or meta.get("config_digest") != config["digest"]
                or meta.get("source_sha256") != payload.get("source_sha256")
                or meta.get("base_sha256") != payload.get("base_sha256")
                or meta.get("checkpoint_bytes") != path.stat().st_size):
            raise ValueError(f"incompatible layer checkpoint: {path}")
        layer = int(payload["layer"])
        block = model.model.layers[layer].mlp
        if len(block.experts) != 60:
            raise ValueError("original model must have 60 routed experts")
        device = next(block.parameters()).device
        if backend == "triton" and case == "C" and device.type != "cuda":
            raise RuntimeError("QJL Triton requires routed expert block on CUDA")
        old_router, old_shared, old_topk = block.gate, getattr(block, "shared_expert", None), block.top_k
        experts, shared_cache = [], {}
        for expert_id in range(60):
            if expert_id not in payload["experts"]:
                raise ValueError("full evaluation requires all 60 routed experts")
            experts.append(build_expert(payload["experts"][expert_id], payload.get("groups", {}),
                                        case, factor, seed, device, backend, fp32_device,
                                        bank_cache, shared_cache))
        _install(block, experts)
        if (block.gate is not old_router or getattr(block, "shared_expert", None) is not old_shared
                or block.top_k != old_topk):
            raise AssertionError("routing topology changed")
        amounts = tensor_bytes(payload)
        for key, value in amounts.items():
            size[key] += value
        size["checkpoint_bytes"] += path.stat().st_size
        del payload
    for layer in config["layers"]:
        block = model.model.layers[layer].mlp
        device = next(block.parameters()).device
        for (cols, _), bank in bank_cache.items():
            bank.get(cols, device)
    size["projection_bank_gpu_bytes"] = sum(b.resident_bytes("cuda") for b in bank_cache.values())
    seen, resident = set(), 0
    for layer in config["layers"]:
        for expert in model.model.layers[layer].mlp._compact_experts:
            for buffer in expert.buffers():
                if buffer.is_cuda:
                    pointer = buffer.untyped_storage().data_ptr()
                    if pointer not in seen:
                        seen.add(pointer)
                        resident += buffer.numel() * buffer.element_size()
    size["expert_gpu_resident_bytes"] = resident
    size["gpu_resident_compact_total_bytes"] = resident + size["projection_bank_gpu_bytes"]
    size["fp32_residual_device"] = str(fp32_device)
    return size
