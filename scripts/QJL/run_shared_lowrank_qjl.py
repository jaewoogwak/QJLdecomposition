#!/usr/bin/env python3
"""ResCode grouping + shared BF16 Base + expert low rank + existing QJL Tail.

Analyze uses only selected safetensors after grouping is supplied. Compress writes
independent A/B/C expert checkpoints. Eval overlays one checkpoint on the original
Qwen model, preserving its router, shared expert and SwiGLU implementation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rescode.merging.clustering import group_experts_by_clustering
from rescode.qjl.core import PackedResidual, ProjectionBank, pack_residual

FORMAT = "qwen-shared-base-lowrank-qjl-v1"
MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
PROJECTIONS = ("gate", "up", "down")
CASES = ("A", "B", "C")
DEFAULT_TASKS = "arc_easy,arc_challenge"
C4_FILENAME = "c4-train.00000-of-01024.json"
C4_REPO_FILE = f"en/{C4_FILENAME}.gz"
C4_SHA256 = "8ef8d75b0e045dec4aa5123a671b4564466b0707086a7ed1ba8721626dfffbc9"
DEFAULT_C4 = ROOT / "rescode/data" / C4_FILENAME


def ensure_c4(path: Path) -> Path:
    """Download and verify the ResCode C4 shard only when it is missing."""
    path = path.expanduser().resolve()
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        if not isinstance(first.get("text"), str) or not first["text"]:
            raise ValueError(f"calibration JSON lacks a nonempty text field: {path}")
        return path
    if path.name != C4_FILENAME:
        raise FileNotFoundError(f"calibration file missing: {path}; automatic download "
                                f"is available for {C4_FILENAME}")
    from huggingface_hub import hf_hub_download

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[shared-lowrank-qjl] downloading allenai/c4/{C4_REPO_FILE}", flush=True)
    compressed = Path(hf_hub_download(
        repo_id="allenai/c4", repo_type="dataset", filename=C4_REPO_FILE,
        local_dir=str(path.parent)))
    digest = hashlib.sha256()
    with compressed.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != C4_SHA256:
        raise ValueError("downloaded C4 shard SHA256 does not match the official file")
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        with temporary.open("r", encoding="utf-8") as handle:
            first = json.loads(handle.readline())
        if not isinstance(first.get("text"), str) or not first["text"]:
            raise ValueError("downloaded C4 JSON lacks a nonempty text field")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    if compressed.parent == path.parent / "en":
        compressed.unlink(missing_ok=True)
    print(f"[shared-lowrank-qjl] C4 ready: {path} ({path.stat().st_size} bytes)",
          flush=True)
    return path


def selected(value: str, total: int) -> list[int]:
    result = list(range(total)) if value == "all" else [int(x) for x in value.split(",")]
    if not result or len(set(result)) != len(result) or any(x < 0 or x >= total for x in result):
        raise ValueError(f"invalid selection {value!r} for range(0,{total})")
    return sorted(result)


def finite_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float | None:
    a = float(numerator)
    b = float(denominator)
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("nonfinite metric input")
    return a / b if b > 0 else None


def squared_norm(x: torch.Tensor) -> torch.Tensor:
    return x.double().square().sum()


def output_metrics(actual: torch.Tensor, target: torch.Tensor) -> dict:
    a, b = actual.float().reshape(-1), target.float().reshape(-1)
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("output shape mismatch or nonfinite output")
    a2, b2 = squared_norm(a), squared_norm(b)
    error2 = squared_norm(a - b)
    cosine = finite_ratio((a.double() * b.double()).sum(), (a2 * b2).sqrt())
    return {"relative_l2": math.sqrt(error2 / b2) if b2 > 0 else None,
            "cosine": cosine, "output_norm_squared_ratio": finite_ratio(a2, b2),
            "one_minus_relative_error_squared": 1 - error2 / b2 if b2 > 0 else None}


def saliency_average(weights: dict[int, torch.Tensor], saliency: torch.Tensor,
                     delta: float = 1e-3) -> torch.Tensor:
    """Match ResCode's alpha_i=(a_i+delta)/sum_j(a_j+delta)."""
    if not weights or delta <= 0:
        raise ValueError("nonempty weights and positive saliency delta required")
    members = sorted(weights)
    shape = weights[members[0]].shape
    if any(weights[i].shape != shape for i in members):
        raise ValueError("group projection shapes differ")
    alpha = (saliency[members].float() + delta).to(weights[members[0]].device)
    if not torch.isfinite(alpha).all() or (alpha <= 0).any():
        raise ValueError("invalid saliency")
    alpha /= alpha.sum()
    base = torch.zeros_like(weights[members[0]], dtype=torch.float32)
    for i, weight in weights.items():
        base.add_(weight.float(), alpha=float(alpha[members.index(i)]))
    return base.to(torch.bfloat16).contiguous()


def groups_from_labels(labels: torch.Tensor, expert_ids: list[int]) -> list[list[int]]:
    labels = torch.as_tensor(labels, dtype=torch.long).cpu().reshape(-1)
    if any(i >= labels.numel() for i in expert_ids):
        raise ValueError("group labels do not cover selected experts")
    groups = defaultdict(list)
    for expert in expert_ids:
        groups[int(labels[expert])].append(expert)
    result = list(groups.values())
    if sorted(i for group in result for i in group) != expert_ids:
        raise AssertionError("groups are not a partition")
    return result


def groups_from_descriptors(descriptors: torch.Tensor, saliency: torch.Tensor,
                            expert_ids: list[int], num_groups: int,
                            singleton_ratio: float, linkage: str = "average") -> list[list[int]]:
    """Use ResCode's saliency singleton rule and output descriptor clustering."""
    if not 0 <= singleton_ratio < 1 or not 1 <= num_groups <= len(expert_ids):
        raise ValueError("invalid grouping configuration")
    if descriptors.ndim != 2 or descriptors.shape[0] <= max(expert_ids):
        raise ValueError("missing expert output descriptors")
    count = min(int(round(len(expert_ids) * singleton_ratio)), num_groups - 1)
    pinned = sorted(expert_ids, key=lambda i: (-float(saliency[i]), i))[:count]
    rest = [i for i in expert_ids if i not in set(pinned)]
    groups = [[i] for i in pinned]
    if rest:
        k = min(num_groups - count, len(rest))
        _, labels = group_experts_by_clustering(
            model="qwen", num_groups=k, cluster="hierarchical", linkage=linkage,
            hierarchical_stopping_metric="silhouette", num_experts=len(rest),
            experts=descriptors[rest].float())
        groups += groups_from_labels(labels, list(range(len(rest))))
        groups = groups[:count] + [[rest[j] for j in group] for group in groups[count:]]
    return groups


def _read_cache(path: Path) -> dict:
    if path.is_dir():
        data = {}
        for name in ("group_state_dict", "saliency_state_dict", "descriptor_state_dict",
                     "routed_samples"):
            file = path / f"{name}.pt"
            if file.is_file():
                data[name] = torch.load(file, map_location="cpu", weights_only=False)
        return data
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError("grouping cache must contain a dictionary")
    return data


def _layer_value(mapping: dict, layer: int):
    key = f"model.layers.{layer}.mlp"
    for candidate in (key, layer, str(layer)):
        if candidate in mapping:
            return mapping[candidate]
    raise KeyError(f"cache lacks layer {layer}")


def grouping_for_layer(cache: dict, layer: int, expert_ids: list[int],
                       num_groups: int, singleton_ratio: float, linkage: str) -> tuple[list[list[int]], torch.Tensor]:
    saliencies = cache.get("saliency_state_dict", cache.get("saliency", {}))
    saliency = torch.as_tensor(_layer_value(saliencies, layer)).float().cpu()
    if saliency.numel() <= max(expert_ids) or not torch.isfinite(saliency).all() or (saliency < 0).any():
        raise ValueError(f"invalid ResCode saliency for layer {layer}")
    labels = cache.get("group_state_dict", cache.get("groups"))
    if labels:
        groups = groups_from_labels(_layer_value(labels, layer), expert_ids)
    else:
        descriptors = cache.get("descriptor_state_dict", cache.get("descriptors", {}))
        groups = groups_from_descriptors(torch.as_tensor(_layer_value(descriptors, layer)),
                                         saliency, expert_ids, num_groups,
                                         singleton_ratio, linkage)
    return groups, saliency


def routed_or_synthetic(cache: dict, layer: int, expert: int, projection: str,
                        input_dim: int, samples: int, seed: int,
                        device: torch.device) -> tuple[torch.Tensor, str]:
    store = cache.get("routed_samples", {})
    if "train" in store:
        store = store["train"]
    try:
        layer_store = _layer_value(store, layer)
        entry = layer_store[projection][expert]
        x = entry[0] if isinstance(entry, (tuple, list)) else entry
        if x.ndim == 2 and x.shape[1] == input_dim and x.shape[0]:
            return x[:samples].to(device=device, dtype=torch.float32), "routed"
    except (KeyError, IndexError, TypeError):
        pass
    generator = torch.Generator(device="cpu").manual_seed(
        seed + 1009 * layer + 37 * expert + PROJECTIONS.index(projection))
    return torch.randn(samples, input_dim, generator=generator).to(device), "synthetic"


def svd_factors(residual: torch.Tensor, max_rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 QR SVD avoids the Jacobi energy drift seen on large Qwen matrices."""
    if residual.dtype != torch.float32 or residual.ndim != 2:
        raise ValueError("SVD input must be FP32 matrix")
    if max_rank == 0:
        return residual.new_zeros((residual.shape[0], 0)), residual.new_zeros((0, residual.shape[1]))
    u, singular, vh = torch.linalg.svd(
        residual, full_matrices=False, driver="gesvd" if residual.is_cuda else None)
    return (u[:, :max_rank] * singular[:max_rank]).contiguous(), vh[:max_rank].contiguous()


def rank_parts(residual: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
               rank: int) -> tuple[dict, torch.Tensor, dict]:
    """Oracle energy uses FP32 SVD; stored-tail uses reconstructed BF16 factors."""
    if rank < 0 or rank > left.shape[1]:
        raise ValueError("rank outside cached SVD range")
    oracle = left[:, :rank] @ right[:rank]
    oracle_tail = residual - oracle
    if not torch.allclose(residual, oracle + oracle_tail, atol=2e-5, rtol=1e-5):
        raise AssertionError("FP32 oracle residual reconstruction failed")
    r2, l2, e2 = map(squared_norm, (residual, oracle, oracle_tail))
    if r2 > 0 and abs(float(l2 + e2 - r2)) > 3e-4 * float(r2):
        raise ArithmeticError("FP32 SVD energy decomposition failed")
    u = left[:, :rank].to(torch.bfloat16).cpu().contiguous()
    v = right[:rank].to(torch.bfloat16).cpu().contiguous()
    restored = u.to(residual.device).float() @ v.to(residual.device).float()
    tail = residual - restored
    stored_l2, stored_e2 = map(squared_norm, (restored, tail))
    stats = {"oracle_eta_uv": finite_ratio(l2, r2),
             "oracle_eta_e": finite_ratio(e2, r2),
             "stored_factor_energy_ratio": finite_ratio(stored_l2, r2),
             "stored_tail_energy_ratio": finite_ratio(stored_e2, r2),
             "stored_factor_reconstruction_rel_l2": (
                 math.sqrt(squared_norm(oracle - restored) / r2) if r2 > 0 else None)}
    return {"u": u, "v": v}, tail, stats


def pack_tail(weight: torch.Tensor, bank: ProjectionBank) -> dict:
    packed, norms = pack_residual(weight, bank)
    if packed.shape != (weight.shape[0], (bank.d + 7) // 8) or norms.shape != (weight.shape[0],):
        raise AssertionError("QJL packed shape mismatch")
    if not torch.isfinite(norms).all():
        raise ValueError("nonfinite QJL row norm")
    return {"packed": packed.cpu().contiguous(), "norms": norms.cpu().contiguous()}


def packed_output(part: dict, x: torch.Tensor, bank: ProjectionBank,
                  backend: str) -> torch.Tensor:
    module = PackedResidual(part["packed"].to(x.device), part["norms"].to(x.device),
                            x.shape[-1], bank.d, bank, backend)
    return module(x)


class WeightReader:
    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.stack = ExitStack()
        self.handles = {}

    def close(self):
        self.stack.close()

    def get(self, layer: int, expert: int, projection: str) -> torch.Tensor:
        name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
        shard = self.index[name]
        if shard not in self.handles:
            self.handles[shard] = self.stack.enter_context(
                safe_open(str(self.root / shard), framework="pt", device="cpu"))
        weight = self.handles[shard].get_tensor(name)
        if weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise ValueError(f"expected BF16 matrix: {name}")
        return weight.contiguous()


def _load_hf_model(model_dir: Path, memory_fraction: float):
    from transformers import AutoModelForCausalLM
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * memory_fraction)
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    return AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True, max_memory=max_memory).eval()


def prepare_cache(args, model_dir: Path, layers: list[int]) -> dict:
    if args.grouping_cache:
        cache = _read_cache(args.grouping_cache)
        if cache and all(str(key).startswith("model.layers.") for key in cache):
            cache = {"group_state_dict": cache}
        if args.saliency_cache:
            source = _read_cache(args.saliency_cache)
            cache["saliency_state_dict"] = source.get("saliency_state_dict", source)
        if args.routed_input_cache:
            source = _read_cache(args.routed_input_cache)
            cache["routed_samples"] = source.get("routed_samples", source)
        return cache
    calibration_data = ensure_c4(args.calibration_data)
    from transformers import AutoTokenizer
    from rescode.evaluation import get_calib_dataloder
    from rescode.merging.grouping_qwen import ExpertsGrouperForQwen2MoE
    from rescode.qjl.calibration import collect_projection_samples

    model = _load_hf_model(model_dir, args.gpu_memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataloader = get_calib_dataloder(
        dataset="c4", tokenizer=tokenizer, max_block_size=args.calibration_seqlen,
        n_blocks_for_stat=args.calibration_blocks, batch_size=args.calibration_batch_size,
        num_workers=0, seed=args.seed, data_path=str(calibration_data))
    grouper = ExpertsGrouperForQwen2MoE(
        config=model.config, similarity_base="rescode", start_layer=min(layers),
        cluster="hierarchical", linkage=args.linkage, descriptor_tokens="all")
    grouper.sparse_layer_indices = layers
    grouper.reset_all()
    grouper.compute_saliency_and_descriptors(model, dataloader)
    grouper.group_experts_by_clustering_rescode(
        model, dataloader, args.num_groups,
        exclude_top_percent=100 * args.singleton_ratio, excl_mode="per-layer")
    # Rebuild the deterministic calibration loader because the grouper consumed it.
    dataloader = get_calib_dataloder(
        dataset="c4", tokenizer=tokenizer, max_block_size=args.calibration_seqlen,
        n_blocks_for_stat=args.calibration_blocks, batch_size=args.calibration_batch_size,
        num_workers=0, seed=args.seed, data_path=str(calibration_data))
    routed, _, _ = collect_projection_samples(
        model, dataloader, "qwen", layers,
        max_train=args.diagnostic_samples, max_heldout=0)
    cache = {"group_state_dict": grouper.group_state_dict(),
             "saliency_state_dict": grouper.saliency_state_dict(),
             "descriptor_state_dict": grouper.descriptor_state_dict(),
             "routed_samples": routed,
             "metadata": {"num_groups": args.num_groups,
                          "singleton_ratio": args.singleton_ratio,
                          "source": "ResCode expert-output descriptors and saliency"}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.output_dir / "grouping_cache.pt")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache


def _tensor_bytes(value) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def checkpoint_components(layer_payload: dict) -> dict:
    base = _tensor_bytes(layer_payload["bases"])
    singleton = _tensor_bytes(layer_payload["singletons"])
    full = factor = packed = norms = 0
    for projections in layer_payload["experts"].values():
        for part in projections.values():
            full += _tensor_bytes(part.get("full"))
            factor += _tensor_bytes(part.get("u")) + _tensor_bytes(part.get("v"))
            qjl = part.get("qjl", {})
            packed += _tensor_bytes(qjl.get("packed"))
            norms += _tensor_bytes(qjl.get("norms"))
    return {"shared_base_bytes": base, "singleton_bf16_bytes": singleton,
            "other_full_bf16_bytes": full, "lowrank_factor_bytes": factor,
            "qjl_packed_bytes": packed, "qjl_row_norm_bytes": norms}


def _save_layer(path: Path, payload: dict) -> tuple[int, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path.stat().st_size, checkpoint_components(payload)


class SharedBase(nn.Module):
    def __init__(self, weights: dict, device: torch.device):
        super().__init__()
        self.weights = nn.ParameterDict({key: nn.Parameter(value.to(device), requires_grad=False)
                                         for key, value in weights.items()})


class CompressedExpert(nn.Module):
    def __init__(self, base: SharedBase, parts: dict, bank: ProjectionBank,
                 backend: str, device: torch.device):
        super().__init__()
        self.base = base
        self.full = nn.ParameterDict()
        self.u = nn.ParameterDict()
        self.v = nn.ParameterDict()
        self.qjl = nn.ModuleDict()
        for key, part in parts.items():
            if "full" in part:
                self.full[key] = nn.Parameter(part["full"].to(device), requires_grad=False)
            if "u" in part:
                self.u[key] = nn.Parameter(part["u"].to(device), requires_grad=False)
                self.v[key] = nn.Parameter(part["v"].to(device), requires_grad=False)
            if "qjl" in part:
                qjl = part["qjl"]
                self.qjl[key] = PackedResidual(
                    qjl["packed"].to(device), qjl["norms"].to(device),
                    part["input_dim"], bank.d, bank, backend)

    def project(self, key: str, x: torch.Tensor) -> torch.Tensor:
        if key in self.full:
            return F.linear(x, self.full[key])
        y = F.linear(x, self.base.weights[key])
        if key in self.u:
            y = y + F.linear(F.linear(x, self.v[key]), self.u[key])
        if key in self.qjl:
            y = y + self.qjl[key](x)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.project("gate", x)
        up = self.project("up", x)
        return self.project("down", F.silu(gate) * up)


def install_checkpoint(model, manifest: Path, backend: str = "triton") -> dict:
    metadata = json.loads(manifest.read_text())
    if metadata.get("format") != FORMAT or metadata.get("case") not in CASES:
        raise ValueError("invalid shared-lowrank-QJL manifest")
    bank = ProjectionBank(int(metadata["qjl_d"]), int(metadata["seed"]))
    model._shared_lowrank_qjl_bank = bank
    for file in metadata["layer_files"]:
        data = torch.load(manifest.parent / file, map_location="cpu", weights_only=False)
        layer = int(data["layer"])
        block = model.model.layers[layer].mlp
        router = block.gate
        shared = getattr(block, "shared_expert", None)
        shared_gate = getattr(block, "shared_expert_gate", None)
        top_k = block.top_k
        device = next(block.parameters()).device
        groups = data["groups"]
        if sorted(i for group in groups for i in group) != sorted(
                list(data["singletons"]) + list(data["experts"])):
            raise ValueError("checkpoint groups do not match selected experts")
        base_modules = {int(cid): SharedBase(weights, device)
                        for cid, weights in data["bases"].items()}
        owner = {i: cid for cid, group in enumerate(groups) for i in group}
        for expert, projections in data["experts"].items():
            expert = int(expert)
            block.experts[expert] = CompressedExpert(
                base_modules[owner[expert]], projections, bank, backend, device)
        for expert, projections in data["singletons"].items():
            expert = int(expert)
            for key, weight in projections.items():
                actual = getattr(block.experts[expert], key + "_proj").weight.detach().cpu()
                if not torch.equal(actual, weight):
                    raise AssertionError(f"singleton {layer}/{expert}/{key} differs from original")
        if (block.gate is not router or getattr(block, "shared_expert", None) is not shared
                or getattr(block, "shared_expert_gate", None) is not shared_gate
                or block.top_k != top_k):
            raise AssertionError("router or shared expert changed")
    return metadata


def _empty_layer(layer: int, groups: list[list[int]]) -> dict:
    return {"layer": layer, "groups": groups, "bases": {},
            "experts": {}, "singletons": {}}


def _put_part(payload: dict, expert: int, projection: str, part: dict) -> None:
    payload["experts"].setdefault(expert, {})[projection] = part


def _singleton_row(layer: int, group_id: int, expert: int, projection: str,
                   rank: int, weight: torch.Tensor, source: str) -> dict:
    w2 = float(squared_norm(weight))
    return {"layer": layer, "group_id": group_id, "expert": expert,
            "projection": projection, "rank": rank, "singleton": True,
            "input_source": source, "weight_squared_norm": w2,
            "residual_squared_norm": 0.0, "tail_squared_norm": 0.0,
            "rho_r_over_w": 0.0, "rho_e_over_w": 0.0, "rho_e_over_r": None,
            "oracle_eta_uv": None, "oracle_eta_e": None,
            "stored_factor_energy_ratio": None,
            "stored_tail_energy_ratio": None,
            "stored_factor_reconstruction_rel_l2": None,
            "qjl_tail_relative_l2": None, "qjl_tail_cosine": None,
            "qjl_tail_output_norm_squared_ratio": None,
            "rhat_relative_l2": None, "rhat_one_minus_relative_error_squared": None,
            **{f"case_{case}_{metric}": value for case in CASES
               for metric, value in (("relative_l2", 0.0), ("cosine", 1.0))}}


def _diagnostic_row(layer: int, group_id: int, expert: int, projection: str,
                    rank: int, weight: torch.Tensor, base: torch.Tensor,
                    residual: torch.Tensor, tail: torch.Tensor,
                    factors: dict, stats: dict, x: torch.Tensor,
                    source: str, a_qjl: torch.Tensor, c_qjl: torch.Tensor) -> dict:
    w2, r2, e2 = map(squared_norm, (weight, residual, tail))
    y_w = F.linear(x, weight)
    y_s = F.linear(x, base)
    y_r = F.linear(x, residual)
    y_l = F.linear(F.linear(x, factors["v"].to(x.device).float()),
                   factors["u"].to(x.device).float())
    y_e = F.linear(x, tail)
    a = output_metrics(y_s + a_qjl, y_w)
    b = output_metrics(y_s + y_l, y_w)
    c = output_metrics(y_s + y_l + c_qjl, y_w)
    qjl = output_metrics(c_qjl, y_e)
    rhat = output_metrics(y_l + c_qjl, y_r)
    row = {"layer": layer, "group_id": group_id, "expert": expert,
           "projection": projection, "rank": rank, "singleton": False,
           "input_source": source, "weight_squared_norm": float(w2),
           "residual_squared_norm": float(r2), "tail_squared_norm": float(e2),
           "rho_r_over_w": math.sqrt(r2 / w2) if w2 > 0 else None,
           "rho_e_over_w": math.sqrt(e2 / w2) if w2 > 0 else None,
           "rho_e_over_r": math.sqrt(e2 / r2) if r2 > 0 else None,
           **stats,
           "qjl_tail_relative_l2": qjl["relative_l2"],
           "qjl_tail_cosine": qjl["cosine"],
           "qjl_tail_output_norm_squared_ratio": qjl["output_norm_squared_ratio"],
           "rhat_relative_l2": rhat["relative_l2"],
           "rhat_one_minus_relative_error_squared": rhat["one_minus_relative_error_squared"],
           "case_A_relative_l2": a["relative_l2"], "case_A_cosine": a["cosine"],
           "case_B_relative_l2": b["relative_l2"], "case_B_cosine": b["cosine"],
           "case_C_relative_l2": c["relative_l2"], "case_C_cosine": c["cosine"]}
    for key, value in row.items():
        if isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"nonfinite diagnostic {key}")
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["projection"], row["rank"], row["input_source"])].append(row)
    result = []
    for (projection, rank, source), items in sorted(buckets.items()):
        w2 = sum(row["weight_squared_norm"] for row in items)
        r2 = sum(row["residual_squared_norm"] for row in items)
        e2 = sum(row["tail_squared_norm"] for row in items)
        means = {}
        for case in CASES:
            values = [row[f"case_{case}_relative_l2"] for row in items
                      if row[f"case_{case}_relative_l2"] is not None]
            means[f"case_{case}_mean_relative_l2"] = sum(values) / len(values) if values else None
        result.append({"projection": projection, "rank": rank,
                       "input_source": source, "expert_count": len(items),
                       "rho_r_over_w": math.sqrt(r2 / w2) if w2 else None,
                       "rho_e_over_w": math.sqrt(e2 / w2) if w2 else None,
                       "rho_e_over_r": math.sqrt(e2 / r2) if r2 else None,
                       "oracle_eta_uv": (sum(row["oracle_eta_uv"] * row["residual_squared_norm"]
                                             for row in items if row["oracle_eta_uv"] is not None)
                                         / r2 if r2 else None),
                       "oracle_eta_e": (sum(row["oracle_eta_e"] * row["residual_squared_norm"]
                                            for row in items if row["oracle_eta_e"] is not None)
                                        / r2 if r2 else None),
                       **means})
    return result


def _plot(out: Path, summaries: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for axis, projection in zip(axes[0], PROJECTIONS):
        for source in sorted({r["input_source"] for r in summaries}):
            rows = sorted((r for r in summaries if r["projection"] == projection
                           and r["input_source"] == source), key=lambda r: r["rank"])
            for case in CASES:
                axis.plot([r["rank"] for r in rows],
                          [r[f"case_{case}_mean_relative_l2"] for r in rows],
                          marker="o", label=f"{case} ({source})")
        axis.set(title=projection, xlabel="rank", ylabel="mean output relative L2")
        axis.grid(alpha=0.2)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rank_vs_output_error.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for axis, projection in zip(axes[0], PROJECTIONS):
        for source in sorted({r["input_source"] for r in summaries}):
            rows = sorted((r for r in summaries if r["projection"] == projection
                           and r["input_source"] == source), key=lambda r: r["rank"])
            axis.plot([r["rank"] for r in rows], [r["rho_e_over_w"] for r in rows],
                      marker="o", label=source)
        axis.set(title=projection, xlabel="rank", ylabel="Tail / original Frobenius norm")
        axis.grid(alpha=0.2)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rank_vs_tail_norm.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    for axis, projection in zip(axes[0], PROJECTIONS):
        for source in sorted({r["input_source"] for r in summaries}):
            rows = sorted((r for r in summaries if r["projection"] == projection
                           and r["input_source"] == source), key=lambda r: r["rank"])
            axis.plot([r["rank"] for r in rows], [r["oracle_eta_uv"] for r in rows],
                      marker="o", label=f"UV ({source})")
            axis.plot([r["rank"] for r in rows], [r["oracle_eta_e"] for r in rows],
                      marker="o", linestyle="--", label=f"Tail ({source})")
        axis.set(title=projection, xlabel="rank", ylabel="FP32 residual energy fraction")
        axis.grid(alpha=0.2)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rank_vs_svd_energy.png", dpi=150)
    plt.close(fig)


def _report(config: dict, summaries: list[dict], checkpoints: list[dict],
            skipped: list[dict]) -> str:
    lines = ["# Shared Base + Low-rank Residual + QJL Tail", "",
             "A uses QJL(R); B uses stored BF16 low-rank factors; C adds QJL(E) "
             "to those same factors. R and E are FP32 weight residuals. "
             "Output error is measured on the same input within each row.", "",
             f"Layers: {config['layers']}; expert IDs: {config['experts']}; "
             f"projections: {config['projections']}; ranks: {config['ranks']}; "
             f"QJL D={config['qjl_d']}.",
             f"Grouping source: {config['grouping_source']}.", "",
             "## Output diagnostics", "",
             "| Projection | Rank | Input | A relative L2 | B relative L2 | "
             "C relative L2 | Tail / W |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in summaries:
        fmt = lambda x: "n/a" if x is None else f"{x:.5f}"
        lines.append(f"| {row['projection']} | {row['rank']} | {row['input_source']} | "
                     f"{fmt(row['case_A_mean_relative_l2'])} | "
                     f"{fmt(row['case_B_mean_relative_l2'])} | "
                     f"{fmt(row['case_C_mean_relative_l2'])} | "
                     f"{fmt(row['rho_e_over_w'])} |")
    lines += ["", "## Checkpoint storage", ""]
    if checkpoints:
        lines += ["| Rank | Case | Checkpoint bytes | Effective bpw | BF16 ratio |",
                  "|---:|---|---:|---:|---:|"]
        for row in checkpoints:
            lines.append(f"| {row['rank']} | {row['case']} | "
                         f"{row['checkpoint_bytes']} | {row['effective_bits_per_weight']:.3f} | "
                         f"{row['bf16_compression_ratio']:.3f}x |")
    else:
        lines.append("Analyze mode does not write checkpoints; storage is measured in compress mode.")
    lines += ["", f"Skipped ranks exceeding the selected matrix dimensions: {len(skipped)}.",
              "Singleton experts remain exact BF16. The checkpoint contains selected "
              "expert tensors only; the original HF backbone, router, and shared expert "
              "remain dependencies for evaluation and are excluded from expert bpw.",
              "When routed calibration inputs are unavailable, `input_source=synthetic` "
              "is recorded. SVD energy metrics are Frobenius quantities; QJL accuracy "
              "uses output vectors and is reported separately.", ""]
    return "\n".join(lines)


@torch.no_grad()
def analyze_or_compress(args) -> dict:
    model_dir = args.model_path.expanduser().resolve()
    model_config = json.loads((model_dir / "config.json").read_text())
    layers = selected(args.layers, model_config["num_hidden_layers"])
    experts = list(range(model_config["num_experts"]))
    if args.max_experts is not None:
        experts = experts[:args.max_experts]
    projections = tuple(x.strip() for x in args.projections.split(","))
    if not projections or len(set(projections)) != len(projections) or any(
            p not in PROJECTIONS for p in projections):
        raise ValueError("invalid projections")
    ranks = sorted({int(x) for x in args.ranks.split(",")})
    if not ranks or any(r < 0 for r in ranks):
        raise ValueError("ranks must be nonnegative")
    if args.qjl_d < 1 or args.diagnostic_samples < 1 or not experts:
        raise ValueError("QJL D, diagnostic samples and expert count must be positive")
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if args.backend == "triton" and device.type != "cuda":
        raise ValueError("Triton diagnostics require CUDA; select --backend reference")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache = prepare_cache(args, model_dir, layers)
    bank = ProjectionBank(args.qjl_d, args.seed)
    records, skipped, layer_files = [], [], defaultdict(list)
    size_totals = defaultdict(lambda: defaultdict(int))
    original_weight_count = 0
    reader = WeightReader(model_dir)
    try:
        for layer in layers:
            groups, saliency = grouping_for_layer(cache, layer, experts,
                                                   args.num_groups,
                                                   args.singleton_ratio,
                                                   args.linkage)
            first = {p: reader.get(layer, experts[0], p).shape for p in projections}
            valid = [r for r in ranks if all(r <= min(first[p]) for p in projections)]
            skipped += [{"layer": layer, "rank": r, "reason": "rank exceeds matrix dimension"}
                        for r in ranks if r not in valid]
            if not valid:
                continue
            layer_payloads = {(rank, case): _empty_layer(layer, groups)
                              for rank in valid for case in CASES} if args.mode == "compress" else {}
            for group_id, members in enumerate(groups):
                if len(members) == 1:
                    expert = members[0]
                    weights = {p: reader.get(layer, expert, p) for p in PROJECTIONS}
                    original_weight_count += sum(w.numel() for w in weights.values())
                    for payload in layer_payloads.values():
                        payload["singletons"][expert] = weights
                    for projection in projections:
                        w = weights[projection].float().to(device)
                        _, source = routed_or_synthetic(cache, layer, expert,
                                                        projection, w.shape[1],
                                                        args.diagnostic_samples,
                                                        args.seed, device)
                        for rank in valid:
                            records.append(_singleton_row(layer, group_id, expert,
                                                          projection, rank, w, source))
                    continue
                for projection in PROJECTIONS:
                    weights = {i: reader.get(layer, i, projection) for i in members}
                    original_weight_count += sum(w.numel() for w in weights.values())
                    if projection not in projections:
                        for payload in layer_payloads.values():
                            for expert, weight in weights.items():
                                _put_part(payload, expert, projection, {"full": weight})
                        continue
                    gpu_weights = {i: w.float().to(device) for i, w in weights.items()}
                    base_bf16 = saliency_average(gpu_weights, saliency,
                                                 args.saliency_delta).cpu()
                    for payload in layer_payloads.values():
                        payload["bases"].setdefault(group_id, {})[projection] = base_bf16
                    base = base_bf16.float().to(device)
                    for expert, w in gpu_weights.items():
                        residual = w - base
                        left, right = svd_factors(residual, max(valid))
                        x, source = routed_or_synthetic(cache, layer, expert,
                                                        projection, w.shape[1],
                                                        args.diagnostic_samples,
                                                        args.seed, device)
                        a_part = pack_tail(residual, bank)
                        a_qjl = packed_output(a_part, x, bank, args.backend)
                        for rank in valid:
                            factors, tail, stats = rank_parts(residual, left, right, rank)
                            restored = (factors["u"].to(device).float() @
                                        factors["v"].to(device).float())
                            if not torch.allclose(w, base + restored + tail,
                                                  rtol=1e-5, atol=2e-5):
                                raise AssertionError("stored-factor tail identity failed")
                            c_part = a_part if rank == 0 else pack_tail(tail, bank)
                            c_qjl = a_qjl if rank == 0 else packed_output(
                                c_part, x, bank, args.backend)
                            records.append(_diagnostic_row(
                                layer, group_id, expert, projection, rank, w,
                                base, residual, tail, factors, stats, x,
                                source, a_qjl, c_qjl))
                            if layer_payloads:
                                _put_part(layer_payloads[(rank, "A")], expert, projection,
                                          {"qjl": a_part, "input_dim": w.shape[1]})
                                b_part = {"input_dim": w.shape[1]}
                                if rank:
                                    b_part.update(factors)
                                _put_part(layer_payloads[(rank, "B")], expert, projection,
                                          b_part)
                                _put_part(layer_payloads[(rank, "C")], expert, projection,
                                          {**b_part, "qjl": c_part})
                        del left, right, residual, w
                    del gpu_weights, base
            if layer_payloads:
                for (rank, case), payload in layer_payloads.items():
                    folder = out / "checkpoints" / f"rank_{rank}" / case
                    file = folder / f"layer_{layer}.pt"
                    size, components = _save_layer(file, payload)
                    layer_files[(rank, case)].append(file.name)
                    size_totals[(rank, case)]["layer_file_bytes"] += size
                    for key, value in components.items():
                        size_totals[(rank, case)][key] += value
            print(f"[shared-lowrank-qjl] layer {layer}: {len(groups)} groups, "
                  f"{len(valid)} ranks", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        reader.close()

    config = {"model": MODEL_ID, "model_path": str(model_dir), "layers": layers,
              "experts": experts, "projections": projections, "ranks": ranks,
              "qjl_d": args.qjl_d, "seed": args.seed, "num_groups": args.num_groups,
              "singleton_ratio": args.singleton_ratio, "backend": args.backend,
              "saliency_delta": args.saliency_delta, "mode": args.mode,
              "grouping_source": cache.get("metadata", {}).get(
                  "source", "supplied grouping cache; origin unspecified"),
              "grouping_cache": str(args.grouping_cache) if args.grouping_cache else
                                str(out / "grouping_cache.pt")}
    checkpoints = []
    if args.mode == "compress":
        for (rank, case), files in sorted(layer_files.items()):
            folder = out / "checkpoints" / f"rank_{rank}" / case
            manifest = folder / "manifest.json"
            metadata = {"format": FORMAT, "case": case, "rank": rank,
                        "qjl_d": args.qjl_d, "seed": args.seed,
                        "model": MODEL_ID, "model_path": str(model_dir),
                        "layers": layers, "layer_files": files}
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")
            totals = dict(size_totals[(rank, case)])
            size = totals["layer_file_bytes"] + manifest.stat().st_size
            component_sum = sum(v for k, v in totals.items() if k != "layer_file_bytes")
            if size < component_sum:
                raise AssertionError("checkpoint smaller than stored tensors")
            checkpoints.append({"rank": rank, "case": case,
                                "manifest": str(manifest), "checkpoint_bytes": size,
                                "effective_bits_per_weight": 8 * size / original_weight_count,
                                "bf16_compression_ratio": 2 * original_weight_count / size,
                                "original_bf16_expert_bytes": 2 * original_weight_count,
                                **totals, "metadata_bytes": size - component_sum})
    if not records:
        raise ValueError("no valid ranks for the selected projections")
    summary = _summaries(records)
    _write_csv(out / "per_expert.csv", records)
    _write_csv(out / "summary.csv", summary)
    (out / "summary.json").write_text(json.dumps({"config": config,
        "summary": summary, "checkpoints": checkpoints,
        "skipped_ranks": skipped}, indent=2) + "\n")
    (out / "REPORT.md").write_text(_report(config, summary, checkpoints, skipped))
    _plot(out, summary)
    return {"config": config, "summary": summary, "checkpoints": checkpoints,
            "skipped_ranks": skipped}


def evaluate(args) -> dict:
    """User-invoked full lm-eval on a saved A/B/C checkpoint."""
    if args.checkpoint is None:
        raise ValueError("eval mode requires --checkpoint path/to/manifest.json")
    if args.eval_limit < 0:
        raise ValueError("eval-limit must be nonnegative; zero means no limit")
    from transformers import AutoTokenizer
    from rescode.evaluation import evaluate_fewshot
    from rescode.merging.eval_summary import report_eval_summary

    manifest = args.checkpoint.expanduser().resolve()
    metadata = json.loads(manifest.read_text())
    model_dir = args.model_path.expanduser().resolve()
    if metadata.get("model") != MODEL_ID:
        raise ValueError("checkpoint model identity mismatch")
    model = _load_hf_model(model_dir, args.gpu_memory_fraction)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    install_checkpoint(model, manifest, args.backend)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "lm_eval.txt"
    result = evaluate_fewshot(
        model, tokenizer, args.tasks, 0, eval_batch_size=args.eval_batch_size,
        output_path=str(log_path), limit=args.eval_limit or None)
    task_summary = report_eval_summary(
        result, args.tasks, str(log_path),
        header=f"Shared Base + Low-rank + QJL case {metadata['case']} rank {metadata['rank']}")
    metrics = {"checkpoint": str(manifest), "case": metadata["case"],
               "rank": metadata["rank"], "tasks": args.tasks,
               "eval_limit": args.eval_limit, "task_summary": task_summary,
               "task_results": result.get("results", {})}
    (args.output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2,
                                                                  default=str) + "\n")
    return metrics


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analyze", "compress", "eval"),
                        required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat"))
    parser.add_argument("--layers", default="all")
    parser.add_argument("--projections", default="gate,up,down")
    parser.add_argument("--max-experts", type=int)
    parser.add_argument("--ranks", default="0,32,64,128,256,512")
    parser.add_argument("--qjl-d", type=int, default=4096)
    parser.add_argument("--singleton-ratio", type=float, default=0.15)
    parser.add_argument("--num-groups", type=int, default=30)
    parser.add_argument("--saliency-delta", type=float, default=1e-3)
    parser.add_argument("--linkage", default="average")
    parser.add_argument("--grouping-cache", type=Path,
                        help="ResCode .pt cache or directory containing group/saliency state .pt files")
    parser.add_argument("--saliency-cache", type=Path,
                        help="optional separate ResCode saliency .pt cache")
    parser.add_argument("--routed-input-cache", type=Path,
                        help="optional collect_projection_samples train cache")
    parser.add_argument("--calibration-data", type=Path, default=DEFAULT_C4,
                        help="C4 JSON path; the standard shard is downloaded if missing")
    parser.add_argument("--calibration-blocks", type=int, default=32)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--calibration-batch-size", type=int, default=2)
    parser.add_argument("--diagnostic-samples", type=int, default=16)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="zero runs the complete selected tasks")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backend", choices=("triton", "reference"), default="triton")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=Path,
                        help="manifest.json for eval mode")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/shared_lowrank_qjl")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.mode == "eval":
        evaluate(args)
    else:
        analyze_or_compress(args)


if __name__ == "__main__":
    main()
