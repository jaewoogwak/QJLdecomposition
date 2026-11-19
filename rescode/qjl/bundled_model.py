"""Majority-bundled QJL sketches with expert-specific low-rank corrections.

The effective base is reconstructed from the *stored packed signs* and the
runtime ProjectionBank. No dense base or individual sketch is kept in a
majority checkpoint.
"""

import math
import os
import time
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (ProjectionBank, pack_residual, pack_signs, unpack_signs,
                   qjl_reference, qjl_triton, qjl_triton_legacy)
from .model import _qjl_moe_forward, projections


FORMAT = "qjl-majority-lowrank-v1"
PROJECTIONS = ("gate", "up", "down")


def tensor_bytes(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def validate_groups(groups, n_experts):
    flat = [int(i) for group in groups for i in group]
    if sorted(flat) != list(range(n_experts)) or any(not group for group in groups):
        raise ValueError("groups must partition original routed expert IDs exactly")


def majority_signs(packed_by_expert, d, return_stats=False):
    """Vote over original sign bits; smallest expert ID resolves every tie."""
    ids = sorted(packed_by_expert)
    if not ids:
        raise ValueError("empty group")
    shape = packed_by_expert[ids[0]].shape
    if any(packed_by_expert[i].shape != shape for i in ids):
        raise ValueError("expert sketch shapes differ")
    counts = torch.zeros((shape[0], d), dtype=torch.int16,
                         device=packed_by_expert[ids[0]].device)
    first = unpack_signs(packed_by_expert[ids[0]], d)
    for expert_id in ids:
        counts += unpack_signs(packed_by_expert[expert_id], d).to(torch.int16)
    chosen = (counts * 2 > len(ids)) | ((counts * 2 == len(ids)) & first)
    result = pack_signs(chosen)
    if not return_stats:
        return result
    n = len(ids)
    equal_pairs = counts.float() * (counts.float() - 1) / 2
    zeros = n - counts.float()
    equal_pairs += zeros * (zeros - 1) / 2
    pairwise = float(equal_pairs.mean() / (n * (n - 1) / 2)) if n > 1 else 1.0
    return result, pairwise


def dense_qjl_base(packed, norms, bank, input_dim, row_chunk=128):
    """Yield FP32 rows of the mathematical estimator with runtime FP16 G."""
    p = bank.get(input_dim, packed.device)
    coefficient = math.sqrt(math.pi / 2) / bank.d
    for start in range(0, packed.shape[0], row_chunk):
        bits = unpack_signs(packed[start:start + row_chunk], bank.d)
        signs = bits.to(torch.float32).mul_(2).sub_(1)
        base = (signs @ p.T.float()) * (norms[start:start + row_chunk, None] * coefficient)
        yield start, base


def reconstruct_residual(weight, packed, norms, bank, row_chunk=128):
    pieces = []
    for start, base in dense_qjl_base(packed, norms, bank, weight.shape[1], row_chunk):
        original = weight[start:start + base.shape[0]].float()
        pieces.append(original - base)
    return torch.cat(pieces, dim=0)


def truncated_svd(residual, rank, seed, oversample=8, niter=2):
    """Randomized FP32 SVD; dimensions are output-by-input linear factors."""
    actual = min(int(rank), *residual.shape)
    if actual < 1:
        raise ValueError("rank must be positive")
    q = min(min(residual.shape), actual + oversample)
    devices = [residual.device.index or 0] if residual.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        u, s, v = torch.svd_lowrank(residual.float(), q=q, niter=niter)
    u = (u[:, :actual] * s[:actual]).to(torch.bfloat16)
    v = v[:, :actual].T.contiguous().to(torch.bfloat16)
    return u, v, actual


def lowrank_weight(u, v):
    return u.float() @ v.float()


def qjl_output(z, packed, norms, d, backend):
    if backend == "triton":
        return qjl_triton(z, packed, norms, d)
    if backend == "triton_legacy":
        return qjl_triton_legacy(z, packed, norms, d)
    if backend == "reference":
        return qjl_reference(z, packed, norms, d)
    raise ValueError(f"unknown backend: {backend}")


class PackedSketch(nn.Module):
    def __init__(self, packed):
        super().__init__()
        self.register_buffer("packed", packed.contiguous())


class ExpertCorrection(nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.norms = nn.ModuleDict()
        self.u = nn.ModuleDict()
        self.v = nn.ModuleDict()
        self.exact = nn.ModuleDict()
        for key, part in parts.items():
            self.norms[key] = _Buffer(part["norms"])
            if "u" in part:
                self.u[key], self.v[key] = _Buffer(part["u"]), _Buffer(part["v"])
            else:
                self.exact[key] = _Buffer(part["residual"])

    def correction(self, key, x):
        if key in self.exact:
            return F.linear(x.float(), self.exact[key].value).to(x.dtype)
        # The factors are stored as BF16 and multiplied as two actual linear layers.
        hidden = F.linear(x.to(self.v[key].value.dtype), self.v[key].value)
        return F.linear(hidden, self.u[key].value).to(x.dtype)


class _Buffer(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", value.contiguous())


class BundledGroup(nn.Module):
    def __init__(self, sketches, parts, bank, backend, bundle_mode):
        super().__init__()
        self.sketches = nn.ModuleDict({key: nn.ModuleDict({str(k): PackedSketch(v)
            for k, v in values.items()}) for key, values in sketches.items()})
        self.experts = nn.ModuleDict({str(i): ExpertCorrection(p) for i, p in parts.items()})
        self.bank, self.backend, self.bundle_mode = bank, backend, bundle_mode

    def packed(self, key, expert_id):
        sketch_id = "shared" if self.bundle_mode == "majority" else str(expert_id)
        return self.sketches[key][sketch_id].packed

    def _projection(self, key, expert_id, x, z=None):
        part = self.experts[str(expert_id)]
        if z is None:
            p = self.bank.get(x.shape[-1], x.device)
            z = x.reshape(-1, x.shape[-1]).to(p.dtype) @ p
        y = qjl_output(z, self.packed(key, expert_id), part.norms[key].value,
                       self.bank.d, self.backend)
        return y.reshape(*x.shape[:-1], -1).to(x.dtype) + part.correction(key, x)

    def forward_expert(self, expert_id, x, hidden_z=None):
        part = self.experts[str(expert_id)]
        if hidden_z is None:
            p = self.bank.get(x.shape[-1], x.device)
            hidden_z = x.reshape(-1, x.shape[-1]).to(p.dtype) @ p
        if self.backend == "triton":
            from .triton_kernel import packed_qjl_fused_gate_up
            g, u = packed_qjl_fused_gate_up(
                hidden_z.contiguous(), self.packed("gate", expert_id),
                part.norms["gate"].value, self.packed("up", expert_id),
                part.norms["up"].value, self.bank.d)
            gate = g.reshape(*x.shape[:-1], -1).to(x.dtype) + part.correction("gate", x)
            up = u.reshape(*x.shape[:-1], -1).to(x.dtype) + part.correction("up", x)
        else:
            gate = self._projection("gate", expert_id, x, hidden_z)
            up = self._projection("up", expert_id, x, hidden_z)
        return self._projection("down", expert_id, F.silu(gate) * up)


def _block(model, layer):
    return model.model.layers[layer].mlp


def install_grouped_block(block, groups, clusters, bank):
    validate_groups(groups, len(block.experts))
    ids = [None] * len(block.experts)
    singletons = {g[0] for g in groups if len(g) == 1}
    for cid, members in enumerate(groups):
        if len(members) > 1:
            for i in members:
                ids[i] = cid
    block.experts = nn.ModuleList([block.experts[i] if i in singletons else nn.Identity()
                                   for i in range(len(ids))])
    block.add_module("_qjl_clusters", clusters)
    block._qjl_expert_to_cluster = ids
    block._qjl_family = "qwen"
    block._qjl_hidden_bank = bank
    block.forward = MethodType(_qjl_moe_forward, block)


def _relative(candidate, reference):
    denom = torch.linalg.vector_norm(reference.float())
    return float(torch.linalg.vector_norm(candidate.float() - reference.float()) /
                 denom.clamp_min(1e-30))


@torch.no_grad()
def compress_model(model, groups, config, activation_samples=None, moe_probes=None):
    """Compress selected layers one at a time; return serializable checkpoint data."""
    bank = ProjectionBank(config["qjl_d"], config["projection_seed"])
    model._qjl_projection_bank = bank
    data, records, timings = {}, [], []
    for layer, layer_groups in sorted(groups.items()):
        block = _block(model, layer)
        validate_groups(layer_groups, len(block.experts))
        probe = (moe_probes or {}).get(layer)
        if probe is not None:
            probe = probe[0].to(next(block.parameters()).device)
            original_moe, original_logits = block(probe)
        old_experts = block.experts
        layer_state = {"groups": layer_groups, "clusters": {}, "singletons": {}}
        clusters = nn.ModuleDict()
        for cid, members in enumerate(layer_groups):
            if len(members) == 1:
                i = members[0]
                layer_state["singletons"][i] = {k: v.detach().cpu()
                    for k, v in old_experts[i].state_dict().items()}
                continue
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            group_start = time.perf_counter()
            sketch_states = {key: {} for key in PROJECTIONS}
            expert_parts = {i: {} for i in members}
            for key in PROJECTIONS:
                weights = {i: projections(old_experts[i], "qwen")[key].detach()
                           for i in members}
                packed = {}
                norms = {}
                for i, weight in weights.items():
                    packed[i], norms[i] = pack_residual(weight, bank)
                if config["bundle_mode"] == "majority":
                    shared, pairwise_agreement = majority_signs(packed, bank.d, True)
                    sketch_states[key]["shared"] = shared
                else:
                    _, pairwise_agreement = majority_signs(packed, bank.d, True)
                    sketch_states[key] = {str(i): packed[i] for i in members}
                for i, weight in weights.items():
                    signs = shared if config["bundle_mode"] == "majority" else packed[i]
                    residual = reconstruct_residual(weight, signs, norms[i], bank)
                    reconstruction = torch.empty_like(residual)
                    for start, base in dense_qjl_base(signs, norms[i], bank, weight.shape[1]):
                        reconstruction[start:start + base.shape[0]] = base + residual[start:start + base.shape[0]]
                    identity_error = _relative(reconstruction, weight)
                    if identity_error > 2e-6:
                        raise AssertionError(f"QJL base/residual identity failed: {layer}/{i}/{key}: {identity_error}")
                    residual_ratio = float(torch.linalg.vector_norm(residual) /
                                           torch.linalg.vector_norm(weight.float()).clamp_min(1e-30))
                    if config["residual_mode"] == "lowrank":
                        u, v, actual = truncated_svd(residual, config["rank"],
                            config["svd_seed"] + layer * 100000 + i * 3 + PROJECTIONS.index(key),
                            config["svd_oversample"], config["svd_niter"])
                        estimate = lowrank_weight(u, v)
                        part = {"norms": norms[i], "u": u, "v": v}
                        retention = float(1 - (residual - estimate).square().sum() /
                                          residual.square().sum().clamp_min(1e-30))
                    else:
                        estimate = residual
                        part = {"norms": norms[i], "residual": residual}
                        actual, retention = min(weight.shape), 1.0
                    expert_parts[i][key] = part
                    flip = (unpack_signs(packed[i], bank.d) !=
                            unpack_signs(signs, bank.d)).float().mean()
                    row = {"layer": layer, "group": cid, "expert": i, "projection": key,
                           "group_size": len(members), "rank": actual,
                           "majority_flip_rate": float(flip),
                           "sign_agreement": float(1 - flip),
                           "group_pairwise_sign_agreement": pairwise_agreement,
                           "residual_frobenius_ratio": residual_ratio,
                           "svd_energy_retention": retention,
                           "svd_residual_relative_error": float(
                               torch.linalg.vector_norm(residual - estimate) /
                               torch.linalg.vector_norm(residual).clamp_min(1e-30)),
                           "weight_relative_error": _relative(reconstruction - residual + estimate, weight),
                           "identity_relative_error": identity_error,
                           "runtime_rounding_relative_error": None,
                           "activation_output_relative_error": None,
                           "expert_output_relative_error": None}
                    sample = (activation_samples or {}).get(layer, {}).get(key, {}).get(i)
                    if sample is not None and sample.numel():
                        sample = sample.to(weight.device)
                        z = sample.to(bank.get(weight.shape[1], sample.device).dtype) @ bank.get(weight.shape[1], sample.device)
                        approximate = qjl_output(z, signs, norms[i], bank.d,
                                                 config["backend"]).to(sample.dtype)
                        correction = (F.linear(F.linear(sample.to(v.dtype), v), u).to(sample.dtype)
                                      if "u" in part else F.linear(sample.float(), residual).to(sample.dtype))
                        approximate = approximate + correction
                        original = F.linear(sample, weight)
                        row["activation_output_relative_error"] = _relative(approximate, original)
                        row["runtime_rounding_relative_error"] = _relative(
                            qjl_output(z, signs, norms[i], bank.d, config["backend"]),
                            F.linear(sample.float(), reconstruction - residual))
                    records.append(row)
            stored = {"sketches": {k: {s: p.cpu() for s, p in v.items()}
                        for k, v in sketch_states.items()},
                      "experts": {i: {k: {s: t.cpu() for s, t in part.items()}
                        for k, part in parts.items()} for i, parts in expert_parts.items()}}
            layer_state["clusters"][cid] = stored
            clusters[str(cid)] = BundledGroup(sketch_states, expert_parts, bank,
                                              config["backend"], config["bundle_mode"])
            for i in members:
                sample = (activation_samples or {}).get(layer, {}).get("gate", {}).get(i)
                if sample is not None and sample.numel():
                    sample = sample.to(next(old_experts[i].parameters()).device)
                    error = _relative(clusters[str(cid)].forward_expert(i, sample),
                                      old_experts[i](sample))
                    for row in records:
                        if row["layer"] == layer and row["expert"] == i and row["projection"] == "gate":
                            row["expert_output_relative_error"] = error
                            break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append({"layer": layer, "group": cid,
                            "compression_seconds": time.perf_counter() - group_start,
                            "peak_gpu_allocated_bytes": (torch.cuda.max_memory_allocated()
                                                         if torch.cuda.is_available() else None),
                            "experts": len(members)})
            print(f"[bundle] layer={layer} group={cid} experts={len(members)} ", flush=True)
        install_grouped_block(block, layer_groups, clusters, bank)
        if probe is not None:
            actual, logits = block(probe)
            if not torch.equal(logits, original_logits):
                raise AssertionError(f"router changed in layer {layer}")
            timings.append({"layer": layer, "group": None,
                            "moe_output_relative_error": _relative(actual, original_moe)})
        data[layer] = layer_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return data, records, timings


def save_checkpoint(data, config, path):
    if config["residual_mode"] == "exact":
        raise ValueError("exact residual is diagnostic-only and cannot be saved as a compression checkpoint")
    payload = {"format": FORMAT, "config": config, "layers": data}
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return checkpoint_sizes(payload, os.path.getsize(path))


def checkpoint_sizes(payload, file_bytes):
    shared = norms = factors = singletons = 0
    for layer in payload["layers"].values():
        singletons += tensor_bytes(layer["singletons"])
        for group in layer["clusters"].values():
            shared += tensor_bytes(group["sketches"])
            for parts in group["experts"].values():
                for part in parts.values():
                    norms += tensor_bytes(part["norms"])
                    factors += tensor_bytes({k: v for k, v in part.items() if k != "norms"})
    tensors = shared + norms + factors + singletons
    return {"checkpoint_bytes": file_bytes, "shared_packed_sign_bytes": shared,
            "expert_row_norm_bytes": norms, "lowrank_factor_bytes": factors,
            "singleton_bf16_bytes": singletons, "tensor_bytes": tensors,
            "metadata_and_zip_bytes": file_bytes - tensors}


def load_checkpoint(model, path, backend=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("not a majority-bundled QJL checkpoint")
    config = payload["config"]
    bank = ProjectionBank(config["qjl_d"], config["projection_seed"])
    model._qjl_projection_bank = bank
    for layer, layer_state in payload["layers"].items():
        block = _block(model, int(layer))
        device = next(block.parameters()).device
        clusters = nn.ModuleDict()
        for cid, group in layer_state["clusters"].items():
            sketches = {key: {sid: p.to(device) for sid, p in parts.items()}
                        for key, parts in group["sketches"].items()}
            parts = {int(i): {key: {name: t.to(device) for name, t in part.items()}
                     for key, part in expert.items()} for i, expert in group["experts"].items()}
            clusters[str(cid)] = BundledGroup(sketches, parts, bank,
                            backend or config["backend"], config["bundle_mode"])
        for i, state in layer_state["singletons"].items():
            block.experts[int(i)].load_state_dict(state)
        install_grouped_block(block, layer_state["groups"], clusters, bank)
    return payload


def gpu_resident_bytes(model):
    packed = norms = factors = singletons = 0
    for layer in model.model.layers:
        block = layer.mlp
        for cluster in getattr(block, "_qjl_clusters", {}).values():
            for sketch in cluster.sketches.values():
                packed += sum(tensor_bytes(s.packed) for s in sketch.values() if s.packed.is_cuda)
            for expert in cluster.experts.values():
                for part in expert.norms.values():
                    if part.value.is_cuda:
                        norms += tensor_bytes(part.value)
                for dictionary in (expert.u, expert.v, expert.exact):
                    factors += sum(tensor_bytes(v.value) for v in dictionary.values() if v.value.is_cuda)
        if hasattr(block, "_qjl_expert_to_cluster"):
            for i, cid in enumerate(block._qjl_expert_to_cluster):
                if cid is None:
                    singletons += sum(tensor_bytes(p) for p in block.experts[i].parameters() if p.is_cuda)
    return {"gpu_shared_packed_sign_bytes": packed, "gpu_expert_row_norm_bytes": norms,
            "gpu_lowrank_factor_bytes": factors, "gpu_singleton_bytes": singletons,
            "gpu_projection_matrix_bytes": model._qjl_projection_bank.resident_bytes("cuda")}
