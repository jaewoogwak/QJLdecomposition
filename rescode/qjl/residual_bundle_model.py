"""Pairwise sharing of exponent-mantissa Base + QJL residual sketches."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .codebook_model import CodebookExpert, bank_for, base_state_bytes, restore_base
from .core import PackedResidual, pack_signs, unpack_signs
from .exponent_mantissa_base import (encode_exponent_mantissa_base,
                                      reconstruct_base_from_weight)
from .exponent_model import PROJECTIONS, _install

FORMAT = "rescode-residual-pair-bundle-v1"
KEYS = tuple(PROJECTIONS)
_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def validate_pairs(pairs, n_experts):
    if len(pairs) * 2 != n_experts or any(len(p) != 2 or p[0] == p[1] for p in pairs):
        raise ValueError("expected disjoint expert pairs")
    if sorted(int(i) for p in pairs for i in p) != list(range(n_experts)):
        raise ValueError("each expert must occur in exactly one pair")


def hamming_agreement(a, b, d):
    """Exact number of agreeing, non-padding bits in two packed sketches."""
    if a.shape != b.shape:
        raise ValueError("sketch shapes differ")
    x = a.numpy()
    y = b.numpy()
    full, remainder = divmod(d, 8)
    different = int(_POPCOUNT[np.bitwise_xor(x[:, :full], y[:, :full])].sum())
    if remainder:
        different += int(_POPCOUNT[np.bitwise_and(
            np.bitwise_xor(x[:, full], y[:, full]), (1 << remainder) - 1)].sum())
    return a.shape[0] * d - different


def make_pairs(sketches, d, method, seed):
    ids = sorted(sketches)
    if len(ids) % 2:
        raise ValueError("expert count must be even")
    generator = np.random.default_rng(seed)
    order = generator.permutation(ids).tolist()
    if method == "random":
        pairs = [tuple(sorted(order[i:i + 2])) for i in range(0, len(order), 2)]
    elif method == "similarity":
        import networkx as nx
        graph = nx.Graph()
        graph.add_nodes_from(order)
        for ix, i in enumerate(order):
            for j in order[ix + 1:]:
                graph.add_edge(i, j, weight=hamming_agreement(sketches[i], sketches[j], d))
        matching = nx.algorithms.matching.max_weight_matching(
            graph, maxcardinality=True, weight="weight")
        pairs = [tuple(sorted(p)) for p in matching]
    else:
        raise ValueError(f"unknown pairing: {method}")
    pairs.sort()
    validate_pairs(pairs, len(ids))
    return pairs


def _tie_generator(seed, layer, key, pair_id):
    generator = torch.Generator(device="cpu")
    generator.manual_seed((seed + 1000003 * layer + 9176 * KEYS.index(key)
                           + 65537 * pair_id) % (2 ** 63))
    return generator


def bundle_signs(first, second, d, method, seed, *, projections=None, row_chunk=128):
    """Bundle packed signs; projections yields per-row normalized projected values.

    Projection values are consumed a chunk at a time for magnitude-aware voting.
    """
    if first.shape != second.shape:
        raise ValueError("pair sketch shapes differ")
    if method not in ("majority", "magnitude-aware"):
        raise ValueError("bundling requires majority or magnitude-aware")
    rng = torch.Generator(device="cpu").manual_seed(seed % (2 ** 63))
    chunks = []
    for start in range(0, first.shape[0], row_chunk):
        end = min(start + row_chunk, first.shape[0])
        a = unpack_signs(first[start:end], d)
        b = unpack_signs(second[start:end], d)
        tie = torch.randint(0, 2, a.shape, generator=rng, dtype=torch.uint8).bool()
        if method == "majority":
            chosen = torch.where(a == b, a, tie)
        else:
            if projections is None:
                raise ValueError("magnitude-aware needs normalized projections")
            zi, zj = projections(start, end)
            if zi.shape != a.shape or zj.shape != a.shape:
                raise ValueError("projection shape differs from sketch")
            chosen = torch.where(a == b, a, torch.where(
                zi.abs() == zj.abs(), tie,
                torch.where(zi.abs() > zj.abs(), a, b)))
        chunks.append(pack_signs(chosen))
    return torch.cat(chunks)


def _residual_chunk(weight, beta, start, end):
    part = weight[start:end]
    base = reconstruct_base_from_weight(part, beta[start:end])
    return torch.where(torch.isfinite(part.float()), part.float() - base,
                       torch.zeros_like(base))


@torch.no_grad()
def encode_projection_rows(weight, bank, row_chunk=128):
    """Reuse the existing Base format; never form a full dense Base or residual."""
    state = encode_exponent_mantissa_base(weight, row_chunk=row_chunk)
    p = bank.get(weight.shape[1], weight.device)
    packed, norms = [], []
    for start in range(0, weight.shape[0], row_chunk):
        end = min(start + row_chunk, weight.shape[0])
        residual = _residual_chunk(weight, state["beta"], start, end)
        norm = torch.linalg.vector_norm(residual, dim=1)
        signs = (residual.to(p.dtype) @ p) >= 0
        signs[norm == 0] = False
        packed.append(pack_signs(signs).cpu())
        norms.append(norm.cpu())
    return state, torch.cat(packed), torch.cat(norms)


@torch.no_grad()
def _normalized_projection(weight, beta, bank, start, end, norms):
    residual = _residual_chunk(weight, beta, start, end)
    p = bank.get(weight.shape[1], weight.device)
    z = (residual.to(p.dtype) @ p).float()
    z /= norms[start:end].to(z.device).clamp_min(1e-30)[:, None]
    z[norms[start:end].to(z.device) == 0] = 0
    return z.cpu()


def _install_layer(block, expert_states, sketches, banks, config):
    device = next(block.parameters()).device
    n_experts = len(expert_states)
    shared = {key: {int(i): value.to(device) for i, value in values.items()}
              for key, values in sketches.items()}
    experts = []
    for expert_id in range(n_experts):
        bases, residuals = {}, {}
        for key in KEYS:
            part = expert_states[expert_id][key]
            base = restore_base(part["base"], device, "exponent-mantissa")
            bank = bank_for(banks, base.shape[1], config["qjl_d_factor"],
                            config["projection_seed"])
            packed = shared[key][part["sketch_id"]]
            bases[key] = base
            residuals[key] = PackedResidual(packed, part["norms"].to(device),
                                             base.shape[1], bank.d, bank,
                                             config["backend"])
        experts.append(CodebookExpert(bases, residuals))
    _install(block, experts)


@torch.no_grad()
def compress_model(model, config):
    """Compress selected layers and return checkpoint state plus size records."""
    banks, layers, records = {}, {}, []
    model._codebook_projection_banks = banks
    for layer in config["layers"]:
        block = model.model.layers[layer].mlp
        original = block.experts
        n = len(original)
        if n != 60:
            raise ValueError(f"layer {layer}: expected 60 routed experts, got {n}")
        expert_states = {i: {} for i in range(n)}
        sketches = {}
        pair_maps = {}
        for key, attr in PROJECTIONS.items():
            temporary = {}
            weights = [getattr(expert, attr).weight.detach() for expert in original]
            bank = bank_for(banks, weights[0].shape[1], config["qjl_d_factor"],
                            config["projection_seed"])
            for i, weight in enumerate(weights):
                base, packed, norms = encode_projection_rows(
                    weight, bank, config.get("row_chunk", 128))
                temporary[i] = packed
                expert_states[i][key] = {"base": base, "norms": norms}
                records.append({"layer": layer, "projection": key, "expert": i,
                                "rows": weight.shape[0], "cols": weight.shape[1],
                                "base_bytes": base_state_bytes(base),
                                "norm_bytes": norms.numel() * norms.element_size()})
            pairs = make_pairs(temporary, bank.d, config["pairing"],
                               config["pairing_seed"] + layer * 1009 + KEYS.index(key) * 9176)
            pair_maps[key] = pairs
            sketches[key] = {}
            if config["bundle_method"] == "none":
                for i in range(n):
                    sketches[key][i] = temporary[i]
                    expert_states[i][key]["sketch_id"] = i
            else:
                for pair_id, (i, j) in enumerate(pairs):
                    projection = None
                    if config["bundle_method"] == "magnitude-aware":
                        projection = lambda start, end, i=i, j=j: (
                            _normalized_projection(weights[i], expert_states[i][key]["base"]["beta"],
                                                   bank, start, end, expert_states[i][key]["norms"]),
                            _normalized_projection(weights[j], expert_states[j][key]["base"]["beta"],
                                                   bank, start, end, expert_states[j][key]["norms"]))
                    sketches[key][pair_id] = bundle_signs(
                        temporary[i], temporary[j], bank.d, config["bundle_method"],
                        _tie_generator(config["bundle_seed"], layer, key, pair_id).initial_seed(),
                        projections=projection, row_chunk=config.get("row_chunk", 128))
                    expert_states[i][key]["sketch_id"] = pair_id
                    expert_states[j][key]["sketch_id"] = pair_id
            del temporary, weights
            print(f"[bundle] layer {layer} {key}: {n} experts, {len(sketches[key])} sketches", flush=True)
        layers[layer] = {"pairs": pair_maps, "experts": expert_states, "sketches": sketches}
        router, shared_expert, shared_gate, top_k = (
            block.gate, getattr(block, "shared_expert", None),
            getattr(block, "shared_expert_gate", None), block.top_k)
        _install_layer(block, expert_states, sketches, banks, config)
        if (block.gate is not router or getattr(block, "shared_expert", None) is not shared_expert
                or getattr(block, "shared_expert_gate", None) is not shared_gate
                or block.top_k != top_k):
            raise AssertionError("routing or uncompressed modules changed")
        del original
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[bundle] layer {layer}: installed", flush=True)
    return layers, records


def checkpoint_sizes(layers, path):
    base = norms = sketch = 0
    for state in layers.values():
        for expert in state["experts"].values():
            for part in expert.values():
                base += base_state_bytes(part["base"])
                norms += part["norms"].numel() * part["norms"].element_size()
        sketch += sum(t.numel() * t.element_size() for values in state["sketches"].values()
                      for t in values.values())
    total = Path(path).stat().st_size
    return {"checkpoint_bytes": total, "base_tensor_bytes": base,
            "qjl_sketch_bytes": sketch, "norm_bytes": norms,
            "qjl_tensor_bytes": sketch + norms, "projection_tensor_bytes": 0,
            "metadata_bytes": total - base - sketch - norms}


def save_checkpoint(layers, config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save({"format": FORMAT, "config": config, "layers": layers}, temp)
    os.replace(temp, path)
    return checkpoint_sizes(layers, path)


def load_checkpoint(model, path, backend=None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("wrong residual bundle checkpoint format")
    config = dict(payload["config"])
    if backend is not None:
        config["backend"] = backend
    banks = {}
    model._codebook_projection_banks = banks
    for layer, state in payload["layers"].items():
        for pairs in state["pairs"].values():
            validate_pairs(pairs, len(state["experts"]))
        _install_layer(model.model.layers[int(layer)].mlp, state["experts"],
                       state["sketches"], banks, config)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def gpu_resident_breakdown(model):
    base = norms = 0
    sketches = {}
    for block in model.model.layers:
        for expert in getattr(block.mlp, "_compact_experts", ()):
            for part in expert.bases.values():
                first = next(part.buffers())
                if first.is_cuda:
                    base += part.resident_bytes()
            for part in expert.residuals.values():
                if part.norms.is_cuda:
                    norms += part.norms.numel() * part.norms.element_size()
                if part.packed.is_cuda:
                    sketches[part.packed.untyped_storage().data_ptr()] = (
                        part.packed.untyped_storage().nbytes())
    projection = sum(bank.resident_bytes("cuda") for bank in
                     getattr(model, "_codebook_projection_banks", {}).values())
    return {"gpu_resident_base_bytes": base, "gpu_resident_norm_bytes": norms,
            "gpu_resident_qjl_sketch_bytes": sum(sketches.values()),
            "gpu_resident_projection_bytes": projection,
            "gpu_resident_compact_total_bytes": base + norms + sum(sketches.values()) + projection}
