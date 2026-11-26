"""Activation-aware pair analysis. This module has no model-loading side effects."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from collections import defaultdict

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F

from .bundled_model import dense_qjl_base
from .core import ProjectionBank, pack_signs, unpack_signs
from .residual_bundle_model import bundle_signs

PROJECTIONS = ("gate", "up", "down")
CASES = ("A", "B", "C2", "C1", "D", "E", "F", "G", "H")


def cosine(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30))


def covariance_error(weight, base, covariance):
    """Tr[(W-B) C (W-B)^T] / Tr[W C W^T], without probes."""
    delta = weight.float() - base.float()
    num = (delta @ covariance * delta).sum().clamp_min(0)
    den = (weight.float() @ covariance * weight.float()).sum().clamp_min(1e-30)
    return float((num / den).sqrt())


def covariance_cosine(weight, base, covariance):
    w, b = weight.float(), base.float()
    cross = (w @ covariance * b).sum()
    nw = (w @ covariance * w).sum().clamp_min(0)
    nb = (b @ covariance * b).sum().clamp_min(0)
    return float(cross / (nw * nb).sqrt().clamp_min(1e-30))


def conditioned_weight_cosine(wi, wj, cg):
    return covariance_cosine(wi, wj, cg)


def psd_sqrt(c, regularization=0.0, inverse=False):
    c = (c.float() + c.float().T) * .5
    values, vectors = torch.linalg.eigh(c)
    values = values.clamp_min(0)
    if inverse:
        factor = (values + regularization).clamp_min(1e-30).rsqrt()
    else:
        factor = values.sqrt()
    return (vectors * factor) @ vectors.T


def covariance_factors(c, regularization, values=None, vectors=None):
    if values is None or vectors is None:
        values, vectors = torch.linalg.eigh((c.float() + c.float().T) * .5)
    values = values.clamp_min(0)
    root = (vectors * values.sqrt()) @ vectors.T
    invroot = (vectors * (values + regularization).clamp_min(1e-30).rsqrt()) @ vectors.T
    return root, invroot


def principal_basis(c, rank):
    values, vectors = torch.linalg.eigh((c + c.T) * .5)
    rank = min(rank, c.shape[0])
    return vectors[:, -rank:], float(values[-rank:].clamp_min(0).sum() /
                                        values.clamp_min(0).sum().clamp_min(1e-30))


def subspace_overlap(a, b):
    return float((a.T @ b).square().sum() / min(a.shape[1], b.shape[1]))


def optimized_scale(w, b, c, wc=None):
    numerator = ((w.float() @ c if wc is None else wc) * b.float()).sum(-1)
    denominator = (b.float() @ c * b.float()).sum(-1)
    return torch.where(denominator > 1e-20,
                       (numerator / denominator.clamp_min(1e-20)).clamp_min(0),
                       torch.zeros_like(denominator))


def optimal_dense_shared(wi, wj, ci, cj, regularization):
    c = ci + cj
    ridge = regularization * max(float(torch.trace(c) / c.shape[0]), 1e-6)
    rhs = wi @ ci + wj @ cj
    return torch.linalg.solve(c + ridge * torch.eye(c.shape[0], device=c.device), rhs.T).T


def decode(packed, bank, n):
    zeros = torch.ones(packed.shape[0], device=packed.device)
    return torch.cat([part for _, part in dense_qjl_base(packed, zeros, bank, n)])


class PrefixProjectionBank:
    """D=n comparison uses the first n columns of the D>=n Gaussian bank."""

    def __init__(self, full_bank, d):
        if full_bank.d < d:
            raise ValueError("the common Gaussian dimension must be at least n")
        self.full_bank, self.d = full_bank, d

    def get(self, input_dim, device):
        return self.full_bank.get(input_dim, device)[:, :self.d]


@dataclass
class ExpertCache:
    w: torch.Tensor
    c: torch.Tensor
    cg: torch.Tensor
    sign: torch.Tensor
    packed: torch.Tensor
    norm: torch.Tensor
    wc: torch.Tensor
    wcg: torch.Tensor
    basis: dict
    energy: dict
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    individual: torch.Tensor
    individual_half: torch.Tensor


def make_expert_cache(weight, covariance, bank, half_bank, ranks):
    w, c = weight.detach().float(), covariance.detach().float()
    g = bank.get(w.shape[1], w.device)
    wg = w.to(g.dtype) @ g
    sign = wg >= 0
    packed = pack_signs(sign)
    norm = w.norm(dim=1)
    wc = w @ c
    half = half_bank.get(w.shape[1], w.device)
    half_packed = pack_signs((w.to(half.dtype) @ half) >= 0)
    values, vectors = torch.linalg.eigh((c + c.T) * .5)
    positive = values.clamp_min(0)
    bases, energies = {}, {}
    for rank in ranks:
        k = min(rank, c.shape[0])
        bases[rank] = vectors[:, -k:]
        energies[rank] = float(positive[-k:].sum() / positive.sum().clamp_min(1e-30))
    return ExpertCache(w, c, wg, sign, packed, norm, wc, wc @ g.float(),
                       bases, energies, values, vectors,
                       decode(packed, bank, w.shape[1]) * norm[:, None],
                       decode(half_packed, half_bank, w.shape[1]) * norm[:, None])


def pair_similarity(i, j, ei, ej, co_count, token_counts, ranks, probe_i=None,
                    probe_j=None, ffn=None, gaussian=None, activation_bits=True):
    wi, wj = ei.w, ej.w
    row_cos = F.cosine_similarity(wi, wj, dim=1)
    union = token_counts[i] + token_counts[j] - co_count
    cg = (ei.c + ej.c) * .5
    result = {"expert_i": i, "expert_j": j,
              "raw_cosine": cosine(wi, wj),
              "row_cosine_mean": float(row_cos.mean()),
              "row_cosine_min": float(row_cos.min()),
              "row_cosine_p10": float(torch.quantile(row_cos, .1)),
              "row_cosine_median": float(torch.median(row_cos)),
              "row_cosine_p90": float(torch.quantile(row_cos, .9)),
              "corouting_jaccard": float(co_count / max(union, 1)),
              "activation_cosine": conditioned_weight_cosine(wi, wj, cg),
              "bit_agreement": float((ei.sign == ej.sign).float().mean()),
              "theory_bit_agreement": float((1 - torch.acos(row_cos.clamp(-1, 1)) / math.pi).mean())}
    for rank in ranks:
        result[f"subspace_overlap_r{rank}"] = subspace_overlap(ei.basis[rank], ej.basis[rank])
        result[f"energy_i_r{rank}"] = ei.energy[rank]
        result[f"energy_j_r{rank}"] = ej.energy[rank]
        result[f"cross_energy_i_r{rank}"] = float((ej.basis[rank].T @ ei.c * ej.basis[rank].T).sum() /
                                              torch.trace(ei.c).clamp_min(1e-30))
        result[f"cross_energy_j_r{rank}"] = float((ei.basis[rank].T @ ej.c * ei.basis[rank].T).sum() /
                                              torch.trace(ej.c).clamp_min(1e-30))
    if activation_bits:
        # C_g is pair-dependent; only this diagnostic requires its square root.
        root = psd_sqrt(cg)
        if gaussian is None:
            raise ValueError("Gaussian projection is needed for conditioned bits")
        bi = wi @ root @ gaussian
        bj = wj @ root @ gaussian
        result["conditioned_bit_agreement"] = float(((bi >= 0) == (bj >= 0)).float().mean())
    for direction, probe in (("i_to_j", probe_i), ("j_to_i", probe_j)):
        if probe is None or probe.numel() == 0:
            result[f"projection_cosine_{direction}"] = math.nan
            result[f"ffn_cosine_{direction}"] = math.nan
            continue
        x = probe.to(wi.device).float()
        result[f"projection_cosine_{direction}"] = cosine(x @ wi.T, x @ wj.T)
        result[f"ffn_cosine_{direction}"] = (cosine(ffn(i, x), ffn(j, x))
                                                  if ffn is not None else math.nan)
    vals = [result[f"projection_cosine_{x}"] for x in ("i_to_j", "j_to_i")]
    result["functional_cosine"] = float(np.nanmean(vals))
    return result


def base_cases(ei, ej, bank, half_bank, seed, regularization):
    """Decode each distinct shared code once; return case -> (base_i, base_j, stats)."""
    wi, wj, ci, cj = ei.w, ej.w, ei.c, ej.c
    n = wi.shape[1]
    equal = (wi + wj) * .5
    optimum = optimal_dense_shared(wi, wj, ci, cj, regularization)
    optimal_code = pack_signs((optimum @ bank.get(n, wi.device).float()) >= 0)
    majority = bundle_signs(ei.packed, ej.packed, bank.d, "majority", seed)
    def normalized(start, end):
        return (ei.cg[start:end] / ei.norm[start:end, None].clamp_min(1e-30),
                ej.cg[start:end] / ej.norm[start:end, None].clamp_min(1e-30))
    magnitude = bundle_signs(ei.packed, ej.packed, bank.d, "magnitude-aware", seed,
                             projections=normalized)
    aware = pack_signs((ei.wcg + ej.wcg) >= 0)
    codes = {"D": optimal_code, "E": majority, "F": majority, "G": magnitude, "H": aware}
    decoded = {key: decode(code, bank, n) for key, code in
               {"D": optimal_code, "majority": majority, "G": magnitude, "H": aware}.items()}
    cases = {"A": (equal, equal, {}), "B": (optimum, optimum, {}),
             "C2": (ei.individual, ej.individual, {}),
             "C1": (ei.individual_half, ej.individual_half, {})}
    for case, code in codes.items():
        b = decoded["majority" if case in ("E", "F") else case]
        if case == "E":
            ai, aj = ei.norm, ej.norm
        else:
            ai, aj = optimized_scale(wi, b, ci, ei.wc), optimized_scale(wj, b, cj, ej.wc)
        bi, bj = b * ai[:, None], b * aj[:, None]
        cases[case] = (bi, bj, {"bit_preservation_i": float((unpack_signs(code, bank.d) == ei.sign).float().mean()),
                                "bit_preservation_j": float((unpack_signs(code, bank.d) == ej.sign).float().mean()),
                                "packed_bytes": code.numel()})
    return cases


def covariance_terms(w, c):
    wc = w.float() @ c.float()
    return wc, (wc * w.float()).sum().clamp_min(1e-30)


def _base_covariance_metrics(w, b, c, terms=None):
    wc, w_energy = terms if terms is not None else covariance_terms(w, c)
    b_energy = (b.float() @ c.float() * b.float()).sum().clamp_min(0)
    cross = (wc * b.float()).sum()
    error2 = (w_energy + b_energy - 2 * cross).clamp_min(0)
    error = float((error2 / w_energy).sqrt())
    similarity = float(cross / (w_energy * b_energy).sqrt().clamp_min(1e-30))
    return error, similarity


def base_metrics(w, b, c, heldout_c, probe=None, cal_terms=None, held_terms=None):
    err = float((w - b).norm() / w.norm().clamp_min(1e-30))
    cal_error, cal_cosine = _base_covariance_metrics(w, b, c, cal_terms)
    held_error, held_cosine = _base_covariance_metrics(w, b, heldout_c, held_terms)
    result = {"weight_error": err, "calibration_error": cal_error,
              "heldout_error": held_error, "calibration_cosine": cal_cosine,
              "heldout_cosine": held_cosine}
    if probe is not None and probe.numel():
        result["probe_output_cosine"] = cosine(probe @ w.T, probe @ b.T)
    return result


def randomized_spectrum(matrix, rank, seed, oversample=8, niter=2, exact=False):
    m = matrix.float()
    if exact:
        u, s, vh = torch.linalg.svd(m, full_matrices=False)
        return u, s, vh.T
    q = min(min(m.shape), max(rank, 1) + oversample)
    if q == min(m.shape):
        return randomized_spectrum(m, rank, seed, exact=True)
    devices = [m.device.index or 0] if m.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return torch.svd_lowrank(m, q=q, niter=niter)


def lowrank_covariance_errors(residual, weight, covariance, left, right, ranks,
                              weight_energy=None):
    """Errors for all factor prefixes using two large GEMMs, then small Grams."""
    c = covariance.float()
    r = residual.float()
    rc = r @ c
    residual_energy = (rc * r).sum().double()
    if weight_energy is None:
        weight_energy = (weight.float() @ c * weight.float()).sum()
    denominator = torch.as_tensor(weight_energy, device=r.device).double().clamp_min(1e-30)
    if left.shape[1] == 0:
        return {rank: float((residual_energy / denominator).clamp_min(0).sqrt())
                for rank in ranks}
    cross_by_column = (left * (rc @ right.T)).sum(0).double()
    right_cov = right @ c
    gram = (left.T @ left).double() * (right_cov @ right.T).double()
    cross_prefix = cross_by_column.cumsum(0)
    quad_prefix = gram.cumsum(0).cumsum(1).diagonal()
    result = {}
    for rank in ranks:
        k = min(rank, left.shape[1])
        numerator = (residual_energy - 2 * cross_prefix[k - 1] + quad_prefix[k - 1]
                     if k else residual_energy).clamp_min(0)
        result[rank] = float((numerator / denominator).sqrt())
    return result


def spectrum_rows(w, b, c, heldout_c, ranks, seed, method="randomized", oversample=8,
                  ridge=1e-5, covariance_root=None, covariance_inverse_root=None,
                  calibration_weight_energy=None, heldout_weight_energy=None):
    r = w.float() - b.float()
    root = covariance_root if covariance_root is not None else psd_sqrt(c)
    weighted = r @ root
    invroot = covariance_inverse_root if covariance_inverse_root is not None else psd_sqrt(c, regularization=ridge, inverse=True)
    max_rank = max(ranks)
    result = []
    for space, matrix in (("weight", r), ("activation", weighted)):
        u, s, v = randomized_spectrum(matrix, max_rank, seed + (space == "activation"),
                                      oversample, exact=method == "exact")
        effective_exact = method == "exact" or max_rank + oversample >= min(matrix.shape)
        total = matrix.square().sum().clamp_min(1e-30)
        rho = float(matrix.norm() / ((w @ root if space == "activation" else w).norm().clamp_min(1e-30)))
        kmax = min(max_rank, len(s))
        left_all = (u[:, :kmax] * s[:kmax]).to(torch.bfloat16).float()
        right_all = (v[:, :kmax].T @ invroot if space == "activation" else
                     v[:, :kmax].T).to(torch.bfloat16).float()
        cal_errors = lowrank_covariance_errors(r, w, c, left_all, right_all, ranks,
                                               calibration_weight_energy)
        held_errors = lowrank_covariance_errors(r, w, heldout_c, left_all, right_all,
                                                ranks, heldout_weight_energy)
        for rank in ranks:
            k = min(rank, len(s))
            captured = s[:k].square().sum()
            estimated_eta = float(captured / total)
            tail = rho * math.sqrt(max(0, 1 - estimated_eta))
            result.append({"space": space, "rank": rank, "method": method,
                           "decomposition_method": "exact_fallback" if method == "randomized" and
                                                   max_rank + oversample >= min(matrix.shape) else method,
                           "residual_norm_ratio": rho, "estimated_energy_retention": estimated_eta,
                           "estimated_normalized_tail": tail,
                           "optimal_tail_exact_svd": tail if effective_exact else math.nan,
                           "tail_randomized_estimate": tail if not effective_exact else math.nan,
                           "actual_calibration_error": cal_errors[rank],
                           "actual_heldout_error": held_errors[rank],
                           "singular_values": s.detach().cpu().tolist() if rank == max_rank else None})
    return result


def perfect_matching(rows, score_key, maximize=True, n_experts=60, seed=0):
    if score_key == "random":
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_experts)
        return sorted(tuple(sorted(map(int, order[k:k + 2]))) for k in range(0, n_experts, 2))
    graph = nx.Graph()
    graph.add_nodes_from(range(n_experts))
    for row in rows:
        value = float(row[score_key])
        if math.isfinite(value):
            graph.add_edge(int(row["expert_i"]), int(row["expert_j"]),
                           weight=value if maximize else -value)
    found = nx.max_weight_matching(graph, maxcardinality=True, weight="weight")
    pairs = sorted(tuple(sorted(edge)) for edge in found)
    if len(pairs) != n_experts // 2 or sorted(e for pair in pairs for e in pair) != list(range(n_experts)):
        raise ValueError("candidate graph has no feasible perfect matching")
    return pairs


class StreamingRoutedMoments:
    """Hooks the router and actual expert projection inputs of Qwen2 MoE.

    Router indices are computed from the router's own output, with the same
    top-k order as Qwen2MoeSparseMoeBlock.forward. Hooked projection inputs
    are the tensors passed into the real experts; no expert forward is replayed.
    """

    def __init__(self, model, layers, probe_limit=32, profile=False):
        self.model = model
        self.layers = layers
        self.probe_limit = probe_limit
        self.profile_enabled = profile
        self.profile = {"covariance_addmm_seconds": 0., "covariance_transfer_seconds": 0.}
        self.stats = defaultdict(dict)
        self.coroute = {}
        self.mask = None
        self.selected = {}
        self.handles = []
        self._install()

    def _install(self):
        for layer in self.layers:
            block = self.model.model.layers[layer].mlp
            n = len(block.experts)
            self.coroute[layer] = torch.zeros(n, n, dtype=torch.int64)
            self.handles.append(block.gate.register_forward_hook(self._router_hook(layer, block)))
            for expert_id, expert in enumerate(block.experts):
                self.handles.append(expert.gate_proj.register_forward_pre_hook(
                    self._input_hook(layer, expert_id, "gate")))
                self.handles.append(expert.down_proj.register_forward_pre_hook(
                    self._input_hook(layer, expert_id, "down")))

    def _router_hook(self, layer, block):
        def hook(_module, _inputs, logits):
            selected = torch.topk(torch.softmax(logits.float(), dim=-1),
                                  block.top_k, dim=-1).indices
            if self.mask is None or len(self.mask) != selected.shape[0]:
                raise RuntimeError("attention mask is not aligned with router output")
            self.selected[layer] = selected
            valid = self.mask.to(selected.device)
            # One batched count; no per-token CPU/GPU synchronization.
            routed = F.one_hot(selected[valid], num_classes=len(block.experts)).sum(1).float()
            self.coroute[layer] += (routed.T @ routed).long().cpu()
        return hook

    def _input_hook(self, layer, expert_id, projection):
        def hook(_module, inputs):
            selected = self.selected[layer]
            # Qwen uses torch.where(expert_mask[expert_id]); its first index is
            # top-k slot, second index is flattened token ID.
            slot, token = torch.where(selected.T == expert_id)
            x = inputs[0]
            if len(token) != x.shape[0]:
                raise RuntimeError("routed input order differs from router selection")
            valid = self.mask.to(token.device)[token]
            x = x[valid].detach().float()
            key = (expert_id, projection)
            if key not in self.stats[layer]:
                n = x.shape[-1]
                self.stats[layer][key] = {"sum_xxt": torch.zeros(n, n, dtype=torch.float64),
                                          "count": 0, "probe": []}
            stat = self.stats[layer][key]
            if x.numel():
                if self.profile_enabled:
                    if x.is_cuda:
                        torch.cuda.synchronize(x.device)
                    started = time.perf_counter()
                    product = x.T @ x
                    if x.is_cuda:
                        torch.cuda.synchronize(x.device)
                    self.profile["covariance_addmm_seconds"] += time.perf_counter() - started
                    started = time.perf_counter()
                    stat["sum_xxt"] += product.double().cpu()
                    self.profile["covariance_transfer_seconds"] += time.perf_counter() - started
                else:
                    stat["sum_xxt"] += (x.T @ x).double().cpu()
                stat["count"] += len(x)
                remaining = self.probe_limit - sum(len(p) for p in stat["probe"])
                if remaining > 0:
                    stat["probe"].append(x[:remaining].to(torch.bfloat16).cpu())
        return hook

    def begin_batch(self, attention_mask):
        self.mask = attention_mask.reshape(-1).bool()
        self.selected.clear()

    def snapshot(self):
        output = {}
        for layer in self.layers:
            experts = {}
            for (expert_id, projection), stat in self.stats[layer].items():
                count = stat["count"]
                experts.setdefault(expert_id, {})[projection] = {
                    "covariance": (stat["sum_xxt"] / max(count, 1)).float().clone(),
                    "count": count,
                    "probe": torch.cat(stat["probe"]) if stat["probe"] else None}
            # Gate and Up receive exactly the same tensor, so they share data.
            for expert in experts.values():
                if "gate" in expert:
                    expert["up"] = expert["gate"]
            for expert_id in range(len(self.coroute[layer])):
                routed = int(self.coroute[layer][expert_id, expert_id])
                observed_gate = experts.get(expert_id, {}).get("gate", {}).get("count", 0)
                observed_down = experts.get(expert_id, {}).get("down", {}).get("count", 0)
                if observed_gate != routed or observed_down != routed:
                    raise RuntimeError(f"layer {layer} expert {expert_id}: routed={routed}, "
                                       f"gate={observed_gate}, down={observed_down}")
            output[layer] = {"experts": experts,
                             "coroute": self.coroute[layer].clone()}
        return output

    def restore(self, saved):
        for layer in self.layers:
            self.coroute[layer] = saved[layer]["coroute"].clone()
            self.stats[layer].clear()
            for expert_id, projections in saved[layer]["experts"].items():
                for projection in ("gate", "down"):
                    if projection not in projections:
                        continue
                    item = projections[projection]
                    self.stats[layer][(int(expert_id), projection)] = {
                        "sum_xxt": item["covariance"].double() * item["count"],
                        "count": int(item["count"]),
                        "probe": [item["probe"]] if item["probe"] is not None else []}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
