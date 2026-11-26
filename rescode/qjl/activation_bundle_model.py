"""Activation-weighted grouped QJL bases and low-rank residuals.

Only packed group codes, expert row scales, and optional BF16 factors are
checkpointed. Calibration covariances are read during compression, never loaded
by inference. The format is separate from all previous QJL checkpoints.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .base_bundle_analysis import optimized_scale
from .base_bundle_model import BundledBaseExpert, LowRankCorrection
from .codebook_model import bank_for
from .core import PackedResidual, pack_signs, unpack_signs
from .exponent_model import PROJECTIONS, _install
from .residual_bundle_model import hamming_agreement, make_pairs

FORMAT = "activation-aware-qjl-group-v1"
KEYS = ("gate", "up", "down")
CASES = ("A", "B", "C", "D", "E")
SQRT_PI_HALF = math.sqrt(math.pi / 2)


def validate_groups(groups, n_experts, group_size):
    if n_experts % group_size or len(groups) != n_experts // group_size:
        raise ValueError("invalid number of groups")
    if any(len(group) != group_size for group in groups):
        raise ValueError("wrong group size")
    if sorted(i for group in groups for i in group) != list(range(n_experts)):
        raise ValueError("each routed expert must occur exactly once")


def make_groups(packed, d, group_size=2, pairing="similarity", seed=0):
    """Pair matching for K=2; deterministic balanced greedy groups for K>2."""
    n = len(packed)
    if n % group_size or group_size < 2:
        raise ValueError("expert count must be divisible by group size >= 2")
    if group_size == 2:
        groups = [list(pair) for pair in make_pairs({i: packed[i] for i in range(n)},
                                                    d, pairing, seed)]
        validate_groups(groups, n, group_size)
        return groups
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(n, generator=generator).tolist()
    if pairing == "random":
        groups = [sorted(order[i:i + group_size]) for i in range(0, n, group_size)]
    elif pairing == "similarity":
        affinity = torch.zeros((n, n), dtype=torch.int64)
        for i in range(n):
            for j in range(i + 1, n):
                affinity[i, j] = affinity[j, i] = hamming_agreement(packed[i].cpu(), packed[j].cpu(), d)
        remaining = set(range(n))
        groups = []
        positions = {expert: position for position, expert in enumerate(order)}
        for anchor in order:
            if anchor not in remaining:
                continue
            group = [anchor]
            remaining.remove(anchor)
            while len(group) < group_size:
                scores = [(sum(int(affinity[candidate, member]) for member in group),
                           -positions[candidate], candidate) for candidate in remaining]
                chosen = max(scores)[2]
                group.append(chosen)
                remaining.remove(chosen)
            groups.append(sorted(group))
    else:
        raise ValueError(f"unknown pairing: {pairing}")
    groups.sort()
    validate_groups(groups, n, group_size)
    return groups


def _random_ties(shape, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, 2, shape, generator=generator, dtype=torch.bool).to(device)


def vote(signs, seed, weights=None):
    """Boolean [expert,row,bit] majority or weighted vote with seeded ties."""
    signs = torch.stack(signs)
    values = signs.float().mul(2).sub(1)
    scores = values.sum(0) if weights is None else (values * torch.stack(weights)).sum(0)
    return torch.where(scores > 0, torch.ones_like(scores, dtype=torch.bool),
                       torch.where(scores < 0, torch.zeros_like(scores, dtype=torch.bool),
                                   _random_ties(scores.shape, seed, scores.device)))


def unscaled_base(signs, g):
    return SQRT_PI_HALF / g.shape[1] * (signs.float().mul(2).sub(1) @ g.float().T)


def regularized_roots(c, ridge):
    """Both factors use the same regularized covariance, so root@inverse=I."""
    c = (c.float() + c.float().T) * .5
    values, vectors = torch.linalg.eigh(c)
    floor = ridge * max(float(values.clamp_min(0).mean()), 1e-8)
    values = values.clamp_min(0) + floor
    root = (vectors * values.sqrt()) @ vectors.T
    invroot = (vectors * values.rsqrt()) @ vectors.T
    return root, invroot


def svd_factors(residual, rank, c=None, ridge=1e-4, method="exact", oversample=16,
                roots=None, niter=2, seed=0, return_source_energy=False):
    """Return BF16 factors, singular values, and FP32 theoretical energy."""
    if c is None:
        source, invroot = residual.float(), None
    else:
        root, invroot = roots if roots is not None else regularized_roots(c, ridge)
        source = residual.float() @ root
    k = min(rank, *source.shape)
    source_energy = source.square().sum().clamp_min(1e-30)
    if k == 0:
        u = source.new_zeros((source.shape[0], 0)).to(torch.bfloat16)
        v = source.new_zeros((0, source.shape[1])).to(torch.bfloat16)
        result = (u, v, source.new_zeros(0), 0.0)
        return result + (source_energy,) if return_source_energy else result
    if method == "exact":
        left, singular, right = torch.linalg.svd(source, full_matrices=False)
        vh = right[:k]
    elif method == "randomized":
        # The only large SVD is over q rows. CPU seeded Ω gives the same
        # starting subspace across devices without changing global RNG state.
        q = min(k + oversample, min(source.shape))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        omega = torch.randn(source.shape[1], q, generator=generator,
                            dtype=torch.float32).to(source.device)
        basis = source @ omega
        for _ in range(niter):
            basis = torch.linalg.qr(basis, mode="reduced")[0]
            right_basis = torch.linalg.qr(source.T @ basis, mode="reduced")[0]
            basis = source @ right_basis
        basis = torch.linalg.qr(basis, mode="reduced")[0]
        small = basis.T @ source
        small_left, singular, right = torch.linalg.svd(small, full_matrices=False)
        left = basis @ small_left
        vh = right[:k]
    else:
        raise ValueError(method)
    u = (left[:, :k] * singular[:k]).to(torch.bfloat16)
    v = (vh @ invroot if invroot is not None else vh).to(torch.bfloat16).contiguous()
    energy = float(singular[:k].square().sum() / source_energy)
    result = (u, v, singular, energy)
    return result + (source_energy,) if return_source_energy else result


def activation_loss(weight, approximation, covariance):
    error = weight.float() - approximation.float()
    return float((error @ covariance.float() * error).sum().clamp_min(0) /
                 (weight.float() @ covariance.float() * weight.float()).sum().clamp_min(1e-30))


def refinement_geometry(weights, covariances, g, *, individual_signs=None,
                        projected_covariances=None, projected_energies=None):
    g32 = g.float()
    cg = (projected_covariances if projected_covariances is not None else
          [c.float() @ g32 for c in covariances])
    kg = [g32.T @ x for x in cg]
    diagonal = (projected_energies if projected_energies is not None else
                [(x * g32).sum(0) for x in cg])
    denominator = [(w.float() @ c.float() * w.float()).sum().clamp_min(1e-30)
                   for w, c in zip(weights, covariances)]
    individual = torch.stack(individual_signs) if individual_signs is not None else torch.stack(
        [(w.float().to(g.dtype) @ g >= 0) for w in weights])
    return {"cg": cg, "gram": kg, "diagonal": diagonal,
            "denominator": denominator, "individual": individual, "g": g32}


def refine_group(signs, weights, covariances, scales, factors, g, *, sweeps=1,
                 block_size=32, row_chunk=64, conflict_only=True, geometry=None):
    """Vectorized block descent with exact group-loss acceptance per row.

    Candidate flips use the exact single-bit delta. Simultaneous candidates in
    a block are accepted only after their full cross terms are evaluated.
    """
    if sweeps == 0:
        return signs, {"accepted_flips": 0, "candidate_flips": 0, "loss_before": None,
                       "loss_after": None}
    h = signs.clone()
    rows, d = h.shape
    row_chunk = row_chunk or rows
    a = SQRT_PI_HALF / d
    geometry = geometry if geometry is not None else refinement_geometry(weights, covariances, g)
    cg, kg = geometry["cg"], geometry["gram"]
    diagonal, denominator = geometry["diagonal"], geometry["denominator"]
    individual, g32 = geometry["individual"], geometry["g"]
    target = [w.float() - u.float() @ v.float() for w, (u, v) in zip(weights, factors)]

    def objective(code):
        b = a * (code.float().mul(2).sub(1) @ g32.T)
        return sum(float(((t - s[:, None] * b) @ c * (t - s[:, None] * b)).sum() / den)
                   for t, s, c, den in zip(target, scales, covariances, denominator))

    before = objective(h)
    candidates = torch.zeros((), dtype=torch.int64, device=signs.device)
    accepted = torch.zeros((), dtype=torch.int64, device=signs.device)
    for _ in range(sweeps):
        for start in range(0, rows, row_chunk):
            end = min(start + row_chunk, rows)
            old_sign = h[start:end].float().mul(2).sub(1)
            current_b = a * old_sign @ g32.T
            ecg = [(t[start:end] - s[start:end, None] * current_b) @ x
                   for t, s, x in zip(target, scales, cg)]
            for lo in range(0, d, block_size):
                hi = min(lo + block_size, d)
                sign_block = old_sign[:, lo:hi]
                dh = -2 * sign_block
                delta = torch.zeros_like(dh)
                for s, ec, diag, den in zip(scales, ecg, diagonal, denominator):
                    amplitude = a * s[start:end, None] * dh
                    delta += (-2 * amplitude * ec[:, lo:hi] +
                              amplitude.square() * diag[lo:hi]) / den
                mask = delta < 0
                if conflict_only:
                    conflict = (individual[:, start:end, lo:hi] != individual[0:1, start:end, lo:hi]).any(0)
                    mask &= conflict
                candidates.add_(mask.sum())
                proposal = dh * mask
                total = torch.zeros(end - start, device=signs.device)
                for s, ec, k, den in zip(scales, ecg, kg, denominator):
                    amp = a * s[start:end, None] * proposal
                    total += (-2 * (amp * ec[:, lo:hi]).sum(1) +
                              (amp @ k[lo:hi, lo:hi] * amp).sum(1)) / den
                take = (total < -1e-10) & mask.any(1)
                proposal *= take[:, None]
                accepted.add_((proposal != 0).sum())
                old_sign[:, lo:hi] += proposal
                h[start:end, lo:hi] = old_sign[:, lo:hi] > 0
                for e, s in enumerate(scales):
                    ecg[e] -= (a * s[start:end, None] * proposal) @ kg[e][lo:hi, :]
    after = objective(h)
    candidate_count, accepted_count = int(candidates.item()), int(accepted.item())
    if after > before + 1e-6:
        return signs, {"accepted_flips": 0, "candidate_flips": candidate_count,
                       "loss_before": before, "loss_after": before,
                       "refinement_objective_rollback": True}
    return h, {"accepted_flips": accepted_count, "candidate_flips": candidate_count,
               "loss_before": before, "loss_after": after,
               "refinement_objective_rollback": False}


def _seeds(config, layer, key, group_id):
    return int(config["bundle_seed"] + layer * 1000003 + KEYS.index(key) * 9176 + group_id)


def _svd_seed(config, layer, key, expert_id, case, rank):
    return int(config.get("svd_seed", 0) + layer * 1000003 + KEYS.index(key) * 9176 +
               expert_id * 65537 + ord(case) * 4099 + rank)


@torch.no_grad()
def compress_layer(block, layer, calibration, config, banks, heldout=None, profile=None,
                   progress=None):
    """Compress one layer; covariance and original weights die with this call."""
    n_experts = len(block.experts)
    k = config["group_size"]
    if n_experts % k:
        raise ValueError("group size does not divide expert count")
    ranks = sorted(set(config.get("ranks", [config["rank"]])))
    if any(rank < 0 for rank in ranks):
        raise ValueError("rank must be nonnegative")
    max_rank = max(ranks)
    base = {case: {"format": FORMAT, "layer": layer, "case": case, "projections": {}}
            for case in ("A", "B")}
    factor = {case: {"format": FORMAT, "layer": layer, "case": case, "rank": max_rank,
                     "projections": {}} for case in ("C", "D")}
    factor["E"] = {rank: {"format": FORMAT, "layer": layer, "case": "E", "rank": rank,
                          "projections": {}} for rank in ranks}
    refined = {rank: {"format": FORMAT, "layer": layer, "case": "E", "rank": rank,
                      "projections": {}} for rank in ranks}
    records, spectra = [], {}
    shared_input_cache = {}
    shared_g32 = None

    def stage_start():
        if profile is None:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def stage_end(name, started):
        if started is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            profile[name] = profile.get(name, 0.) + time.perf_counter() - started

    for key, attr in PROJECTIONS.items():
        if progress is not None:
            progress(layer, key, -1, n_experts // k)
        if key == "down":
            shared_input_cache.clear()
            shared_g32 = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        encoding_started = stage_start()
        weights = [getattr(expert, attr).weight.detach() for expert in block.experts]
        rows, cols = weights[0].shape
        bank = bank_for(banks, cols, config["qjl_d_factor"], config["projection_seed"])
        g = bank.get(cols, weights[0].device)
        if shared_g32 is None:
            shared_g32 = g.float()
        g32 = shared_g32
        signs = [torch.cat([(w[start:start + config["row_chunk"]].float().to(g.dtype) @ g >= 0)
                            for start in range(0, rows, config["row_chunk"])]) for w in weights]
        packed = [pack_signs(h).cpu() for h in signs]
        groups = make_groups(packed, bank.d, k, config["pairing"],
                             config["pairing_seed"] + layer * 1009 + KEYS.index(key) * 9176)
        stage_end("encoding_and_grouping_seconds", encoding_started)
        parts = {}
        for case in ("A", "B"):
            parts[case] = {"shape": (rows, cols), "d": bank.d, "groups": groups,
                           "sketches": {}, "group_id": {}, "scales": {}}
        parts_e = {rank: {"shape": (rows, cols), "d": bank.d, "groups": groups,
                          "sketches": {}, "group_id": {}, "scales": {}} for rank in ranks}
        for case in ("C", "D"):
            factor[case]["projections"][key] = {}
        for rank in ranks:
            factor["E"][rank]["projections"][key] = {}
        for group_id, members in enumerate(groups):
            started = time.perf_counter()
            entries = [calibration["experts"].get(i, {}).get(key) for i in members]
            covariance_started = stage_start()
            covariance_items = []
            for expert_id, entry in zip(members, entries):
                item = shared_input_cache.get(expert_id) if key in ("gate", "up") else None
                if item is None:
                    c_i = (entry["covariance"].to(g.device).float() if entry is not None and
                           entry.get("count", 0) else torch.eye(cols, device=g.device))
                    cg_i = c_i @ g32
                    eig_started = stage_start()
                    roots_i = (regularized_roots(c_i, config["ridge"])
                               if "D" in config["cases"] or "E" in config["cases"] else None)
                    stage_end("covariance_eigendecomposition_seconds", eig_started)
                    item = {"covariance": c_i, "cg": cg_i,
                            "projected_energy": (cg_i * g32).sum(0),
                            "roots": roots_i}
                    if key in ("gate", "up"):
                        shared_input_cache[expert_id] = item
                covariance_items.append(item)
            c = [item["covariance"] for item in covariance_items]
            roots = [item["roots"] for item in covariance_items]
            cg = [item["cg"] for item in covariance_items]
            p = [item["projected_energy"] for item in covariance_items]
            stage_end("covariance_eigendecomposition_and_projection_seconds", covariance_started)
            held_entries = ([heldout["experts"].get(i, {}).get(key) for i in members]
                            if heldout is not None else [None] * len(members))
            held_c = [(entry["covariance"].to(g.device).float() if entry is not None and
                       entry.get("count", 0) else None) for entry in held_entries]
            w = [weights[i].float() for i in members]
            s = [x.norm(dim=1) for x in w]
            members_sign = [signs[i] for i in members]
            base_started = stage_start()
            majority = vote(members_sign, _seeds(config, layer, key, group_id))
            aware = vote(members_sign, _seeds(config, layer, key, group_id),
                         [si[:, None].square() * pi[None, :] for si, pi in zip(s, p)])
            code = {"A": majority, "B": aware}
            scales = {}
            for case in ("A", "B"):
                unscaled = unscaled_base(code[case], g)
                scales[case] = [optimized_scale(wi, unscaled, ci) for wi, ci in zip(w, c)]
                parts[case]["sketches"][group_id] = pack_signs(code[case]).cpu()
            residual = {case: [wi - si[:, None] * unscaled_base(code[case], g)
                               for wi, si in zip(w, scales[case])]
                        for case in ("A", "B")}
            stage_end("base_voting_and_scale_seconds", base_started)
            f = {case: [] for case in ("C", "D")}
            source_energies = {}
            pair_agreement = (sum(float((members_sign[i] == members_sign[j]).float().mean())
                                  for i in range(len(members)) for j in range(i + 1, len(members))) /
                              (len(members) * (len(members) - 1) / 2))
            for local, expert_id in enumerate(members):
                requested = []
                if "C" in config["cases"]:
                    requested.append(("C", "B", None))
                if "D" in config["cases"] or "E" in config["cases"]:
                    requested.append(("D", "B", c[local]))
                for case, base_case, covariance in requested:
                    svd_started = stage_start()
                    u, v, singular, _, spectrum_energy = svd_factors(residual[base_case][local],
                        max_rank, covariance, config["ridge"], config["svd_method"],
                        oversample=config.get("svd_oversample", 16),
                        roots=roots[local] if covariance is not None else None,
                        niter=config.get("svd_niter", 2),
                        seed=_svd_seed(config, layer, key, expert_id, case, max_rank),
                        return_source_energy=True)
                    stage_end(f"svd_{case}_seconds", svd_started)
                    f[case].append((u, v))
                    source_energies[(case, local)] = float(spectrum_energy)
                    spectra[f"{key}_{expert_id}_{case}"] = singular.cpu().numpy()
                    factor[case]["projections"][key][expert_id] = {"u": u.cpu(), "v": v.cpu()}
                    # Randomized SVD returns only its captured singular values.
                    # Normalize retention by the full source energy, not by
                    # that truncated list.
                    for rank in ranks:
                        ui, vi = u[:, :rank], v[:rank]
                        energy = float(singular[:rank].square().sum() / spectrum_energy)
                        approx = scales[base_case][local][:, None] * unscaled_base(code[base_case], g) + ui.float() @ vi.float()
                        records.append({"layer": layer, "projection": key, "expert": expert_id,
                                        "group": group_id, "case": case, "rank": rank,
                                        "pre_bundle_bit_agreement": pair_agreement,
                                        "bit_preservation": float((members_sign[local] == aware).float().mean()),
                                        "base_weight_relative_error": float(residual[base_case][local].norm() /
                                                                            w[local].norm().clamp_min(1e-30)),
                                        "residual_weight_ratio": float(residual[base_case][local].norm() /
                                                                       w[local].norm().clamp_min(1e-30)),
                                        "base_activation_loss": activation_loss(w[local],
                                            scales[base_case][local][:, None] * unscaled_base(aware, g), c[local]),
                                        "residual_activation_ratio": math.sqrt(activation_loss(w[local],
                                            scales[base_case][local][:, None] * unscaled_base(aware, g), c[local])),
                                        "svd_energy": energy,
                                        "theoretical_tail_error": math.sqrt(max(0., 1. - energy)),
                                        "spectrum_file": f"spectrum_layer_{layer}.npz:{key}_{expert_id}_{case}",
                                        "spectrum_source_energy": source_energies[(case, local)],
                                        "spectrum_is_estimate": config["svd_method"] == "randomized",
                                        "activation_loss": activation_loss(w[local], approx, c[local]),
                                        "heldout_activation_loss": (activation_loss(w[local], approx, held_c[local])
                                                                    if held_c[local] is not None else None),
                                        "weight_relative_error": float((w[local] - approx).norm() /
                                                                       w[local].norm().clamp_min(1e-30))})
            if "E" in config["cases"]:
                geometry_started = stage_start()
                geometry = (refinement_geometry(w, c, g, individual_signs=members_sign,
                             projected_covariances=cg, projected_energies=p)
                            if config["refine_sweeps"] else None)
                stage_end("refinement_geometry_seconds", geometry_started)
                for rank in ranks:
                    d_factors = [(u[:, :rank], v[:rank]) for u, v in f["D"]]
                    refine_started = stage_start()
                    refine_wall_started = time.perf_counter()
                    refined_code, refinement = refine_group(aware, w, c, scales["B"], d_factors, g,
                        sweeps=config["refine_sweeps"], block_size=config["refine_block"],
                        row_chunk=config.get("refine_row_chunk", config["row_chunk"]),
                        conflict_only=config["conflict_only"],
                        geometry=geometry)
                    refine_wall_seconds = time.perf_counter() - refine_wall_started
                    stage_end("bit_refinement_seconds", refine_started)
                    unscaled = unscaled_base(refined_code, g)
                    refined_scale = [optimized_scale(wi - ui.float() @ vi.float(), unscaled, ci)
                                     for wi, ci, (ui, vi) in zip(w, c, d_factors)]
                    proposal = []
                    for local, expert_id in enumerate(members):
                        ri = w[local] - refined_scale[local][:, None] * unscaled
                        svd_started = stage_start()
                        u, v, singular, energy, source_energy = svd_factors(ri, rank, c[local],
                            config["ridge"], config["svd_method"],
                            oversample=config.get("svd_oversample", 16),
                            roots=roots[local], niter=config.get("svd_niter", 2),
                            seed=_svd_seed(config, layer, key, expert_id, "E", rank),
                            return_source_energy=True)
                        stage_end("svd_E_seconds", svd_started)
                        before = activation_loss(w[local], scales["B"][local][:, None] *
                            unscaled_base(aware, g) + d_factors[local][0].float() @ d_factors[local][1].float(), c[local])
                        after = activation_loss(w[local], refined_scale[local][:, None] * unscaled +
                                                u.float() @ v.float(), c[local])
                        proposal.append((u, v, energy, before, after, singular,
                                         float(source_energy)))
                    rollback = sum(item[4] for item in proposal) > sum(item[3] for item in proposal) + 1e-8
                    parts_e[rank]["sketches"][group_id] = pack_signs(aware if rollback else refined_code).cpu()
                    for local, expert_id in enumerate(members):
                        u, v, energy, before, after, singular, source_energy = proposal[local]
                        if rollback:
                            u, v = d_factors[local]
                            singular = torch.from_numpy(spectra[f"{key}_{expert_id}_D"])
                            source_energy = source_energies[("D", local)]
                            energy = float(singular[:rank].square().sum() /
                                           max(source_energy, 1e-30))
                        parts_e[rank]["scales"][expert_id] = (scales["B"][local] if rollback
                                                               else refined_scale[local]).cpu()
                        factor["E"][rank]["projections"][key][expert_id] = {"u": u.cpu(), "v": v.cpu()}
                        spectra[f"{key}_{expert_id}_E_r{rank}"] = singular.cpu().numpy()
                        approximation = (scales["B"][local][:, None] * unscaled_base(aware, g)
                                         if rollback else refined_scale[local][:, None] * unscaled) + u.float() @ v.float()
                        final_base = (scales["B"][local][:, None] * unscaled_base(aware, g)
                                      if rollback else refined_scale[local][:, None] * unscaled)
                        base_loss = activation_loss(w[local], final_base, c[local])
                        records.append({"layer": layer, "projection": key, "expert": expert_id,
                                        "group": group_id, "case": "E", "rank": rank,
                                        "pre_bundle_bit_agreement": pair_agreement,
                                        "bit_preservation": float((members_sign[local] ==
                                            (aware if rollback else refined_code)).float().mean()),
                                        "svd_energy": energy,
                                        "base_weight_relative_error": float((w[local] -
                                            final_base).norm() /
                                             w[local].norm().clamp_min(1e-30)),
                                        "residual_weight_ratio": float((w[local] - final_base).norm() /
                                                                       w[local].norm().clamp_min(1e-30)),
                                        "base_activation_loss": base_loss,
                                        "residual_activation_ratio": math.sqrt(base_loss),
                                        "theoretical_tail_error": math.sqrt(max(0., 1. - energy)),
                                        "spectrum_file": f"spectrum_layer_{layer}.npz:{key}_{expert_id}_E_r{rank}",
                                        "spectrum_source_energy": source_energy,
                                        "spectrum_is_estimate": config["svd_method"] == "randomized",
                                        "activation_loss": before if rollback else after,
                                        "heldout_activation_loss": (activation_loss(w[local], approximation, held_c[local])
                                                                    if held_c[local] is not None else None),
                                        "weight_relative_error": float((w[local] - approximation).norm() /
                                                                       w[local].norm().clamp_min(1e-30)),
                                        "pre_refine_loss": before, "post_refine_loss": after,
                                        "refine_rollback": rollback,
                                        "accepted_flip_ratio": refinement["accepted_flips"] / (rows * bank.d),
                                        "refinement_wall_seconds": refine_wall_seconds,
                                        **refinement})
            for local, expert_id in enumerate(members):
                for case in ("A", "B"):
                    parts[case]["scales"][expert_id] = scales[case][local].cpu()
                    approx = scales[case][local][:, None] * unscaled_base(code[case], g)
                    records.append({"layer": layer, "projection": key, "expert": expert_id,
                                    "group": group_id, "case": case, "rank": 0,
                                    "pre_bundle_bit_agreement": pair_agreement,
                                    "bit_preservation": float((members_sign[local] == code[case]).float().mean()),
                                    "activation_loss": activation_loss(w[local], approx, c[local]),
                                    "heldout_activation_loss": (activation_loss(w[local], approx, held_c[local])
                                                                if held_c[local] is not None else None),
                                    "weight_relative_error": float((w[local] - approx).norm() /
                                                                   w[local].norm().clamp_min(1e-30)),
                                    "base_weight_relative_error": float(residual[case][local].norm() /
                                                                        w[local].norm().clamp_min(1e-30)),
                                    "base_activation_loss": activation_loss(w[local], approx, c[local]),
                                    "residual_weight_ratio": float(residual[case][local].norm() /
                                                                   w[local].norm().clamp_min(1e-30)),
                                    "residual_activation_ratio": math.sqrt(activation_loss(w[local], approx, c[local])),
                                    "calibration_count": entries[local]["count"] if entries[local] else 0})
                for case in parts:
                    parts[case]["group_id"][expert_id] = group_id
                for part in parts_e.values():
                    part["group_id"][expert_id] = group_id
            group_seconds = time.perf_counter() - started
            for row in records:
                if row["layer"] == layer and row["projection"] == key and row["group"] == group_id:
                    row["group_compression_seconds"] = group_seconds
            if progress is not None:
                progress(layer, key, group_id, len(groups))
        for case in ("A", "B"):
            base[case]["projections"][key] = parts[case]
        for rank in ranks:
            refined[rank]["projections"][key] = parts_e[rank]
    for row in records:
        row["activation_relative_l2_error"] = math.sqrt(max(0., row["activation_loss"]))
        heldout_loss = row.get("heldout_activation_loss")
        row["heldout_relative_l2_error"] = (math.sqrt(max(0., heldout_loss))
                                             if heldout_loss is not None else None)
    return base, factor, refined, records, spectra


def install_layer(block, base_state, factor_state, config, banks, backend):
    if base_state.get("format") != FORMAT or (factor_state is not None and
                                             factor_state.get("format") != FORMAT):
        raise ValueError("wrong activation bundle format")
    device = next(block.parameters()).device
    n = len(block.experts)
    shared = {}
    for key in KEYS:
        part = base_state["projections"][key]
        validate_groups(part["groups"], n, config["group_size"])
        shared[key] = {int(i): t.to(device) for i, t in part["sketches"].items()}
    experts = []
    for expert_id in range(n):
        bases, corrections = {}, {}
        for key in KEYS:
            part = base_state["projections"][key]
            cols = part["shape"][1]
            bank = bank_for(banks, cols, config["qjl_d_factor"], config["projection_seed"])
            bases[key] = PackedResidual(shared[key][part["group_id"][expert_id]],
                part["scales"][expert_id].to(device), cols, bank.d, bank, backend)
            if factor_state is not None:
                pair = factor_state["projections"][key][expert_id]
                corrections[key] = LowRankCorrection(pair["u"].to(device), pair["v"].to(device))
        experts.append(BundledBaseExpert(bases, corrections, backend))
    router = block.gate
    shared_expert = getattr(block, "shared_expert", None)
    shared_gate = getattr(block, "shared_expert_gate", None)
    top_k = block.top_k
    _install(block, experts)
    if (block.gate is not router or getattr(block, "shared_expert", None) is not shared_expert or
            getattr(block, "shared_expert_gate", None) is not shared_gate or block.top_k != top_k):
        raise AssertionError("router or shared expert changed")


def load_checkpoint(model, checkpoint_dir, case, backend, rank=None):
    if case not in CASES:
        raise ValueError(case)
    checkpoint_dir = Path(checkpoint_dir)
    config = json.loads((checkpoint_dir / "manifest.json").read_text())
    if config.get("format") != FORMAT:
        raise ValueError("wrong checkpoint format")
    ranks = config.get("ranks", [config["rank"]])
    if rank is None:
        rank = max(ranks)
    if case in ("C", "D", "E") and rank not in ranks:
        raise ValueError(f"rank {rank} is unavailable in this checkpoint")
    banks = {}
    model._activation_bundle_banks = banks
    for layer in config["layers"]:
        base_name = (f"base_E_r{rank}_layer_{layer}.pt" if case == "E" else
                     f"base_{'A' if case == 'A' else 'B'}_layer_{layer}.pt")
        base = torch.load(checkpoint_dir / base_name,
                          map_location="cpu", weights_only=False)
        factor = (torch.load(checkpoint_dir / f"factor_{case}_r{rank}_layer_{layer}.pt",
                             map_location="cpu", weights_only=False) if case in ("C", "D", "E") else None)
        install_layer(model.model.layers[layer].mlp, base, factor, config, banks, backend)
    return config


def checkpoint_sizes(checkpoint_dir, config, case, rank=None):
    root = Path(checkpoint_dir)
    if rank is None:
        rank = max(config.get("ranks", [config["rank"]]))
    disk = (root / "manifest.json").stat().st_size
    sketch = scales = lowrank = count = 0
    for layer in config["layers"]:
        path = (root / f"base_E_r{rank}_layer_{layer}.pt" if case == "E" else
                root / f"base_{'A' if case == 'A' else 'B'}_layer_{layer}.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        disk += path.stat().st_size
        for part in state["projections"].values():
            rows, cols = part["shape"]
            count += len(part["scales"]) * rows * cols
            sketch += sum(t.numel() * t.element_size() for t in part["sketches"].values())
            scales += sum(t.numel() * t.element_size() for t in part["scales"].values())
        if case in ("C", "D", "E"):
            path = root / f"factor_{case}_r{rank}_layer_{layer}.pt"
            factors = torch.load(path, map_location="cpu", weights_only=False)
            disk += path.stat().st_size
            lowrank += sum(t.numel() * t.element_size() for part in
                           factors["projections"].values() for expert in part.values()
                           for t in expert.values())
    return {"shared_sketch_bytes": sketch, "row_scale_bytes": scales,
            "lowrank_residual_bytes": lowrank, "actual_checkpoint_bytes": disk,
            "metadata_and_container_bytes": disk - sketch - scales - lowrank,
            "routed_weight_count": count, "effective_bpw": 8 * disk / count,
            "bf16_storage_reduction": 1 - disk / (2 * count)}
