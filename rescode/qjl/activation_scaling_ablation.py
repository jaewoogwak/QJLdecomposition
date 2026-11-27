"""Row-L2 scale ablation for an existing activation-aware bundled QJL base."""

from __future__ import annotations

import math

import torch

from .activation_bundle_model import KEYS, unscaled_base, validate_groups
from .codebook_model import bank_for
from .core import unpack_signs
from .exponent_model import PROJECTIONS


MATCH_KEYS = ("format", "model_path", "layers", "qjl_d_factor", "group_size",
              "pairing", "projection_seed", "pairing_seed", "bundle_seed",
              "calibration_source")


def validate_source(source: dict, baseline: dict, cache_manifest: dict, layers: list[int]):
    """A/B measurements and reused sketches must have identical provenance."""
    for key in MATCH_KEYS:
        if source.get(key) != baseline.get(key):
            raise ValueError(f"source and optimized A/B baseline differ in {key}")
    if source.get("group_size") != 3 or source.get("qjl_d_factor") != 2:
        raise ValueError("this ablation requires group size 3 and D=2n")
    if source.get("calibration_source") != cache_manifest:
        raise ValueError("covariance cache manifest differs from the source checkpoint")
    if not set(layers).issubset(source.get("layers", [])):
        raise ValueError("requested layers are absent from the source checkpoint")


def replace_row_scales(states: dict, block) -> dict:
    """Replace only A/B row scales; leave groups, packed codes and seeds intact."""
    n_experts = len(block.experts)
    for key, attr in PROJECTIONS.items():
        norms = [torch.linalg.vector_norm(
            getattr(expert, attr).weight.detach().float(), dim=1).cpu()
            for expert in block.experts]
        for state in states.values():
            part = state["projections"][key]
            validate_groups(part["groups"], n_experts, 3)
            for expert_id, norm in enumerate(norms):
                if part["scales"][expert_id].shape != norm.shape:
                    raise ValueError(f"{key} expert {expert_id} row count differs")
                part["scales"][expert_id] = norm
    return states


def _loss(weight: torch.Tensor, base: torch.Tensor, covariance: torch.Tensor,
          denominator: torch.Tensor) -> float:
    residual = weight - base
    numerator = (residual @ covariance * residual).sum().clamp_min(0)
    return float(numerator / denominator.clamp_min(1e-30))


@torch.no_grad()
def layer_metrics(block, states: dict, calibration: dict, heldout: dict,
                  config: dict, banks: dict, layer: int) -> list[dict]:
    """Measure both scales against the same original weights and covariances."""
    rows = []
    for key in KEYS:
        attr = PROJECTIONS[key]
        parts = {case: state["projections"][key] for case, state in states.items()}
        dimensions = {tuple(part["shape"]) for part in parts.values()}
        if len(dimensions) != 1:
            raise ValueError(f"{key} A/B dimensions differ")
        output_dim, input_dim = next(iter(dimensions))
        device = getattr(block.experts[0], attr).weight.device
        bank = bank_for(banks, input_dim, config["qjl_d_factor"],
                        config["projection_seed"])
        projection = bank.get(input_dim, device)
        for case, part in parts.items():
            for group_id, members in enumerate(part["groups"]):
                signs = unpack_signs(part["sketches"][group_id].to(device), part["d"])
                base_unit = unscaled_base(signs, projection)
                for expert_id in members:
                    weight = getattr(block.experts[expert_id], attr).weight.detach().float()
                    scale = part["scales"][expert_id].to(device)
                    if weight.shape != (output_dim, input_dim):
                        raise ValueError(f"{key} expert {expert_id} weight shape differs")
                    base = scale[:, None] * base_unit
                    relative_weight_error = float((weight - base).norm() /
                                                  weight.norm().clamp_min(1e-30))
                    entries = (calibration["experts"][expert_id][key],
                               heldout["experts"][expert_id][key])
                    losses = []
                    for entry in entries:
                        if not entry.get("count", 0):
                            losses.append(None)
                            continue
                        covariance = entry["covariance"].to(device).float()
                        denominator = (weight @ covariance * weight).sum()
                        losses.append(_loss(weight, base, covariance, denominator))
                        del covariance
                    cal_loss, heldout_loss = losses
                    rows.append({"layer": layer, "projection": key, "expert": expert_id,
                                 "group": group_id, "case": case + "0", "rank": 0,
                                 "scaling": "original_weight_row_l2",
                                 "weight_relative_error": relative_weight_error,
                                 "calibration_activation_loss": cal_loss,
                                 "heldout_activation_loss": heldout_loss,
                                 "calibration_relative_l2_error": (math.sqrt(cal_loss)
                                                                 if cal_loss is not None else None),
                                 "heldout_relative_l2_error": (math.sqrt(heldout_loss)
                                                               if heldout_loss is not None else None),
                                 "calibration_count": entries[0]["count"],
                                 "heldout_count": entries[1]["count"]})
    return rows
