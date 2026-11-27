"""Small numerical and provenance checks for fixed-code row scaling."""

import copy

import pytest
import torch
import torch.nn as nn

from rescode.qjl.activation_bundle_model import FORMAT, activation_loss, unscaled_base
from rescode.qjl.activation_scaling_ablation import (
    layer_metrics, replace_row_scales, validate_source,
)
from rescode.qjl.codebook_model import bank_for
from rescode.qjl.core import pack_signs, unpack_signs


class Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)


def fixture():
    torch.manual_seed(21)
    block = nn.Module()
    block.experts = nn.ModuleList(Expert() for _ in range(3))
    states = {}
    for case in ("A", "B"):
        projections = {}
        for key in ("gate", "up", "down"):
            code = pack_signs(torch.rand(4, 8) > .5)
            projections[key] = {"shape": (4, 4), "d": 8,
                "groups": [[0, 1, 2]], "sketches": {0: code},
                "group_id": {0: 0, 1: 0, 2: 0},
                "scales": {i: torch.full((4,), .25) for i in range(3)}}
        states[case] = {"format": FORMAT, "projections": projections}
    cov = {"experts": {i: {key: {"covariance": torch.eye(4), "count": 7}
                              for key in ("gate", "up", "down")}
                       for i in range(3)}}
    return block, states, cov


def test_only_row_scales_change_and_metrics_are_exact():
    block, states, cov = fixture()
    original = copy.deepcopy(states)
    replace_row_scales(states, block)
    for case in ("A", "B"):
        for key, attr in (("gate", "gate_proj"), ("up", "up_proj"),
                          ("down", "down_proj")):
            part = states[case]["projections"][key]
            assert torch.equal(part["sketches"][0], original[case]["projections"][key]["sketches"][0])
            assert part["groups"] == original[case]["projections"][key]["groups"]
            for i, expert in enumerate(block.experts):
                assert torch.allclose(part["scales"][i],
                                      getattr(expert, attr).weight.float().norm(dim=1))
    config = {"group_size": 3, "qjl_d_factor": 2., "projection_seed": 0}
    rows = layer_metrics(block, states, cov, cov, config, {}, 0)
    assert len(rows) == 3 * 3 * 2
    row = next(item for item in rows if item["case"] == "A0" and
               item["projection"] == "gate" and item["expert"] == 0)
    part = states["A"]["projections"]["gate"]
    bank = bank_for({}, 4, 2., 0)
    code = unpack_signs(part["sketches"][0], 8)
    base = part["scales"][0][:, None] * unscaled_base(code, bank.get(4, "cpu"))
    weight = block.experts[0].gate_proj.weight.detach().float()
    assert row["weight_relative_error"] == pytest.approx(float((weight - base).norm() / weight.norm()))
    assert row["calibration_activation_loss"] == pytest.approx(
        activation_loss(weight, base, torch.eye(4)), abs=1e-6)
    assert row["heldout_relative_l2_error"] == pytest.approx(
        row["calibration_activation_loss"] ** .5)


def test_source_provenance_requires_fixed_codes_and_cache():
    source = {"format": FORMAT, "model_path": "/model", "layers": [0, 1],
              "qjl_d_factor": 2, "group_size": 3, "pairing": "similarity",
              "projection_seed": 0, "pairing_seed": 0, "bundle_seed": 0,
              "calibration_source": {"fingerprint": "same"}}
    validate_source(source, copy.deepcopy(source), source["calibration_source"], [0])
    changed = copy.deepcopy(source)
    changed["bundle_seed"] = 1
    with pytest.raises(ValueError, match="bundle_seed"):
        validate_source(source, changed, source["calibration_source"], [0])
    with pytest.raises(ValueError, match="cache manifest"):
        validate_source(source, source, {"fingerprint": "different"}, [0])
