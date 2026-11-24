"""Loaded Individual QJL row-norm scaling without checkpoint changes."""

import copy
import hashlib

import pytest
import torch
import torch.nn as nn

from rescode.qjl import codebook_model
from rescode.qjl.codebook_model import (encode_projection, load_checkpoint,
    save_checkpoint, scale_loaded_residual_norms)
from rescode.qjl.core import ProjectionBank


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        block = nn.Module()
        block.gate = nn.Linear(64, 60, bias=False, dtype=torch.bfloat16)
        block.shared_expert = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        block.shared_expert_gate = nn.Linear(64, 1, bias=False, dtype=torch.bfloat16)
        block.experts = nn.ModuleList(nn.Identity() for _ in range(60))
        block.top_k = 2
        block.norm_topk_prob = False
        layer.mlp = block
        self.model.layers = nn.ModuleList([layer])


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(32)
    parts = {}
    for key, rows, cols in (("gate", 32, 64), ("up", 32, 64), ("down", 64, 32)):
        weight = (torch.randn(rows, cols) * .1).to(torch.bfloat16)
        bank = ProjectionBank(cols * 2, 7)
        _, _, part, _ = encode_projection(weight, bank, "reference",
                                          base_method="exponent-mantissa",
                                          qjl_row_chunk=16)
        parts[key] = part
    config = {"layers": [0], "qjl_d_factor": 2.0, "projection_seed": 7,
              "backend": "reference", "base_method": "exponent-mantissa"}
    path = tmp_path / "individual.pt"
    save_checkpoint({0: {i: parts for i in range(60)}}, config, path)
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_alpha_one_exact_and_zero_base_only(checkpoint):
    source = TinyModel()
    original_router = source.model.layers[0].mlp.gate
    original_shared = source.model.layers[0].mlp.shared_expert
    baseline, scaled = copy.deepcopy(source), copy.deepcopy(source)
    payload = load_checkpoint(baseline, checkpoint, backend="reference")
    load_checkpoint(scaled, checkpoint, backend="reference")
    scale_loaded_residual_norms(scaled, 1.0)
    x = torch.randn(3, 64).to(torch.bfloat16)
    expected, logits = baseline.model.layers[0].mlp(x)
    actual, new_logits = scaled.model.layers[0].mlp(x)
    assert torch.equal(actual, expected)
    assert torch.equal(new_logits, logits)
    assert payload["config"]["base_method"] == "exponent-mantissa"

    base_only = copy.deepcopy(source)
    router = base_only.model.layers[0].mlp.gate
    shared = base_only.model.layers[0].mlp.shared_expert
    before = digest(checkpoint)
    original_payload = load_checkpoint(base_only, checkpoint, backend="reference")
    expert = base_only.model.layers[0].mlp._compact_experts[0]
    old_base = {k: {name: value.clone() for name, value in base.state_dict().items()}
                for k, base in expert.bases.items()}
    old_sketches = {k: residual.packed.clone() for k, residual in expert.residuals.items()}
    old_norms = {k: original_payload["layers"][0][0][k]["norms"].clone()
                 for k in ("gate", "up", "down")}
    bank_ids = {k: id(residual.bank) for k, residual in expert.residuals.items()}
    scale_loaded_residual_norms(base_only, 0.0)
    for key, sample in (("gate", x), ("up", x),
                        ("down", torch.randn(3, 32).to(torch.bfloat16))):
        assert torch.equal(expert._projection(key, sample), expert.bases[key](sample))
        assert torch.count_nonzero(expert.residuals[key].norms) == 0
        assert torch.equal(expert.residuals[key].packed, old_sketches[key])
        assert torch.equal(original_payload["layers"][0][0][key]["norms"], old_norms[key])
        assert id(expert.residuals[key].bank) == bank_ids[key]
        for name, value in expert.bases[key].state_dict().items():
            assert torch.equal(value, old_base[key][name])
    assert base_only.model.layers[0].mlp.gate is router
    assert base_only.model.layers[0].mlp.shared_expert is shared
    assert original_router is source.model.layers[0].mlp.gate
    assert original_shared is source.model.layers[0].mlp.shared_expert
    assert digest(checkpoint) == before


def test_half_scale_reference_and_reload_independence(checkpoint):
    source = TinyModel()
    full, half = copy.deepcopy(source), copy.deepcopy(source)
    load_checkpoint(full, checkpoint, backend="reference")
    payload = load_checkpoint(half, checkpoint, backend="reference")
    expert_full = full.model.layers[0].mlp._compact_experts[0]
    expert_half = half.model.layers[0].mlp._compact_experts[0]
    original_norm = payload["layers"][0][0]["gate"]["norms"].clone()
    scale_loaded_residual_norms(half, 0.5)
    for key, sample in (("gate", torch.randn(3, 64).to(torch.bfloat16)),
                        ("up", torch.randn(3, 64).to(torch.bfloat16)),
                        ("down", torch.randn(3, 32).to(torch.bfloat16))):
        residual_full = expert_full.residuals[key](sample).float()
        residual_half = expert_half.residuals[key](sample).float()
        assert torch.allclose(residual_half, residual_full * .5, atol=.008, rtol=.015)
        assert torch.allclose(expert_half.residuals[key].norms,
                              expert_full.residuals[key].norms * .5)
    assert torch.equal(payload["layers"][0][0]["gate"]["norms"], original_norm)
    with pytest.raises(RuntimeError):
        scale_loaded_residual_norms(half, 0.75)
    load_checkpoint(half, checkpoint, backend="reference")
    scale_loaded_residual_norms(half, 0.75)
    assert torch.allclose(half.model.layers[0].mlp._compact_experts[0].residuals["gate"].norms,
                          expert_full.residuals["gate"].norms * .75)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_triton_fused_gate_up_scale(checkpoint, monkeypatch):
    torch.manual_seed(55)
    source = TinyModel().cuda()
    reference, fused = copy.deepcopy(source), copy.deepcopy(source)
    load_checkpoint(reference, checkpoint, backend="reference")
    load_checkpoint(fused, checkpoint, backend="triton")
    scale_loaded_residual_norms(reference, 0.5)
    scale_loaded_residual_norms(fused, 0.5)
    calls = []
    original = codebook_model.qjl_triton_fused_gate_up

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(codebook_model, "qjl_triton_fused_gate_up", counted)
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    triton_y = fused.model.layers[0].mlp._compact_experts[0](x)
    reference_y = reference.model.layers[0].mlp._compact_experts[0](x)
    assert calls
    assert torch.allclose(triton_y.float(), reference_y.float(), atol=.06, rtol=.03)
    for key in ("gate", "up", "down"):
        triton_part = fused.model.layers[0].mlp._compact_experts[0].residuals[key]
        ref_part = reference.model.layers[0].mlp._compact_experts[0].residuals[key]
        assert torch.equal(triton_part.norms, ref_part.norms)

    zero = copy.deepcopy(source)
    load_checkpoint(zero, checkpoint, backend="triton")
    scale_loaded_residual_norms(zero, 0.0)
    zero_expert = zero.model.layers[0].mlp._compact_experts[0]
    assert torch.equal(zero_expert(x),
                       zero_expert.bases["down"](
                           torch.nn.functional.silu(zero_expert.bases["gate"](x))
                           * zero_expert.bases["up"](x)))
    for key, sample in (("gate", x), ("up", x),
                        ("down", torch.randn(4, 32, device="cuda", dtype=torch.bfloat16))):
        assert torch.equal(zero_expert._projection(key, sample),
                           zero_expert.bases[key](sample))
