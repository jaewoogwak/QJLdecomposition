"""Contract tests for pairwise residual sketch sharing."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.codebook_model import encode_projection
from rescode.qjl.core import (ProjectionBank, pack_signs, qjl_reference, qjl_triton,
                              unpack_signs)
from rescode.qjl.residual_bundle_model import (
    bundle_signs, checkpoint_sizes, compress_model, encode_projection_rows,
    gpu_resident_breakdown, load_checkpoint, make_pairs, save_checkpoint,
    validate_pairs)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/QJL"))
from analyze_residual_bundle import analyze


def test_none_matches_existing_exponent_mantissa_qjl():
    torch.manual_seed(4)
    weight = (torch.randn(17, 16) * .1).to(torch.bfloat16)
    bank = ProjectionBank(32, 7)
    _, old, old_state, _ = encode_projection(weight, bank, "reference",
        base_method="exponent-mantissa", base_row_chunk=5, qjl_row_chunk=5)
    base, packed, norms = encode_projection_rows(weight, bank, row_chunk=5)
    assert torch.equal(base["beta"], old_state["base"]["beta"])
    assert torch.equal(packed, old_state["packed"])
    assert torch.equal(norms, old_state["norms"])
    x = torch.randn(3, 16).to(torch.bfloat16)
    z = x.to(torch.float16) @ bank.get(16, "cpu")
    assert torch.allclose(qjl_reference(z, packed, norms, bank.d), old(x).float(),
                          atol=.01, rtol=.01)


def test_bundle_bits_ties_and_preservation_identity():
    a = torch.tensor([[1, 1, 0, 0, 1, 0, 1, 0, 1]], dtype=torch.bool)
    b = torch.tensor([[1, 0, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.bool)
    p, q = pack_signs(a), pack_signs(b)
    assert torch.equal(unpack_signs(p, 9), a)
    assert torch.equal(unpack_signs(q, 9), b)
    voted = unpack_signs(bundle_signs(p, q, 9, "majority", 42), 9)
    assert torch.equal(voted, unpack_signs(bundle_signs(p, q, 9, "majority", 42), 9))
    assert torch.equal(voted[a == b], a[a == b])
    zi = torch.tensor([[2., 3., .1, 1., 4., .5, 1., 2., 1.]])
    zj = torch.tensor([[1., 1., 5., 2., 2., 2., 3., 1., 3.]])
    magnitude = unpack_signs(bundle_signs(p, q, 9, "magnitude-aware", 42,
        projections=lambda _start, _end: (zi, zj)), 9)
    assert torch.equal(magnitude[a == b], a[a == b])
    assert magnitude[0, 1] == a[0, 1]
    assert magnitude[0, 2] == b[0, 2]
    tie = unpack_signs(bundle_signs(p, q, 9, "magnitude-aware", 42,
        projections=lambda _start, _end: (torch.ones_like(zi), torch.ones_like(zj))), 9)
    assert torch.equal(tie, voted)
    agreement = (a == b).float().mean()
    preservation = ((voted == a).float().mean() + (voted == b).float().mean()) / 2
    assert torch.allclose(preservation, (1 + agreement) / 2)


def test_pairing_exact_partition_and_seeds():
    sketches = {i: pack_signs(torch.tensor([[i // 2 == k for k in range(9)]],
                                           dtype=torch.bool)) for i in range(6)}
    for method in ("random", "similarity"):
        pairs = make_pairs(sketches, 9, method, 19)
        validate_pairs(pairs, 6)
        assert pairs == make_pairs(sketches, 9, method, 19)
    assert make_pairs(sketches, 9, "similarity", 19) == [(0, 1), (2, 3), (4, 5)]
    with pytest.raises(ValueError):
        validate_pairs([(0, 1), (1, 2), (3, 4)], 6)


def test_bundle_methods_use_identical_pairs():
    torch.manual_seed(101)
    source = TinyModel()
    majority, _ = compress_model(copy.deepcopy(source), config("majority"))
    magnitude, _ = compress_model(copy.deepcopy(source), config("magnitude-aware"))
    assert majority[0]["pairs"] == magnitude[0]["pairs"]


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(16, 60, bias=False, dtype=torch.bfloat16)
        self.experts = nn.ModuleList(TinyExpert() for _ in range(60))
        self.shared_expert = TinyExpert()
        self.shared_expert_gate = nn.Linear(16, 1, bias=False, dtype=torch.bfloat16)
        self.top_k = 2
        self.norm_topk_prob = False

    def forward(self, hidden):
        flat = hidden.reshape(-1, 16)
        logits = self.gate(flat)
        weights, selected = torch.topk(F.softmax(logits, dim=-1, dtype=torch.float), 2, -1)
        output = torch.zeros_like(flat)
        for expert_id in selected.unique().tolist():
            token, slot = torch.where(selected == expert_id)
            y = self.experts[expert_id](flat[token])
            output.index_add_(0, token, y * weights[token, slot, None].to(y.dtype))
        output += torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return output.reshape(hidden.shape), logits


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.mlp = TinyBlock()
        self.model.layers = nn.ModuleList([layer])


def config(method):
    return {"layers": [0], "bundle_method": method, "pairing": "similarity",
            "qjl_d_factor": 2, "projection_seed": 3, "pairing_seed": 9,
            "bundle_seed": 15, "row_chunk": 4, "backend": "reference"}


@pytest.mark.parametrize("method", ["none", "majority", "magnitude-aware"])
def test_compress_reload_eval_and_analysis_smoke(tmp_path, method):
    torch.manual_seed(29)
    original = TinyModel()
    compressed = copy.deepcopy(original)
    x = torch.randn(2, 16).to(torch.bfloat16)
    router = compressed.model.layers[0].mlp.gate
    shared = compressed.model.layers[0].mlp.shared_expert
    original_logits = original.model.layers[0].mlp(x)[1]
    state, records = compress_model(compressed, config(method))
    assert compressed.model.layers[0].mlp.gate is router
    assert compressed.model.layers[0].mlp.shared_expert is shared
    assert torch.equal(compressed.model.layers[0].mlp(x)[1], original_logits)
    before = compressed.model.layers[0].mlp(x)[0]
    path = tmp_path / "checkpoint.pt"
    sizes = save_checkpoint(state, config(method), path)
    assert sizes == checkpoint_sizes(state, path)
    restored = copy.deepcopy(original)
    payload = load_checkpoint(restored, path, backend="reference")
    after = restored.model.layers[0].mlp(x)[0]
    assert torch.equal(before, after)
    for key in ("gate", "up", "down"):
        assert payload["layers"][0]["pairs"][key] == state[0]["pairs"][key]
        validate_pairs(state[0]["pairs"][key], 60)
    if method != "none":
        for key in ("gate", "up", "down"):
            assert len(state[0]["sketches"][key]) == 30
            pair = state[0]["pairs"][key][0]
            ra = restored.model.layers[0].mlp._compact_experts[pair[0]].residuals[key]
            rb = restored.model.layers[0].mlp._compact_experts[pair[1]].residuals[key]
            assert ra.packed.untyped_storage().data_ptr() == rb.packed.untyped_storage().data_ptr()
        individual_bytes = sum(
            part["norms"].numel() * ((part["base"]["shape"][1] * 2 + 7) // 8)
            for expert in state[0]["experts"].values() for part in expert.values())
        assert sizes["qjl_sketch_bytes"] * 2 == individual_bytes
    # Synthetic evaluation and independent analysis use the saved pair map/sketch.
    args = SimpleNamespace(layers="0", projections="gate,up,down", pairs="all",
                           row_chunk=4, synthetic_samples=2, input_seed=11)
    analysis = analyze(original, payload, args)
    assert len(analysis) == 60 * 3 * 3
    assert all(row["weight_output_relative_l2"] >= 0 for row in analysis)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_reference_triton_output_and_unique_gpu_storage():
    torch.manual_seed(31)
    model = TinyModel().cuda()
    state, _ = compress_model(model, {**config("majority"), "backend": "triton"})
    part = model.model.layers[0].mlp._compact_experts[0].residuals["gate"]
    x = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    p = part.bank.get(16, "cuda")
    z = x.to(p.dtype) @ p
    assert torch.allclose(qjl_reference(z, part.packed, part.norms, part.d),
                          qjl_triton(z, part.packed, part.norms, part.d), atol=.02, rtol=.02)
    breakdown = gpu_resident_breakdown(model)
    expected = sum(t.numel() for group in state[0]["sketches"].values()
                   for t in group.values())
    assert breakdown["gpu_resident_qjl_sketch_bytes"] == expected
