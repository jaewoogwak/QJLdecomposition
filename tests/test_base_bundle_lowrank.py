"""Pairwise QJL Base and post-bundle SVD residual contracts."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rescode.qjl.base_bundle_model import (FORMAT, compress_layer, install_layer,
    load_checkpoint, resident_bytes, spectrum_metrics)
from rescode.qjl.bundled_model import dense_qjl_base
from rescode.qjl.codebook_model import bank_for
from rescode.qjl.residual_bundle_model import validate_pairs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/QJL"))
from run_base_bundling import arguments


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        block = nn.Module()
        block.gate = nn.Linear(16, 60, bias=False, dtype=torch.bfloat16)
        block.shared_expert = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
        block.shared_expert_gate = nn.Linear(16, 1, bias=False, dtype=torch.bfloat16)
        block.experts = nn.ModuleList(TinyExpert() for _ in range(60))
        block.top_k = 2
        block.norm_topk_prob = False
        layer.mlp = block
        self.model.layers = nn.ModuleList([layer])


def config():
    return {"format": FORMAT, "layers": [0], "qjl_d_factor": 2., "rank": 4,
            "projection_seed": 7, "pairing_seed": 11, "bundle_seed": 13,
            "row_chunk": 4}


def test_explicit_eval_variant_and_legacy_case_alias():
    explicit = arguments(["--mode", "eval", "--eval-variant", "base-plus-lowrank"])
    legacy = arguments(["--mode", "eval", "--case", "B"])
    assert explicit.case == legacy.case == "B"
    assert explicit.eval_variant == legacy.eval_variant == "base-plus-lowrank"
    assert arguments(["--mode", "eval", "--eval-variant", "base-only"]).case == "A"
    with pytest.raises(SystemExit):
        arguments(["--mode", "eval"])
    with pytest.raises(SystemExit):
        arguments(["--mode", "eval", "--case", "A",
                   "--eval-variant", "base-plus-lowrank"])


@pytest.fixture(scope="module")
def encoded():
    torch.manual_seed(101)
    model = TinyModel()
    base, factors, rows, spectra = compress_layer(model.model.layers[0].mlp,
                                                   0, config(), {})
    return model, base, factors, rows, spectra


def test_pairs_spectrum_and_exact_reconstruction(encoded):
    model, base, factors, rows, spectra = encoded
    assert len(rows) == 180 == len(spectra)
    for key in ("gate", "up", "down"):
        part = base["projections"][key]
        validate_pairs(part["pairs"], 60)
        assert len(part["pairs"]) == len(part["sketches"]) == 30
        assert set(part["sketch_id"]) == set(part["norms"]) == set(range(60))
        assert set(factors["projections"][key]) == set(range(60))
        for i in range(60):
            assert factors["projections"][key][i]["u"].dtype == torch.bfloat16
            assert factors["projections"][key][i]["v"].dtype == torch.bfloat16
    row = next(r for r in rows if r["projection"] == "gate" and r["expert"] == 0)
    part = base["projections"]["gate"]
    bank = bank_for({}, 16, 2., 7)
    packed = part["sketches"][part["sketch_id"][0]]
    dense_base = torch.cat([b for _, b in dense_qjl_base(
        packed, part["norms"][0], bank, 16, 4)])
    weight = model.model.layers[0].mlp.experts[0].gate_proj.weight.detach().float()
    residual = weight - dense_base
    assert torch.allclose(dense_base + residual, weight, atol=1e-6)
    s = torch.linalg.svdvals(residual)
    assert torch.allclose(torch.from_numpy(spectra["gate_0"]), s, atol=1e-5)
    u = factors["projections"]["gate"][0]["u"]
    v = factors["projections"]["gate"][0]["v"]
    metrics = spectrum_metrics(s, residual, weight, dense_base, u, v, ranks=(4,))
    expected_energy = float(s[:4].square().sum() / s.square().sum())
    assert abs(metrics["energy_rank_4"] - expected_energy) < 1e-5
    assert abs(metrics["optimal_residual_error_rank_4"] -
               (1 - expected_energy) ** .5) < 1e-5
    assert metrics["optimal_weight_error_rank_4"] == pytest.approx(
        metrics["optimal_residual_error_rank_4"] *
        metrics["residual_frobenius_ratio"])
    assert row["energy_rank_32"] == pytest.approx(1., abs=1e-5)
    assert row["bf16_factor_residual_error"] >= metrics["optimal_residual_error_rank_4"] - 1e-5


def test_case_a_b_share_base_and_reload(encoded, tmp_path):
    source, base, factors, _, _ = encoded
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(config()))
    torch.save(base, root / "base_layer_0.pt")
    torch.save(factors, root / "rank_layer_0.pt")
    model_a, model_b = copy.deepcopy(source), copy.deepcopy(source)
    router = model_a.model.layers[0].mlp.gate
    shared = model_a.model.layers[0].mlp.shared_expert
    load_checkpoint(model_a, root, "A", "reference")
    load_checkpoint(model_b, root, "B", "reference")
    assert model_a.model.layers[0].mlp.gate is router
    assert model_a.model.layers[0].mlp.shared_expert is shared
    assert all(isinstance(expert, nn.Identity) for expert in model_a.model.layers[0].mlp.experts)
    for i in range(60):
        a = model_a.model.layers[0].mlp._compact_experts[i]
        b = model_b.model.layers[0].mlp._compact_experts[i]
        assert not a.corrections
        assert set(b.corrections) == {"gate", "up", "down"}
        for key in ("gate", "up", "down"):
            assert torch.equal(a.bases[key].packed, b.bases[key].packed)
            assert torch.equal(a.bases[key].norms, b.bases[key].norms)
    first, second = base["projections"]["gate"]["pairs"][0]
    compact = model_b.model.layers[0].mlp._compact_experts
    assert compact[first].bases["gate"].packed.data_ptr() == compact[second].bases["gate"].packed.data_ptr()
    x = torch.randn(3, 16).to(torch.bfloat16)
    reloaded = copy.deepcopy(source)
    load_checkpoint(reloaded, root, "B", "reference")
    assert torch.equal(compact[0](x), reloaded.model.layers[0].mlp._compact_experts[0](x))
    saved_base = torch.load(root / "base_layer_0.pt", weights_only=False)
    assert all("u" not in part and "v" not in part
               for part in saved_base["projections"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton requires CUDA")
def test_reference_triton_and_unique_gpu_storage(encoded):
    source, base, factors, _, _ = encoded
    reference, triton = copy.deepcopy(source).cuda(), copy.deepcopy(source).cuda()
    gpu_base = copy.deepcopy(base)
    gpu_factors = copy.deepcopy(factors)
    install_layer(reference.model.layers[0].mlp, gpu_base, gpu_factors,
                  config(), {}, "reference")
    install_layer(triton.model.layers[0].mlp, gpu_base, gpu_factors,
                  config(), {}, "triton")
    x = torch.randn(3, 16, device="cuda", dtype=torch.bfloat16)
    a = reference.model.layers[0].mlp._compact_experts[0](x)
    b = triton.model.layers[0].mlp._compact_experts[0](x)
    assert torch.allclose(a.float(), b.float(), atol=.15, rtol=.05)
    sizes = resident_bytes(triton)
    expected = sum(t.numel() for part in base["projections"].values()
                   for t in part["sketches"].values())
    assert sizes["gpu_shared_sketch_bytes"] == expected
