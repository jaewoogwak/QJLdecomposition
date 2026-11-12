"""Synthetic correctness checks for Base QJL versus packed Base."""

import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.base_qjl_model import (
    FORMAT, DualQJLExpert, encode_qjl_base, gpu_resident_breakdown,
    load_checkpoint, save_checkpoint,
)
from rescode.qjl.codebook_model import (
    CodebookExpert, load_checkpoint as load_baseline,
    save_checkpoint as save_baseline,
)
from rescode.qjl.core import PackedResidual, ProjectionBank, pack_residual, qjl_reference, qjl_triton
from rescode.qjl.exponent_mantissa_base import (
    PackedExponentMantissaBase, encode_exponent_mantissa_base,
    reconstruct_base_from_weight,
)


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(64, 32, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(32, 64, bias=False, dtype=torch.bfloat16)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        block = nn.Module()
        block.gate = nn.Linear(64, 60, bias=False, dtype=torch.bfloat16)
        block.experts = nn.ModuleList(TinyExpert() for _ in range(60))
        block.top_k = 4
        self.model.layers[0].mlp = block


def test_dual_qjl_synthetic():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(47)
    expert = TinyExpert().to(device)
    residual_states = {}
    base_states = {}
    original_bases = {}
    residuals = {}
    residual_banks = {64: ProjectionBank(128, 0), 32: ProjectionBank(64, 0)}
    for key, input_dim in (("gate", 64), ("up", 64), ("down", 32)):
        w = getattr(expert, key + "_proj").weight.detach()
        state = encode_exponent_mantissa_base(w)
        base = PackedExponentMantissaBase(state, device)
        reconstructed = reconstruct_base_from_weight(w, base.beta)
        assert torch.equal(reconstructed, base.decode())
        residual = w.float() - reconstructed
        assert torch.allclose(reconstructed + residual, w.float(), atol=1e-6)
        r_bank = residual_banks[input_dim]
        packed, norms = pack_residual(residual, r_bank, row_chunk=16)
        assert torch.equal(norms, torch.linalg.vector_norm(residual, dim=1))
        residual_states[key] = {"packed": packed.cpu(), "norms": norms.cpu()}
        base_states[key] = state
        original_bases[key] = base
        residuals[key] = PackedResidual(packed, norms, input_dim, r_bank.d, r_bank,
                                        "triton" if device.type == "cuda" else "reference")
    x = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    a = CodebookExpert(original_bases, residuals)
    y_a = a(x)
    assert y_a.shape == (4, 64)
    baseline_parts = {key: {"base": base_states[key], **residual_states[key]}
                      for key in ("gate", "up", "down")}
    baseline_config = {"base_method": "exponent-mantissa", "qjl_d_factor": 2,
                       "projection_seed": 0, "backend": "triton" if device.type == "cuda" else "reference"}
    with tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "baseline.pt"
        save_baseline({0: {i: baseline_parts for i in range(60)}}, baseline_config, checkpoint)
        restored_a = TinyModel().to(device)
        load_baseline(restored_a, checkpoint, backend=baseline_config["backend"])
        assert torch.equal(restored_a.model.layers[0].mlp._compact_experts[0](x), y_a)
    for factor in (2, 4):
        banks = {64: ProjectionBank(64 * factor, 1),
                 32: ProjectionBank(32 * factor, 1)}
        base_modules, parts = {}, {}
        for key, input_dim in (("gate", 64), ("up", 64), ("down", 32)):
            w = getattr(expert, key + "_proj").weight.detach()
            state = encode_exponent_mantissa_base(w)
            b_state, b = encode_qjl_base(state, banks[input_dim], device, row_chunk=16)
            b.backend = "triton" if device.type == "cuda" else "reference"
            assert b.packed.shape == (w.shape[0], input_dim * factor // 8)
            assert torch.equal(b.norms, torch.linalg.vector_norm(original_bases[key].decode(), dim=1))
            base_modules[key] = b
            parts[key] = {"base": b_state, "residual": residual_states[key]}
            assert torch.equal(parts[key]["residual"]["packed"], residual_states[key]["packed"])
            assert torch.equal(parts[key]["residual"]["norms"], residual_states[key]["norms"])
            if device.type == "cuda":
                projection = banks[input_dim].get(input_dim, device)
                inputs = x if input_dim == 64 else torch.randn(4, 32, device=device, dtype=torch.bfloat16)
                z = inputs.half() @ projection
                ref = qjl_reference(z, b.packed, b.norms, b.d)
                tri = qjl_triton(z, b.packed, b.norms, b.d)
                assert torch.allclose(tri, ref, rtol=0.02, atol=0.02)
        dual = DualQJLExpert(base_modules, residuals)
        y_b = dual(x)
        assert y_b.shape == y_a.shape
        assert torch.isfinite(y_b).all()
        config = {"base_factor": factor, "base_seed": 1, "residual_seed": 0,
                  "input_dims": {"gate": 64, "up": 64, "down": 32}}
        payload = {"format": FORMAT, "config": config, "layers": {0: {0: parts}}}
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / "dual.pt"
            sizes = save_checkpoint(payload, checkpoint)
            assert sizes["checkpoint_bytes"] == checkpoint.stat().st_size
            restored = TinyModel().to(device)
            router = restored.model.layers[0].mlp.gate
            load_checkpoint(restored, checkpoint,
                            "triton" if device.type == "cuda" else "reference")
            assert restored.model.layers[0].mlp.gate is router
            assert restored.model.layers[0].mlp.top_k == 4
            actual = restored.model.layers[0].mlp.experts[0](x)
            assert torch.equal(actual, y_b)
            if device.type == "cuda":
                assert gpu_resident_breakdown(restored)["gpu_resident_compact_total_bytes"] > 0


if __name__ == "__main__":
    test_dual_qjl_synthetic()
    print("Base QJL synthetic encode, Triton/reference, residual parity, and checkpoint reload: passed")
