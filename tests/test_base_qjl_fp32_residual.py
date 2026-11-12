"""Small exact-residual checks for the Base QJL ablation."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.base_qjl_fp32_model import (
    FORMAT, FP32Residual, QJLBaseFP32Expert, load_checkpoint,
    resident_breakdown, save_checkpoint,
)
from rescode.qjl.base_qjl_model import encode_qjl_base
from rescode.qjl.core import ProjectionBank, qjl_reference, qjl_triton
from rescode.qjl.exponent_mantissa_base import (
    encode_exponent_mantissa_base, PackedExponentMantissaBase,
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


def test_base_qjl_fp32_residual():
    torch.manual_seed(59)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert = TinyExpert().to(device)
    x = torch.randn(3, 64, device=device, dtype=torch.bfloat16)
    original, decoded, residuals = {}, {}, {}
    for key in ("gate", "up", "down"):
        w = getattr(expert, key + "_proj").weight.detach()
        base_state = encode_exponent_mantissa_base(w)
        base = PackedExponentMantissaBase(base_state, device).decode()
        residual = w.float() - base
        assert torch.allclose(base + residual, w.float(), rtol=1e-6, atol=1e-6)
        original[key], decoded[key], residuals[key] = w, base, residual

    for factor in (2, 4):
        banks = {64: ProjectionBank(64 * factor, 1),
                 32: ProjectionBank(32 * factor, 1)}
        bases, dense_modules, parts = {}, {}, {}
        for key in ("gate", "up", "down"):
            w = original[key]
            state = encode_exponent_mantissa_base(w)
            b_state, b_module = encode_qjl_base(state, banks[w.shape[1]], device, 16)
            b_module.backend = "triton" if device.type == "cuda" else "reference"
            dense_modules[key] = FP32Residual(residuals[key].cpu())
            parts[key] = {"base": b_state, "residual": residuals[key].cpu()}
            bases[key] = b_module
            assert b_state["packed"].dtype == torch.uint8
            assert b_state["norms"].dtype == torch.float32
            assert torch.equal(b_state["norms"].to(device),
                               torch.linalg.vector_norm(decoded[key], dim=1))
            x_key = x if key != "down" else torch.randn(3, 32, device=device,
                                                         dtype=torch.bfloat16)
            reference = x_key.float() @ residuals[key].T
            assert torch.allclose(dense_modules[key].forward_fp32(x_key), reference,
                                  rtol=1e-6, atol=1e-6)
            if device.type == "cuda":
                p = banks[w.shape[1]].get(w.shape[1], device)
                z = x_key.half() @ p
                ref = qjl_reference(z, b_module.packed, b_module.norms, b_module.d)
                tri = qjl_triton(z, b_module.packed, b_module.norms, b_module.d)
                assert torch.allclose(tri, ref, rtol=0.02, atol=0.02)
        before = QJLBaseFP32Expert(bases, dense_modules)
        output_before = before(x)
        config = {"base_factor": factor, "base_seed": 1,
                  "input_dims": {"gate": 64, "up": 64, "down": 32},
                  "output_dims": {"gate": 32, "up": 32, "down": 64}}
        payload = {"format": FORMAT, "config": config,
                   "layers": {0: {0: parts}}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "compact_qjl_state.pt"
            sizes = save_checkpoint(payload, path)
            assert sizes["fp32_residual_bytes"] == sum(r.numel() * 4
                                                       for r in residuals.values())
            saved = torch.load(path, map_location="cpu", weights_only=False)
            for key in ("gate", "up", "down"):
                assert torch.equal(saved["layers"][0][0][key]["base"]["packed"],
                                   parts[key]["base"]["packed"])
                assert torch.equal(saved["layers"][0][0][key]["base"]["norms"],
                                   parts[key]["base"]["norms"])
                assert torch.equal(saved["layers"][0][0][key]["residual"],
                                   parts[key]["residual"])
            model = TinyModel().to(device)
            router = model.model.layers[0].mlp.gate
            load_checkpoint(model, path,
                            "triton" if device.type == "cuda" else "reference",
                            residual_device="cpu")
            assert model.model.layers[0].mlp.gate is router
            assert model.model.layers[0].mlp.top_k == 4
            loaded = model.model.layers[0].mlp.experts[0]
            for key, x_key in (("gate", x), ("up", x),
                               ("down", torch.randn(3, 32, device=device,
                                                    dtype=torch.bfloat16))):
                assert torch.equal(loaded.bases[key](x_key), bases[key](x_key))
                assert torch.equal(loaded.residuals[key](x_key), dense_modules[key](x_key))
            assert torch.equal(loaded(x), output_before)
            memory = resident_breakdown(model)
            assert memory["fp32_residual_cpu_bytes"] == sizes["fp32_residual_bytes"]


if __name__ == "__main__":
    test_base_qjl_fp32_residual()
    print("FP32 residual, Base QJL parity, CPU load, and checkpoint forward: passed")
