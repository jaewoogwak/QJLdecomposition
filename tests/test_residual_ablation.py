"""Focused tests for the independent fixed-Base residual A–E study."""

from pathlib import Path
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.merging.lowrank_init import compute_covariance_init
from rescode.qjl.core import ProjectionBank, pack_residual, qjl_reference
from rescode.qjl.exponent_mantissa_base import (
    encode_exponent_mantissa_base, reconstruct_base_from_weight,
)
from rescode.qjl.residual_ablation_methods import (
    PackedINT2Residual, decode_int2, fit_shared_lowrank, gptq_int2,
    lowrank_group_bytes, matched_rank, pack_int2, unpack_int2,
)
from rescode.qjl.residual_ablation_model import (
    FORMAT, build_expert, load_variant, tensor_bytes,
)


def test_int2_pack_and_gptq_hessian_feedback():
    codes = torch.tensor([[0, 1, 2, 3, 1], [3, 2, 1, 0, 2]], dtype=torch.uint8)
    assert torch.equal(unpack_int2(pack_int2(codes), 5), codes)
    torch.manual_seed(17)
    residual = torch.randn(7, 16) * 0.3
    x = torch.randn(64, 16)
    x[:, 1] = x[:, 0] * 0.8 + x[:, 1] * 0.2
    state = gptq_int2(residual, x, block_size=4)
    assert state["packed"].dtype == torch.uint8
    assert state["scale"].dtype == state["offset"].dtype == torch.float16
    assert state["packed"].numel() == 7 * 4
    assert torch.equal(decode_int2(state), PackedINT2Residual(state).decode())
    assert torch.equal(state["packed"], gptq_int2(residual, x, block_size=4)["packed"])
    changed = x.clone()
    changed[:, 1] *= -1
    second = gptq_int2(residual, changed, block_size=4)
    assert not torch.equal(state["packed"], second["packed"]), "GPTQ must depend on Hessian"
    y = PackedINT2Residual(state)(x)
    assert torch.allclose(y, F.linear(x, decode_int2(state)))


def test_rescode_covariance_init_and_budget():
    torch.manual_seed(9)
    residual = torch.randn(3, 12, 16)
    frequency = torch.tensor([2.0, 1.0, 3.0])
    rank, budget = matched_rank(3, 12, 16, 2 * 16)
    assert lowrank_group_bytes(3, 12, 16, rank) <= budget
    assert lowrank_group_bytes(3, 12, 16, rank + 1) > budget
    left, right, codes = fit_shared_lowrank(residual, rank, 0,
                                            freqs=frequency, alpha=2.0)
    a, b, z = compute_covariance_init([r for r in residual], frequency, rank, 2.0)
    assert torch.equal(left, a.to(torch.bfloat16))
    assert torch.equal(right, b.to(torch.bfloat16))
    assert torch.equal(codes, z.to(torch.bfloat16))
    assert left.dtype == right.dtype == codes.dtype == torch.bfloat16


def _tiny_payload(device):
    torch.manual_seed(19)
    shapes = {"gate": (8, 16), "up": (8, 16), "down": (16, 8)}
    parts, originals, residuals = {}, {}, {}
    for key, shape in shapes.items():
        original = (torch.randn(shape, device=device) * 0.1).to(torch.bfloat16)
        base = encode_exponent_mantissa_base(original)
        decoded = reconstruct_base_from_weight(original, base["beta"].to(device))
        residual = original.float() - decoded
        assert torch.allclose(decoded + residual, original.float(), atol=1e-6)
        originals[key], residuals[key] = original, residual
        parts[key] = {"base": base, "residual": residual.cpu()}
    return parts, originals, residuals


def test_base_identity_qjl_and_checkpoint_parity():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts, originals, residuals = _tiny_payload(device)
    x = torch.randn(4, 16, device=device, dtype=torch.bfloat16)
    for key, weight in originals.items():
        base = parts[key]["base"]
        assert torch.equal(base["beta"], encode_exponent_mantissa_base(weight)["beta"])
    case_b = build_expert(parts, {}, "B", 0, 0, device, "reference", device, {})
    for key in ("gate", "up"):
        expected = F.linear(x.float(), originals[key].float())
        actual = case_b.bases[key].decode() @ x.float().T + residuals[key] @ x.float().T
        assert torch.allclose(actual.T, expected, atol=1e-5)
    before = case_b(x)
    payload = {"format": FORMAT, "case": "B", "factor": 0,
               "layer": 0, "config_digest": "test", "experts": {0: parts}, "groups": {}}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "layer_0.pt"
        torch.save(payload, path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        after = build_expert(saved["experts"][0], {}, "B", 0, 0,
                             device, "reference", device, {})(x)
        assert torch.equal(before, after)
        assert tensor_bytes(saved)["residual_bytes"] == sum(r.numel() * 4 for r in residuals.values())
    for key, residual in residuals.items():
        bank = ProjectionBank(residual.shape[1] * 2, 0)
        packed, norms = pack_residual(residual, bank, row_chunk=4)
        assert torch.equal(norms, torch.linalg.vector_norm(residual, dim=1))
        test_x = x if key != "down" else torch.randn(4, 8, device=device, dtype=torch.bfloat16)
        z = test_x.half() @ bank.get(residual.shape[1], device)
        ref = qjl_reference(z, packed, norms, bank.d)
        assert torch.isfinite(ref).all()
        if device.type == "cuda":
            from rescode.qjl.core import qjl_triton
            actual = qjl_triton(z, packed, norms, bank.d)
            assert torch.allclose(actual, ref, atol=0.03, rtol=0.03)


def test_gpu_smoke_residual_methods():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    residual = torch.randn(8, 16, device=device) * 0.1
    samples = torch.randn(32, 16, device=device)
    state = gptq_int2(residual, samples, block_size=4)
    module = PackedINT2Residual(state, device)
    assert torch.isfinite(module(samples)).all()
    triple = torch.stack([residual, residual * 1.1])
    left, right, code = fit_shared_lowrank(triple, 4, 0,
                                           freqs=torch.ones(2, device=device))
    from rescode.qjl.residual_ablation_methods import SharedLowRankResidual
    lowrank = SharedLowRankResidual(left, right, code[0], 2.0)
    assert torch.isfinite(lowrank(samples)).all()


def test_all_cases_checkpoint_forward_and_shared_storage():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts, _, residuals = _tiny_payload(device)
    x = torch.randn(3, 16, device=device, dtype=torch.bfloat16)
    calibrations = {"gate": torch.randn(32, 16, device=device),
                    "up": torch.randn(32, 16, device=device),
                    "down": torch.randn(32, 8, device=device)}
    for case in "ABCDE":
        current, groups = {}, {}
        for key in ("gate", "up", "down"):
            r = residuals[key]
            if case == "A":
                correction = None
            elif case == "B":
                correction = r.cpu()
            elif case == "C":
                bank = ProjectionBank(r.shape[1], 11)
                packed, norms = pack_residual(r, bank)
                correction = {"packed": packed.cpu(), "norms": norms.cpu()}
            elif case == "D":
                left, right, codes = fit_shared_lowrank(
                    torch.stack([r, r * 1.1]), 2, 0,
                    freqs=torch.tensor([1.0, 2.0], device=device))
                groups.setdefault(0, {})[key] = {"a": left.cpu(), "b": right.cpu(),
                                                    "codes": codes.cpu(), "alpha": 2.0}
                correction = {"group_id": 0, "code_index": 0}
            else:
                correction = gptq_int2(r, calibrations[key], block_size=4)
            current[key] = {"base": parts[key]["base"], "residual": correction}
        module = build_expert(current, groups, case, 1, 11, device,
                              "reference", device, {})
        before = module(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "layer_0.pt"
            payload = {"format": FORMAT, "case": case, "factor": 1,
                       "layer": 0, "config_digest": "test",
                       "experts": {0: current}, "groups": groups}
            torch.save(payload, path)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            after = build_expert(saved["experts"][0], saved["groups"], case, 1,
                                 11, device, "reference", device, {})(x)
            assert torch.equal(before, after), case
            if case == "D":
                expected = sum(g["a"].numel() + g["b"].numel() +
                               g["codes"].numel() for g in groups[0].values()) * 2
                assert tensor_bytes(saved)["shared_factor_bytes"] == expected


def test_loader_preserves_routing_and_shared_expert():
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
            self.model.layers = nn.ModuleList([nn.Module()])
            block = nn.Module()
            block.gate = nn.Linear(16, 60, bias=False, dtype=torch.bfloat16)
            block.shared_expert = nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
            block.experts = nn.ModuleList(TinyExpert() for _ in range(60))
            block.top_k = 4
            block.norm_topk_prob = False
            self.model.layers[0].mlp = block

    model = TinyModel()
    block = model.model.layers[0].mlp
    original_router, original_shared = block.gate, block.shared_expert
    source = block.experts[0]
    state = {key: {"base": encode_exponent_mantissa_base(
                getattr(source, key + "_proj").weight.detach()), "residual": None}
             for key in ("gate", "up", "down")}
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "layer_0.pt"
        torch.save({"format": FORMAT, "case": "A", "factor": 0, "layer": 0,
                    "config_digest": "test", "experts": {i: state for i in range(60)},
                    "groups": {}}, path)
        config = {"case": "A", "factor": 0, "qjl_seed": 0,
                  "layers": [0], "digest": "test"}
        load_variant(model, Path(folder), config, backend="reference")
        assert block.gate is original_router
        assert block.shared_expert is original_shared
        assert block.top_k == 4 and len(block._compact_experts) == 60
        x = torch.randn(2, 16, dtype=torch.bfloat16)
        output, router_logits = block(x)
        assert output.shape == (2, 16)
        assert torch.equal(router_logits, original_router(x))
