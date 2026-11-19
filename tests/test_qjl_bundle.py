"""Small deterministic tests for shared sketches, reconstruction and routing."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.bundled_model import (BundledGroup, checkpoint_sizes,
    compress_model, dense_qjl_base, load_checkpoint, majority_signs,
    reconstruct_residual, save_checkpoint, truncated_svd, validate_groups)
from rescode.qjl.core import ProjectionBank, pack_residual, pack_signs, unpack_signs


class ToyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(7, 11, bias=False)
        self.up_proj = nn.Linear(7, 11, bias=False)
        self.down_proj = nn.Linear(11, 7, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(7, 4, bias=False)
        self.experts = nn.ModuleList(ToyExpert() for _ in range(4))
        self.shared_expert = ToyExpert()
        self.shared_expert_gate = nn.Linear(7, 1, bias=False)
        self.top_k = 2
        self.norm_topk_prob = False

    def forward(self, hidden):
        flat = hidden.reshape(-1, 7)
        logits = self.gate(flat)
        weights, selected = torch.topk(F.softmax(logits, -1, dtype=torch.float), 2, -1)
        output = torch.zeros_like(flat)
        for i in range(4):
            token, slot = torch.where(selected == i)
            if token.numel():
                output.index_add_(0, token, self.experts[i](flat[token]) * weights[token, slot, None])
        output += torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return output.reshape(hidden.shape), logits


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        layer.mlp = ToyBlock()
        self.model.layers = nn.ModuleList([layer])


def config(mode="majority", residual="lowrank"):
    return {"qjl_d": 16, "projection_seed": 9, "bundle_mode": mode,
            "residual_mode": residual, "rank": 4, "svd_seed": 8,
            "svd_oversample": 2, "svd_niter": 1, "backend": "reference"}


class BundleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(18)

    def test_vote_ties_and_bit_order(self):
        first = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=torch.bool)
        other = ~first
        vote = majority_signs({7: pack_signs(other), 2: pack_signs(first)}, 9)
        self.assertTrue(torch.equal(unpack_signs(vote, 9), first))
        self.assertTrue(torch.equal(vote, majority_signs({2: pack_signs(first), 7: pack_signs(other)}, 9)))
        _, pairwise = majority_signs({2: pack_signs(first), 7: pack_signs(other)}, 9, True)
        self.assertEqual(pairwise, 0.0)

    def test_exact_identity_and_lowrank_energy(self):
        w = torch.randn(11, 7)
        bank = ProjectionBank(16, 9)
        packed, norms = pack_residual(w, bank)
        r = reconstruct_residual(w, packed, norms, bank, row_chunk=3)
        base = torch.cat([v for _, v in dense_qjl_base(packed, norms, bank, 7, 3)])
        self.assertTrue(torch.allclose(base + r, w, atol=1e-6))
        u, v, rank = truncated_svd(r, 4, 42, 2, 1)
        u2, v2, _ = truncated_svd(r, 4, 42, 2, 1)
        self.assertTrue(torch.equal(u, u2))
        self.assertTrue(torch.equal(v, v2))
        self.assertEqual(rank, 4)
        self.assertEqual(u.dtype, torch.bfloat16)
        self.assertEqual(v.dtype, torch.bfloat16)
        self.assertLess(torch.linalg.vector_norm(r - u.float() @ v.float()), torch.linalg.vector_norm(r))
        stored_error = float(torch.linalg.vector_norm(r - u.float() @ v.float()) /
                             torch.linalg.vector_norm(r))
        energy = float(1 - (r - u.float() @ v.float()).square().sum() / r.square().sum())
        self.assertAlmostEqual(energy + stored_error ** 2, 1.0, places=5)

    def test_grouping_and_checkpoint_parity(self):
        model = ToyModel()
        fresh = copy.deepcopy(model)
        groups = {0: [[0, 1], [2], [3]]}
        validate_groups(groups[0], 4)
        x = torch.randn(2, 3, 7)
        original_logits = model.model.layers[0].mlp(x)[1]
        data, records, _ = compress_model(model, groups, config())
        stored_group = data[0]["clusters"][0]
        self.assertEqual(len(stored_group["sketches"]["gate"]), 1)
        self.assertNotIn("packed", stored_group["experts"][0]["gate"])
        block = model.model.layers[0].mlp
        self.assertTrue(torch.equal(original_logits, block(x)[1]))
        self.assertIsInstance(block.experts[2], ToyExpert)
        self.assertEqual(len(block._qjl_clusters["0"].sketches["gate"]), 1)
        self.assertEqual(len(records), 6)
        before = block(x)[0]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bundle.pt"
            size = save_checkpoint(data, config(), path)
            self.assertGreater(size["shared_packed_sign_bytes"], 0)
            self.assertEqual(size["checkpoint_bytes"], path.stat().st_size)
            payload = load_checkpoint(fresh, path, backend="reference")
            self.assertEqual(payload["format"], "qjl-majority-lowrank-v1")
            after = fresh.model.layers[0].mlp(x)[0]
            self.assertTrue(torch.equal(before, after))
            self.assertTrue(torch.equal(original_logits, fresh.model.layers[0].mlp(x)[1]))
            self.assertTrue(torch.equal(block.experts[2](x), fresh.model.layers[0].mlp.experts[2](x)))

    def test_none_and_exact(self):
        model = ToyModel()
        data, records, _ = compress_model(model, {0: [[0, 1], [2], [3]]},
                                           config("none", "exact"))
        self.assertEqual(len(model.model.layers[0].mlp._qjl_clusters["0"].sketches["gate"]), 2)
        self.assertLess(max(r["weight_relative_error"] for r in records), 1e-6)
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            save_checkpoint(data, config("none", "exact"), "/tmp/should-not-write.pt")

    def test_none_lowrank_uses_individual_signs_and_reloads(self):
        model = ToyModel()
        original = copy.deepcopy(model)
        fresh = copy.deepcopy(model)
        groups = {0: [[0, 1], [2], [3]]}
        x = torch.randn(3, 7)
        data, records, _ = compress_model(model, groups, config("none"))
        stored = data[0]["clusters"][0]
        self.assertEqual(set(stored["sketches"]["gate"]), {"0", "1"})
        bank = ProjectionBank(config()["qjl_d"], config()["projection_seed"])
        for expert_id in (0, 1):
            for projection in ("gate", "up", "down"):
                weight = getattr(original.model.layers[0].mlp.experts[expert_id],
                                 projection + "_proj").weight.detach()
                packed, norms = pack_residual(weight, bank)
                self.assertTrue(torch.equal(stored["sketches"][projection][str(expert_id)], packed))
                self.assertTrue(torch.equal(stored["experts"][expert_id][projection]["norms"], norms))
        self.assertTrue(torch.equal(model.model.layers[0].mlp(x)[1],
                                    original.model.layers[0].mlp(x)[1]))
        self.assertEqual(len(records), 6)
        before = model.model.layers[0].mlp(x)[0]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "individual.pt"
            save_checkpoint(data, config("none"), path)
            loaded = load_checkpoint(fresh, path, backend="reference")
            self.assertEqual(loaded["layers"][0]["groups"], groups[0])
            self.assertTrue(torch.equal(before, fresh.model.layers[0].mlp(x)[0]))
            self.assertTrue(torch.equal(original.model.layers[0].mlp.experts[2](x),
                                        fresh.model.layers[0].mlp.experts[2](x)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_triton_reference_parity(self):
        from rescode.qjl.bundled_model import qjl_output
        from rescode.qjl.triton_kernel import packed_qjl_fused_gate_up
        w = torch.randn(11, 7, device="cuda")
        bank = ProjectionBank(32, 3)
        packed, norms = pack_residual(w, bank)
        x = torch.randn(4, 7, device="cuda")
        z = x.half() @ bank.get(7, x.device)
        a = qjl_output(z, packed, norms, 32, "reference")
        b = qjl_output(z, packed, norms, 32, "triton")
        self.assertTrue(torch.allclose(a, b, rtol=1e-3, atol=1e-3))
        legacy = qjl_output(z, packed, norms, 32, "triton_legacy")
        self.assertTrue(torch.allclose(a, legacy, rtol=1e-3, atol=1e-3))
        gate, up = packed_qjl_fused_gate_up(z, packed, norms, packed, norms, 32)
        self.assertTrue(torch.allclose(a, gate, rtol=1e-3, atol=1e-3))
        self.assertTrue(torch.allclose(a, up, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
