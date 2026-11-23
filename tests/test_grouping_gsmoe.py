"""Toy checks for GS-MoE gain grouping and shared projection residual SVD."""

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from scripts.QJL.run_grouping_gsmoe import (
    _part_for_rank, _projected_output, affinity_from_gains, analyze_or_compress, arguments,
    balanced_kmedoids, calibration_coefficient, down_projection_gain,
    input_projection_gain, install_checkpoint, reconstruct_shared,
    shared_projection, validate_groups, load_calibration_layer,
)
from scripts.QJL.run_shared_lowrank_qjl import svd_factors


class ToyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.shared_expert = ToyExpert()
        self.shared_expert_gate = nn.Linear(4, 1, bias=False, dtype=torch.bfloat16)
        self.top_k = 2
        self.experts = nn.ModuleList([ToyExpert() for _ in range(3)])

    def forward(self, x):
        flat = x.reshape(-1, 4)
        logits = self.gate(flat)
        scores, choices = torch.topk(torch.softmax(logits.float(), dim=-1),
                                     self.top_k, dim=-1)
        output = torch.zeros_like(flat)
        for expert in choices.unique().tolist():
            token, slot = torch.where(choices == expert)
            value = self.experts[expert](flat[token])
            output.index_add_(0, token, value * scores[token, slot, None].to(value.dtype))
        output += torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return output.reshape(x.shape), logits


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = ToyBlock()


class GroupingGSMoETests(unittest.TestCase):
    def test_reconstruction_gain_matches_direct_gsmoe_formula(self):
        torch.manual_seed(7)
        x = torch.randn(20, 5, dtype=torch.float64)
        w = torch.randn(4, 5, dtype=torch.float64)
        p, _ = torch.linalg.qr(torch.randn(5, 2, dtype=torch.float64))
        cov = x.T @ x / len(x)
        gram = p.T @ cov @ p
        ridge = 1e-8
        gram = gram + ridge * gram.trace() / 2 * torch.eye(2, dtype=torch.float64)
        direct = torch.trace(torch.linalg.solve(
            gram, p.T @ cov @ w.T @ w @ cov @ p))
        actual = input_projection_gain(x, x @ w.T, p, ridge)
        self.assertAlmostEqual(float(actual), float(direct), places=8)
        y = torch.randn(20, 5, dtype=torch.float64)
        direct_down = torch.trace(p.T @ (y.T @ y / len(y)) @ p)
        self.assertAlmostEqual(float(down_projection_gain(y, p)),
                               float(direct_down), places=8)

    def test_affinity_and_fixed_size_grouping(self):
        gains = torch.tensor([[4., 2., 1., 1.], [2., 4., 1., 1.],
                              [1., 1., 5., 3.], [1., 1., 3., 5.]])
        distance, own = affinity_from_gains(gains)
        self.assertTrue(np.allclose(own, [4, 4, 5, 5]))
        self.assertAlmostEqual(distance[0, 1], 0.5)
        groups = balanced_kmedoids(distance, 2, seed=3)
        validate_groups(groups, list(range(4)), 2)
        self.assertEqual(sorted(map(tuple, map(sorted, groups))), [(0, 1), (2, 3)])
        with self.assertRaises(ValueError):
            validate_groups([[0, 1], [1, 2]], list(range(4)), 2)

    def test_shared_coefficient_and_residual_svd(self):
        torch.manual_seed(9)
        x = torch.randn(30, 4)
        cov = x.T @ x / len(x)
        direction, _ = torch.linalg.qr(torch.randn(4, 2))
        w = torch.randn(3, 2) @ direction.T
        p = shared_projection(cov, w.T @ w, 2, "gate", 1e-4)
        coeff = calibration_coefficient(w, p, cov, "gate", 1e-4)
        shared = reconstruct_shared(coeff, p, "gate")
        self.assertLess(float(torch.linalg.norm(w - shared)), 1e-3)
        down = torch.randn(4, 3)
        p_down = shared_projection(cov, down @ down.T, 2, "down", 1e-4)
        coeff_down = calibration_coefficient(down, p_down, cov, "down", 1e-4)
        torch.testing.assert_close(coeff_down, p_down.T @ down)
        residual = torch.randn(4, 3)
        left, right = svd_factors(residual, 3)
        errors = []
        for rank in (0, 1, 2, 3):
            part, stats = _part_for_rank(residual, left, right, rank,
                                         torch.empty(2, 3).to(torch.bfloat16))
            errors.append(stats["oracle_tail_squared_norm"])
            self.assertAlmostEqual(stats["oracle_uv_energy_over_residual"] +
                                   stats["oracle_tail_energy_over_residual"],
                                   1, places=4)
            if rank == 0:
                self.assertNotIn("u", part)
        self.assertTrue(all(a >= b - 1e-5 for a, b in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 1e-9)

    def _fixture(self, directory):
        torch.manual_seed(11)
        root = Path(directory)
        model_dir = root / "model"
        model_dir.mkdir()
        model = ToyModel()
        tensors = {}
        routed = {"train": {0: {key: {} for key in ("gate", "up", "down")}}}
        for i, expert in enumerate(model.model.layers[0].mlp.experts):
            x = torch.randn(10, 4).to(torch.bfloat16)
            middle = F.silu(expert.gate_proj(x)) * expert.up_proj(x)
            q = torch.ones(10)
            for key in ("gate", "up", "down"):
                name = f"model.layers.0.mlp.experts.{i}.{key}_proj.weight"
                tensors[name] = getattr(expert, key + "_proj").weight.detach().cpu().contiguous()
                routed["train"][0][key][i] = (middle if key == "down" else x, q)
        save_file(tensors, str(model_dir / "model-00001.safetensors"))
        (model_dir / "config.json").write_text(json.dumps({
            "num_hidden_layers": 1, "num_experts": 3,
            "hidden_size": 4, "moe_intermediate_size": 3}))
        (model_dir / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {key: "model-00001.safetensors" for key in tensors}}))
        cache = root / "routed.pt"
        torch.save({"routed_samples": routed}, cache)
        return model_dir, cache, model

    def test_toy_analyze_compress_reload_and_router(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir, cache, model = self._fixture(directory)
            out = Path(directory) / "results"
            common = ["--model-path", str(model_dir), "--layers", "0",
                      "--max-experts", "3", "--num-groups", "2",
                      "--probe-rank", "1", "--shared-ranks", "1,2",
                      "--residual-ranks", "0,1,3", "--calibration-cache", str(cache),
                      "--device", "cpu", "--output-dir", str(out)]
            analyzed = analyze_or_compress(arguments(["--mode", "analyze", *common]))
            self.assertEqual(len(analyzed["projection_summary"]), 18)
            self.assertFalse((out / "checkpoints").exists())
            compressed = analyze_or_compress(arguments(["--mode", "compress", *common]))
            self.assertEqual(len(compressed["checkpoints"]), 6)
            with (out / "analysis/per_projection.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3 * 3 * 2 * 3)
            with (out / "analysis/routing_frequency.csv").open() as handle:
                routing_rows = list(csv.DictReader(handle))
            self.assertEqual(len(routing_rows), 3)
            for row in rows:
                self.assertTrue(math.isfinite(float(row["output_relative_l2"])))
                if int(row["residual_rank"]) == 3:
                    self.assertLess(float(row["oracle_tail_rho"]), 1e-5)
                    self.assertLess(float(row["stored_tail_rho"]), 0.01)
            for item in compressed["checkpoints"]:
                self.assertEqual(item["checkpoint_bytes"],
                                 item["shared_projection_bytes"] +
                                 item["expert_coefficient_bytes"] +
                                 item["residual_factor_bytes"] +
                                 item["metadata_bytes"])
            rank0 = torch.load(out / "checkpoints/shared_1/residual_0/layer_0.pt",
                               weights_only=False)
            self.assertEqual(len(rank0["bases"]), 2)
            self.assertTrue(all("u" not in part for expert in rank0["experts"].values()
                                for part in expert.values()))
            x = torch.randn(6, 4).to(torch.bfloat16)
            for rr in (0, 3):
                loaded = copy.deepcopy(model)
                block = loaded.model.layers[0].mlp
                router, shared, shared_gate = block.gate, block.shared_expert, block.shared_expert_gate
                original_logits = block(x)[1]
                manifest = out / f"checkpoints/shared_1/residual_{rr}/manifest.json"
                install_checkpoint(loaded, manifest)
                self.assertIs(block.gate, router)
                self.assertIs(block.shared_expert, shared)
                self.assertIs(block.shared_expert_gate, shared_gate)
                actual, logits = block(x)
                torch.testing.assert_close(logits, original_logits, atol=0, rtol=0)
                self.assertTrue(torch.isfinite(actual.float()).all())
                payload = torch.load(manifest.parent / "layer_0.pt", weights_only=False)
                owner = next(j for j, members in enumerate(payload["groups"]) if 0 in members)
                expected_gate = _projected_output(
                    x, payload["experts"][0]["gate"], payload["bases"][owner]["gate"], "gate")
                actual_gate = block.experts[0].project("gate", x)
                torch.testing.assert_close(actual_gate, expected_gate, atol=0, rtol=0)
                loaded_again = copy.deepcopy(model)
                install_checkpoint(loaded_again, manifest)
                other = loaded_again.model.layers[0].mlp(x)[0]
                torch.testing.assert_close(actual, other, atol=0, rtol=0)
            self.assertTrue((out / "analysis/rank_vs_output_error.png").is_file())

    def test_layerwise_calibration_cache_keeps_uncapped_route_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            _, cache, _ = self._fixture(directory)
            raw = torch.load(cache, weights_only=False)["routed_samples"]["train"][0]
            folder = Path(directory) / "calibration"
            folder.mkdir()
            torch.save({"samples": raw, "frequency": {0: 37, 1: 53, 2: 61},
                        "frequency_source": "full_routing_count"}, folder / "layer_0.pt")
            samples, frequency, source = load_calibration_layer(folder, 0, [0, 1, 2])
            self.assertEqual(source, "full_routing_count")
            self.assertEqual(frequency, {0: 37., 1: 53., 2: 61.})
            self.assertEqual(samples[0]["gate"].shape, (10, 4))
            self.assertEqual(samples[0]["down"].shape, (10, 3))


if __name__ == "__main__":
    unittest.main()
