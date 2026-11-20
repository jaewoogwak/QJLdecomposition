"""CPU checks for rank-constrained grouping and diagnostic outputs."""

import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.QJL.diagnose_shared_rank_base import run
from scripts.QJL.shared_rank_clustering import (
    ResidualCosts, captured_energy, group_metrics, random_groups, truncated_base,
)


class SharedRankTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.weights = torch.randn(5, 4, 3, dtype=torch.float32)

    def test_closed_form_matches_explicit_and_rank_clamps(self):
        for rank in (0, 1, 2, 20):
            costs = ResidualCosts(self.weights, rank)
            group = costs.union(costs.groups[(0,)], costs.groups[(2,)])
            base = truncated_base(group.weight_sum / 2, rank)
            explicit = sum(float((self.weights[i] - base).square().sum()) for i in group.members)
            self.assertAlmostEqual(costs.cost(group), explicit, delta=1e-4)
            expected_capture = float(torch.linalg.svdvals(group.weight_sum / 2)
                                     [:min(rank, 3)].square().sum())
            self.assertAlmostEqual(captured_energy(group.weight_sum / 2, rank),
                                   expected_capture, delta=1e-5)

    def test_greedy_first_merge_is_global_minimum(self):
        costs = ResidualCosts(self.weights, 1)
        pairs = [(i, j) for i in range(5) for j in range(i + 1, 5)]
        expected = min(pairs, key=lambda p: (costs.delta(costs.groups[(p[0],)],
                                                         costs.groups[(p[1],)]), p))
        groups = costs.cluster(4, "rank_aware", candidate_shortlist=0)
        self.assertIn(expected, groups)

    def test_merge_cost_nonnegative_and_deterministic(self):
        costs = ResidualCosts(self.weights, 2)
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertGreaterEqual(costs.delta(costs.groups[(i,)], costs.groups[(j,)]), 0)
        self.assertEqual(costs.cluster(2, "rank_aware", 0),
                         ResidualCosts(self.weights, 2).cluster(2, "rank_aware", 0))
        self.assertEqual(costs.cluster(2, "rank_aware", 3),
                         ResidualCosts(self.weights, 2).cluster(2, "rank_aware", 3))
        self.assertEqual(random_groups(5, 2, 37), random_groups(5, 2, 37))
        self.assertEqual(sorted(map(len, random_groups(5, 2, 37, (1, 4)))), [1, 4])

    def test_zero_rank_singletons_and_identical_experts(self):
        groups = ResidualCosts(self.weights, 0).cluster(5, "rank_aware", 0)
        self.assertEqual(groups, tuple((i,) for i in range(5)))
        summary, _ = group_metrics(self.weights, groups, 0)
        self.assertAlmostEqual(summary["rho_global"], 1.0, places=7)
        same = torch.stack([self.weights[0]] * 3)
        summary, _ = group_metrics(same, ((0, 1, 2),), 20)
        self.assertLess(summary["rho_global"], 1e-5)

    def test_global_ratio_uses_squared_sums(self):
        weights = torch.tensor([[[100.0, 0.0]], [[0.0, 1.0]]])
        summary, items = group_metrics(weights, ((0, 1),), 1)
        expected = math.sqrt(sum(i["residual_squared_norm"] for i in items) /
                             sum(i["weight_squared_norm"] for i in items))
        self.assertAlmostEqual(summary["rho_global"], expected)
        self.assertNotAlmostEqual(summary["rho_global"],
                                  sum(i["residual_ratio"] for i in items) / 2)

    def test_cli_outputs_and_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            model, out = path / "model", path / "out"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "num_experts": 5}))
            tensors = {f"model.layers.0.mlp.experts.{i}.gate_proj.weight":
                       self.weights[i].to(torch.bfloat16).contiguous() for i in range(5)}
            save_file(tensors, str(model / "model-00001.safetensors"))
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "model-00001.safetensors" for name in tensors}}))
            args = Namespace(model_path=model, layers="0", experts="all", projections="gate",
                             group_counts="2,5", ranks="0,2", methods="random,cosine,rank_aware",
                             candidate_shortlist=0, device="cpu", seed=42, qjl_d=8,
                             skip_exponent_reference=True, output_dir=out)
            run(args)
            for name in ("summary.csv", "summary.json", "expert_metrics.csv", "groups.json",
                         "residual_vs_rank_layer0_gate.png"):
                self.assertTrue((out / name).is_file(), name)
            records = json.loads((out / "summary.json").read_text())["records"]
            groups = json.loads((out / "groups.json").read_text())
            self.assertEqual(len(records), 2 * 2 * 4)
            self.assertEqual(len(groups), len(records))
            for item in groups:
                self.assertEqual(len(item["members"]), item["group_count"])
                self.assertEqual(sorted(i for group in item["members"] for i in group),
                                 list(range(5)))
            self.assertTrue(all(r["theoretical_qjl_residual_bits"] > 0 for r in records))


if __name__ == "__main__":
    unittest.main()
