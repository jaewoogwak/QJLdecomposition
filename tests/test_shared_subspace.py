"""Small CPU checks for the independent shared-subspace diagnostic."""

import csv
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file

from scripts.QJL.analyze_shared_subspace import (
    groups_for, individual_metrics, parameter_count, run, shared_metrics,
)


class SharedSubspaceTests(unittest.TestCase):
    def test_cosine_zero_can_share_input_subspace(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        b = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        self.assertEqual(float((a * b).sum()), 0)
        metrics = shared_metrics({0: a, 1: b}, [0, 1, 2], "input")
        self.assertLess(metrics[0][1]["oracle_rho"], 1e-5)
        self.assertLess(metrics[1][1]["oracle_rho"], 1e-5)

    def test_no_shared_direction_has_large_residual(self):
        a = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        b = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        for side in ("input", "output"):
            metrics = shared_metrics({0: a, 1: b}, [1], side)
            group_rho = math.sqrt(sum(metrics[e][1]["oracle_residual_squared_norm"] for e in (0, 1)) / 2)
            self.assertGreater(group_rho, 0.7)

    def test_individual_optimality_and_monotonicity(self):
        torch.manual_seed(31)
        weights = {i: torch.randn(4, 3) for i in range(3)}
        ranks = [0, 1, 2, 3]
        individual = {i: individual_metrics(w, ranks) for i, w in weights.items()}
        for side in ("input", "output"):
            shared = shared_metrics(weights, ranks, side)
            for i in weights:
                for rank in ranks:
                    self.assertLessEqual(individual[i][rank]["oracle_residual_squared_norm"],
                                         shared[i][rank]["oracle_residual_squared_norm"] + 1e-4)
                energies = [shared[i][r]["oracle_eta"] for r in ranks]
                residuals = [shared[i][r]["oracle_rho"] for r in ranks]
                self.assertTrue(all(x <= y + 1e-5 for x, y in zip(energies, energies[1:])))
                self.assertTrue(all(x >= y - 1e-5 for x, y in zip(residuals, residuals[1:])))
                for rank in ranks:
                    metric = shared[i][rank]
                    self.assertAlmostEqual(metric["oracle_eta"] + metric["oracle_rho"] ** 2,
                                           1.0, places=4)

    def test_parameter_counts_and_fp16_reconstruction(self):
        self.assertEqual(parameter_count("shared_input", 4, 3, 5, 2), 5 * 2 + 4 * 3 * 2)
        self.assertEqual(parameter_count("shared_output", 4, 3, 5, 2), 3 * 2 + 4 * 5 * 2)
        self.assertEqual(parameter_count("individual_svd", 4, 3, 5, 2), 4 * 2 * 8)
        weight = torch.tensor([[1.2, -2.4], [0.3, 1.1], [-0.2, 0.7]])
        metrics = individual_metrics(weight, [1])
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        reconstructed = ((u[:, :1] * s[:1]).half().float() @ vh[:1].half().float())
        expected = float((weight - reconstructed).double().square().sum())
        self.assertAlmostEqual(metrics[1]["fp16_residual_squared_norm"], expected, places=8)

    def test_groups_aggregation_and_output_files(self):
        self.assertEqual(groups_for([0, 1, 2, 3, 4], 2, "sequential", 0),
                         [(0, 1), (2, 3), (4,)])
        self.assertEqual(groups_for([0, 1, 2, 3, 4], 2, "random", 17),
                         groups_for([0, 1, 2, 3, 4], 2, "random", 17))
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            out = Path(directory) / "out"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "num_experts": 2}))
            tensors = {}
            for expert in range(2):
                for projection in ("gate", "up", "down"):
                    tensors[f"model.layers.0.mlp.experts.{expert}.{projection}_proj.weight"] = (
                        torch.tensor([[1.0 + expert, -1.25], [0.5, 2.0 + expert]],
                                     dtype=torch.bfloat16).contiguous())
            save_file(tensors, str(model / "model-00001.safetensors"))
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "model-00001.safetensors" for name in tensors}}))
            args = Namespace(model_path=model, layers="0", experts="all",
                             projections="gate,up,down", group_sizes="1,2",
                             ranks="0,1,2,3", groupings="sequential,random",
                             device="cpu", seed=4, target_rho=0.195,
                             output_dir=out)
            result = run(args)
            self.assertEqual(len(result["skipped_ranks"]), 3)
            required = {"per_expert.csv", "per_group.csv", "summary.json", "REPORT.md",
                        "rank_vs_residual.png", "rank_vs_energy.png",
                        "parameter_reduction_vs_residual.png"}
            self.assertEqual({path.name for path in out.iterdir()}, required)
            with (out / "per_expert.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 216)
            with (out / "per_group.csv").open() as handle:
                groups = list(csv.DictReader(handle))
            self.assertEqual(len(groups), 162)
            self.assertEqual(len(result["records"]), 108)
            selected = [g for g in groups if g["projection"] == "gate" and
                        g["grouping"] == "sequential" and g["group_size"] == "1" and
                        g["method"] == "shared_input" and g["rank"] == "1"]
            summary = next(r for r in result["records"] if r["projection"] == "gate" and
                           r["grouping"] == "sequential" and r["group_size"] == 1 and
                           r["method"] == "shared_input" and r["rank"] == 1)
            expected = math.sqrt(sum(float(g["fp16_residual_squared_norm"]) for g in selected) /
                                 sum(float(g["weight_squared_norm"]) for g in selected))
            self.assertAlmostEqual(summary["fp16_rho"], expected)
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
