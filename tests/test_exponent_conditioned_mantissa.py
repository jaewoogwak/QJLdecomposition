"""Deterministic CPU tests for the independent exponent-conditioned ablation."""

import csv
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from safetensors.torch import save_file

from rescode.qjl.exponent_conditioned_mantissa import (
    _fields, _reconstruct, analyze_matrix, fit_exponent_beta,
)
from rescode.qjl.exponent_mantissa_base import (
    encode_exponent_mantissa_base, fit_row_beta, reconstruct_base_from_weight,
)
from scripts.QJL.analyze_exponent_conditioned_mantissa import aggregate, run


class ExponentConditionedTests(unittest.TestCase):
    def test_one_exponent_per_row_matches_baseline(self):
        weight = torch.tensor([[1.0, -1.25, 1.5, -1.75],
                               [-2.0, 2.5, -3.0, 3.5]], dtype=torch.bfloat16)
        records, stats = analyze_matrix(weight)
        self.assertAlmostEqual(records["A"]["rho"], records["B2"]["rho"], places=6)
        self.assertEqual(stats["distinct_exponent_groups"], 2)
        self.assertEqual(stats["unique_exponents_per_row_max"], 1)

    def test_different_mantissas_improve_and_similar_do_not(self):
        different = torch.tensor([[1.0, -1.125, 1.25, 2.75, -3.0, 3.25]],
                                 dtype=torch.bfloat16)
        records, stats = analyze_matrix(different)
        self.assertLess(records["B1"]["residual_squared_norm"],
                        records["A"]["residual_squared_norm"])
        self.assertLess(records["B2"]["residual_squared_norm"],
                        records["A"]["residual_squared_norm"])
        self.assertGreater(stats["within_row_beta_std_mean"], 0)
        similar = torch.tensor([[1.0, -1.25, 2.0, -2.5]], dtype=torch.bfloat16)
        records, _ = analyze_matrix(similar)
        self.assertAlmostEqual(records["A"]["rho"], records["B2"]["rho"], places=6)

    def test_missing_groups_rare_fallback_and_sparse_storage(self):
        weight = torch.tensor([[1.0, 1.25, 1.5, 1.75, 2.5],
                               [4.0, 4.5, 5.0, 0.0, 0.0]], dtype=torch.bfloat16)
        row_beta = fit_row_beta(weight)
        beta, counts, _ = fit_exponent_beta(weight, row_beta, 4)
        self.assertEqual(int((counts > 0).sum()), 3)
        self.assertEqual(float(beta[0, 128]), float(row_beta[0]))
        self.assertEqual(float(beta[1, 129]), float(row_beta[1]))
        self.assertEqual(float(beta[0, 200]), 0)
        records, stats = analyze_matrix(weight, 4)
        self.assertEqual(stats["rare_group_fallback_count"], 2)
        self.assertEqual(stats["fp16_beta_table_bytes"], 2 * 3)
        self.assertEqual(stats["exponent_id_bytes"], 3)
        self.assertEqual(stats["row_offset_bytes"], 4 * (2 + 1))
        self.assertEqual(records["B1"]["base_bytes"], None)

    def test_special_values_and_original_encoder_unchanged(self):
        raw = torch.tensor([[0x0000, 0x8000, 0x0001, 0x8002, 0x3f80,
                             0xbf80, 0x7f80, 0xff80, 0x7fc1]], dtype=torch.uint16)
        weight = raw.view(torch.bfloat16)
        before = encode_exponent_mantissa_base(weight, block_size=32)
        records, stats = analyze_matrix(weight)
        after = encode_exponent_mantissa_base(weight, block_size=32)
        for key in before:
            if isinstance(before[key], torch.Tensor):
                self.assertTrue(torch.equal(before[key], after[key]), key)
            else:
                self.assertEqual(before[key], after[key], key)
        original_base = reconstruct_base_from_weight(weight, before["beta"])
        self.assertEqual(torch.signbit(original_base[0, 1]).item(), True)
        self.assertTrue(torch.isinf(original_base[0, 6]))
        self.assertTrue(torch.isnan(original_base[0, 8]))
        self.assertEqual(stats["positive_zero_count"], 1)
        self.assertEqual(stats["negative_zero_count"], 1)
        self.assertEqual(stats["subnormal_count"], 2)
        self.assertEqual(stats["inf_count"], 2)
        self.assertEqual(stats["nan_count"], 1)
        self.assertTrue(math.isfinite(records["B2"]["rho"]))

    def test_fp16_stored_beta_matches_explicit_rounding_and_oracle_optimum(self):
        weight = torch.tensor([[1.0, 1.015625, -1.25, 2.625, 2.75,
                                -2.875, 0.0]], dtype=torch.bfloat16)
        row = fit_row_beta(weight)
        beta, _, fields = fit_exponent_beta(weight, row)
        state = encode_exponent_mantissa_base(weight)
        special = reconstruct_base_from_weight(weight, state["beta"])
        rounded = _reconstruct(weight, beta.half().float(), fields, special)
        stored_sq = float((weight.float() - rounded).double().square().sum())
        records, _ = analyze_matrix(weight)
        self.assertAlmostEqual(records["B2"]["residual_squared_norm"], stored_sq, places=10)
        self.assertLessEqual(records["B1"]["residual_squared_norm"],
                             records["B2"]["residual_squared_norm"] + 1e-8)

    def test_fp32_oracle_beats_fp32_row_beta(self):
        weight = torch.tensor([[1.0, -1.25, 1.5, 2.625, -2.875, 3.25],
                               [0.5, -0.75, 1.125, -2.25, 2.75, 3.0]],
                              dtype=torch.bfloat16)
        row = fit_row_beta(weight)
        fields = _fields(weight)
        stored = encode_exponent_mantissa_base(weight)["beta"]
        special = reconstruct_base_from_weight(weight, stored)
        row_base = _reconstruct(weight, row[:, None].expand(-1, 256), fields, special)
        row_cost = float((weight.float() - row_base).double().square().sum())
        beta, _, _ = fit_exponent_beta(weight, row)
        oracle_base = _reconstruct(weight, beta, fields, special)
        oracle_cost = float((weight.float() - oracle_base).double().square().sum())
        self.assertLessEqual(oracle_cost, row_cost + 1e-10)

    def test_global_aggregate_uses_squared_norms(self):
        big = torch.tensor([[100.0, 125.0, 150.0]], dtype=torch.bfloat16)
        small = torch.tensor([[1.0, 1.25, 1.5]], dtype=torch.bfloat16)
        records = []
        for weight in (big, small):
            methods, _ = analyze_matrix(weight)
            records.extend(methods.values())
        overall = {r["method"]: r for r in aggregate(records)}
        for method in ("A", "B1", "B2"):
            selected = [r for r in records if r["method"] == method]
            expected = math.sqrt(sum(r["residual_squared_norm"] for r in selected) /
                                 sum(r["weight_squared_norm"] for r in selected))
            self.assertAlmostEqual(overall[method]["rho_global"], expected)

    def test_synthetic_cli_outputs_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            out = Path(directory) / "results"
            model.mkdir()
            (model / "config.json").write_text(json.dumps({"num_hidden_layers": 1, "num_experts": 2}))
            tensors = {}
            for expert in range(2):
                for projection in ("gate", "up", "down"):
                    tensors[f"model.layers.0.mlp.experts.{expert}.{projection}_proj.weight"] = (
                        torch.tensor([[1.0 + expert, -1.25 - expert, 2.5 + expert]],
                                     dtype=torch.bfloat16).contiguous())
            save_file(tensors, str(model / "model-00001.safetensors"))
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {name: "model-00001.safetensors" for name in tensors}}))
            args = Namespace(model_path=model, layers="0", experts="all",
                             projections="gate,up,down", device="cpu",
                             min_exponent_count=1, output_dir=out)
            run(args)
            for name in ("config.json", "per_matrix.csv", "per_projection.csv",
                         "per_layer.csv", "summary.json", "exponent_statistics.json",
                         "REPORT.md"):
                self.assertTrue((out / name).is_file(), name)
            with (out / "per_matrix.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 18)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(len(summary["overall"]), 3)
            self.assertEqual(len(summary["by_projection"]), 9)
            report = (out / "REPORT.md").read_text()
            self.assertIn("B2 lowers rho on", report)
            self.assertIn("observed FP16 rounding penalty", report)
            self.assertIn("Whether it improves QJL accuracy or runtime", report)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_cpu_cuda_parity_if_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        weight = torch.tensor([[1.0, -1.25, 2.75, 3.0, 0.0]], dtype=torch.bfloat16)
        cpu, _ = analyze_matrix(weight)
        gpu, _ = analyze_matrix(weight.cuda())
        for method in ("A", "B1", "B2"):
            self.assertAlmostEqual(cpu[method]["rho"], gpu[method]["rho"], places=6)


if __name__ == "__main__":
    unittest.main()
