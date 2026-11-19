"""Exact small-matrix checks for the four exponent/Base QJL cases."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from rescode.qjl.core import ProjectionBank, pack_residual
from rescode.qjl.bundled_model import majority_signs
from rescode.qjl.exponent_mantissa_base import (
    PackedExponentMantissaBase, encode_exponent_mantissa_base)
from scripts.QJL import analyze_exponent_bundle_ablation as experiment


class ExponentBundleAblationTests(unittest.TestCase):
    def test_cache_rejects_changed_inputs_or_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first = {"layers": [12], "max_experts": 4, "groups_sha256": "a"}
            experiment.prepare_output(output, first)
            experiment.prepare_output(output, first)
            with self.assertRaisesRegex(ValueError, "different model, groups, or settings"):
                experiment.prepare_output(output, {**first, "max_experts": 0})
            with self.assertRaisesRegex(ValueError, "different model, groups, or settings"):
                experiment.prepare_output(output, {**first, "groups_sha256": "b"})

    def test_case_a_matches_stored_encoder_and_identity(self):
        weight = torch.tensor([[0.25, -0.75, 1.25, 0.0],
                               [0.03125, -0.09375, 0.125, -0.5]], dtype=torch.bfloat16)
        base, beta = experiment.exponent_base(weight)
        state = encode_exponent_mantissa_base(weight)
        self.assertEqual(beta.dtype, torch.float16)
        self.assertTrue(torch.equal(beta, state["beta"]))
        self.assertTrue(torch.allclose(base, PackedExponentMantissaBase(state).decode(), atol=1e-6))
        record, sigma = experiment.case_statistics(weight, base, "A", 12, 0, 0, "up")
        self.assertLess(record["identity_relative_error"], 1e-6)
        self.assertLess(record["svd_energy_relative_gap"], 1e-3)
        self.assertTrue(np.isfinite(sigma).all())
        self.assertLessEqual(record["r90"], record["r95"])
        self.assertLessEqual(record["r95"], record["r99"])

    def test_full_group_vote_and_deterministic_four_cases(self):
        weights = {}
        for projection_index, projection in enumerate(experiment.PROJECTIONS):
            for expert in range(60):
                generator = torch.Generator().manual_seed(500 + projection_index * 100 + expert)
                weights[expert, projection] = torch.randn(4, 5, generator=generator).bfloat16()

        def fake_weight(_, __, layer, expert, projection):
            self.assertEqual(layer, 12)
            return weights[expert, projection].clone()

        bank = ProjectionBank(16, 0)
        with patch.object(experiment, "weight_tensor", fake_weight):
            first = experiment.analyze_layer(12, {12: [list(range(60))]},
                                             None, None, bank, torch.device("cpu"), 2, 42)
            second = experiment.analyze_layer(12, {12: [list(range(60))]},
                                              None, None, bank, torch.device("cpu"), 2, 42)
        spectra, records, sign_rows, group_rows = first
        self.assertTrue(np.array_equal(spectra, second[0]))
        self.assertEqual(records, second[1])
        self.assertEqual(sign_rows, second[2])
        self.assertEqual(group_rows, second[3])
        self.assertEqual(len(records), 2 * 3 * 4)
        self.assertEqual(len(sign_rows), 2 * 3)
        self.assertTrue(all(row["group_size"] == 60 for row in group_rows))
        self.assertTrue(all(row["encoder_decode_max_abs"] < 1e-6 for row in group_rows))
        for projection in experiment.PROJECTIONS:
            packed = {}
            for expert in range(60):
                base, _ = experiment.exponent_base(weights[expert, projection])
                packed[expert], _ = pack_residual(base, bank)
            majority, pairwise = majority_signs(packed, bank.d, True)
            reported = next(row for row in group_rows if row["projection"] == projection)
            self.assertAlmostEqual(reported["signed_base_pairwise_agreement"], pairwise)
            for expert in (0, 1):
                observed = next(row for row in sign_rows if row["projection"] == projection
                                and row["expert"] == expert)
                self.assertAlmostEqual(observed["signed_base_vs_majority_agreement"],
                    experiment.sign_agreement(packed[expert], majority, bank.d))
        self.assertTrue(all(row["identity_relative_error"] < 1e-6 for row in records))
        self.assertTrue(all(row["svd_energy_relative_gap"] < 1e-3 for row in records))
        self.assertTrue(all(row["r90"] <= row["r95"] <= row["r99"] for row in records))
        self.assertTrue(all(np.isfinite(row["residual_norm_ratio"]) for row in records))
        summary = experiment.summary_statistics(records, sign_rows, group_rows,
                                                {"layers": [12], "qjl_d": 16}, {})
        self.assertEqual(summary["matrix_count"], 24)
        self.assertEqual(summary["paired_comparison"]["all"]["C_to_D"]["count"], 6)


if __name__ == "__main__":
    unittest.main()
