"""Small exact-SVD checks; full-model results are tested by the analysis run."""

import unittest

import numpy as np
import torch

from rescode.qjl.core import ProjectionBank, pack_residual
from rescode.qjl.bundled_model import dense_qjl_base, majority_signs
from scripts.QJL.analyze_residual_spectrum import spectrum_statistics


class SpectrumStatisticsTests(unittest.TestCase):
    def test_full_energy_and_truncated_svd_bound(self):
        matrix = torch.diag(torch.tensor([4.0, 3.0, 1.0]))
        sigma = torch.linalg.svdvals(matrix).numpy()
        stats, normalized, cumulative = spectrum_statistics(
            sigma, float(torch.linalg.vector_norm(matrix)),
            float(torch.linalg.vector_norm(matrix)))
        self.assertTrue(np.allclose(sigma, [4, 3, 1]))
        self.assertAlmostEqual(float(normalized.sum()), 1.0)
        self.assertTrue(np.all(np.diff(cumulative) >= 0))
        self.assertAlmostEqual(float(cumulative[-1]), 1.0)
        self.assertEqual((stats["r90"], stats["r95"], stats["r99"]), (2, 2, 3))
        self.assertAlmostEqual(stats["energy_r64"], 1.0)
        self.assertAlmostEqual(stats["optimal_weight_error_r64"], 0.0)
        u, s, vh = torch.linalg.svd(matrix)
        for rank in (1, 2, 3):
            reconstructed = (u[:, :rank] * s[:rank]) @ vh[:rank]
            direct = float(torch.linalg.vector_norm(matrix - reconstructed) /
                           torch.linalg.vector_norm(matrix))
            expected = float(np.sqrt(np.square(sigma[rank:]).sum()) /
                             torch.linalg.vector_norm(matrix))
            self.assertAlmostEqual(direct, expected, places=6)

    def test_invalid_spectrum(self):
        with self.assertRaises(ValueError):
            spectrum_statistics(np.array([2.0, np.nan]), 3.0, 3.0)
        with self.assertRaises(AssertionError):
            spectrum_statistics(np.array([1.0, 2.0]), 3.0, 3.0)

    def test_individual_and_majority_residuals_use_same_original(self):
        torch.manual_seed(7)
        original = {0: torch.randn(6, 5).bfloat16(),
                    1: torch.randn(6, 5).bfloat16()}
        unchanged = {key: weight.clone() for key, weight in original.items()}
        bank = ProjectionBank(16, 0)
        packed, norms = {}, {}
        for expert, weight in original.items():
            packed[expert], norms[expert] = pack_residual(weight, bank)
        shared = majority_signs(packed, bank.d)
        for expert, weight in original.items():
            individual_base = torch.cat([base for _, base in dense_qjl_base(
                packed[expert], norms[expert], bank, weight.shape[1])])
            majority_base = torch.cat([base for _, base in dense_qjl_base(
                shared, norms[expert], bank, weight.shape[1])])
            for base in (individual_base, majority_base):
                residual = weight.float() - base
                self.assertTrue(torch.allclose(base + residual, weight.float(), atol=1e-6))
                singular = torch.linalg.svdvals(residual)
                self.assertAlmostEqual(float(singular.square().sum()),
                                       float(residual.square().sum()), places=4)
            self.assertTrue(torch.equal(weight, unchanged[expert]))


if __name__ == "__main__":
    unittest.main()
