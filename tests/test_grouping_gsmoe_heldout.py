"""Held-out data separation, pooling, and saved-factor output checks."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.QJL.evaluate_grouping_gsmoe_heldout import (
    _record_metrics, _summarize, prepare_disjoint_shard, sha256_file,
)


class HeldoutTests(unittest.TestCase):
    def test_exact_document_overlap_is_removed_without_touching_train(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.json"
            held = root / "held.json"
            train.write_text(''.join(json.dumps({"text": x}) + "\n" for x in ("alpha", "beta")))
            held.write_text(''.join(json.dumps({"text": x}) + "\n" for x in ("beta", "gamma")))
            train_hash = sha256_file(train)
            info = prepare_disjoint_shard(train, held, download=False)
            self.assertEqual(info["exact_text_overlap_documents"], 1)
            self.assertEqual(info["selected_heldout_documents"], 1)
            self.assertEqual(sha256_file(train), train_hash)
            selected = Path(info["selected_heldout_file"])
            self.assertEqual(json.loads(selected.read_text())["text"], "gamma")

    def test_pooled_l2_uses_squared_norm_sums(self):
        rows = [{"split": "train", "layer": 0, "projection": "gate",
                 "shared_rank": 2, "residual_rank": 0,
                 "relative_l2": 0.1, "cosine": 0.9,
                 "output_norm_squared_ratio": 1.0,
                 "error_squared_norm": 1.0, "reference_squared_norm": 100.0},
                {"split": "train", "layer": 0, "projection": "gate",
                 "shared_rank": 2, "residual_rank": 0,
                 "relative_l2": 1.0, "cosine": 0.5,
                 "output_norm_squared_ratio": 1.0,
                 "error_squared_norm": 1.0, "reference_squared_norm": 1.0}]
        summary, = _summarize(rows, projection=True)
        self.assertAlmostEqual(summary["mean_relative_l2"], 0.55)
        self.assertAlmostEqual(summary["pooled_relative_l2"], math.sqrt(2 / 101))

    def test_train_and_heldout_use_unchanged_bf16_factors(self):
        torch.manual_seed(3)
        weights = {key: torch.randn(3 if key != "down" else 4,
                                     4 if key != "down" else 3).to(torch.bfloat16)
                   for key in ("gate", "up", "down")}
        bases = {key: torch.randn(4, 2).to(torch.bfloat16)
                 for key in ("gate", "up", "down")}
        parts = {key: {"coeff": torch.randn(2, 3).to(torch.bfloat16) if key == "down"
                       else torch.randn(3, 2).to(torch.bfloat16),
                       "u": torch.randn(4 if key == "down" else 3, 1).to(torch.bfloat16),
                       "v": torch.randn(1, 3 if key == "down" else 4).to(torch.bfloat16)}
                 for key in ("gate", "up", "down")}
        before = {key: {name: tensor.clone() for name, tensor in part.items()}
                  for key, part in parts.items()}
        x_train = torch.randn(5, 4).to(torch.bfloat16)
        x_held = torch.randn(7, 4).to(torch.bfloat16)
        for split, x in (("train", x_train), ("heldout", x_held)):
            middle = torch.nn.functional.silu(torch.nn.functional.linear(x, weights["gate"])) * \
                     torch.nn.functional.linear(x, weights["up"])
            sample = {"gate": x, "up": x, "down": middle}
            rows, whole = _record_metrics(0, 0, 2, 1, split, sample, weights, parts,
                                          bases, torch.device("cpu"))
            self.assertEqual(len(rows), 3)
            self.assertEqual(whole["samples"], len(x))
            self.assertTrue(all(math.isfinite(row["relative_l2"]) for row in rows))
        for key, part in parts.items():
            for name, tensor in part.items():
                torch.testing.assert_close(tensor, before[key][name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
