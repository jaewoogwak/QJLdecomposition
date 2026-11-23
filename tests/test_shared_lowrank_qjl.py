"""Synthetic checks for the ResCode shared Base / low rank / QJL tail experiment."""

import copy
import csv
import gzip
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from scripts.QJL.run_shared_lowrank_qjl import (
    C4_FILENAME, _summaries, analyze_or_compress, arguments, ensure_c4, main,
    groups_from_descriptors, groups_from_labels,
    install_checkpoint, output_metrics, rank_parts, routed_or_synthetic,
    saliency_average, svd_factors,
)
from rescode.qjl.core import ProjectionBank, pack_residual, qjl_reference, qjl_triton


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
        probs = torch.softmax(logits.float(), dim=-1)
        scores, choices = torch.topk(probs, self.top_k, dim=-1)
        out = torch.zeros_like(flat)
        for expert in choices.unique().tolist():
            token, slot = torch.where(choices == expert)
            value = self.experts[expert](flat[token])
            out.index_add_(0, token, value * scores[token, slot, None].to(value.dtype))
        out += torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return out.reshape(x.shape), logits


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = ToyBlock()


class SharedLowrankQJLTests(unittest.TestCase):
    def test_summary_excludes_singletons_from_norms_and_output_errors(self):
        decomposed = {"projection": "gate", "rank": 0, "input_source": "routed",
                      "singleton": False, "weight_squared_norm": 4.0,
                      "residual_squared_norm": 1.0, "tail_squared_norm": 0.25,
                      "oracle_eta_uv": 0.75, "oracle_eta_e": 0.25,
                      "case_A_relative_l2": 0.1, "case_B_relative_l2": 0.2,
                      "case_C_relative_l2": 0.3}
        singleton = {**decomposed, "singleton": True,
                     "weight_squared_norm": 100.0,
                     "residual_squared_norm": 0.0, "tail_squared_norm": 0.0,
                     "oracle_eta_uv": None, "oracle_eta_e": None,
                     "case_A_relative_l2": 0.0, "case_B_relative_l2": 0.0,
                     "case_C_relative_l2": 0.0}
        summary, = _summaries([decomposed, singleton])
        self.assertEqual(summary["expert_count"], 1)
        self.assertEqual(summary["excluded_singleton_count"], 1)
        self.assertAlmostEqual(summary["rho_r_over_w"], 0.5)
        self.assertAlmostEqual(summary["rho_e_over_w"], 0.25)
        self.assertAlmostEqual(summary["rho_e_over_r"], 0.5)
        self.assertAlmostEqual(summary["case_A_mean_relative_l2"], 0.1)

    def test_c4_download_verify_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compressed = root / "source.gz"
            with gzip.open(compressed, "wb") as handle:
                handle.write(b'{"text":"calibration example"}\n')
            digest = hashlib.sha256(compressed.read_bytes()).hexdigest()
            target = root / "data" / C4_FILENAME
            with patch("huggingface_hub.hf_hub_download", return_value=str(compressed)) as download:
                with patch("scripts.QJL.run_shared_lowrank_qjl.C4_SHA256", digest):
                    self.assertEqual(ensure_c4(target), target.resolve())
                    download.assert_called_once()
            self.assertEqual(json.loads(target.read_text().splitlines()[0])["text"],
                             "calibration example")
            with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("redownloaded")):
                self.assertEqual(ensure_c4(target), target.resolve())
            target.unlink()
            with patch("huggingface_hub.hf_hub_download", return_value=str(compressed)):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    ensure_c4(target)
            self.assertFalse(target.exists())

    def test_saliency_base_and_grouping(self):
        weights = {0: torch.full((2, 2), 1.0), 1: torch.full((2, 2), 3.0)}
        base = saliency_average(weights, torch.tensor([3.0, 1.0]), delta=1e-3)
        self.assertEqual(base.dtype, torch.bfloat16)
        self.assertAlmostEqual(float(base.float().mean()), 1.5, places=2)
        self.assertEqual(groups_from_labels(torch.tensor([7, 7, 2]), [0, 1, 2]),
                         [[0, 1], [2]])
        descriptors = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0]])
        groups = groups_from_descriptors(descriptors, torch.tensor([1.0, 2.0, 9.0]),
                                         [0, 1, 2], 2, 1 / 3)
        self.assertIn([2], groups)
        routed = {"routed_samples": {0: {"gate": {0: torch.ones(3, 4)}}}}
        x, source = routed_or_synthetic(routed, 0, 0, "gate", 4, 2, 7,
                                        torch.device("cpu"))
        self.assertEqual(source, "routed")
        torch.testing.assert_close(x, torch.ones(2, 4))

    def test_oracle_energy_rank_zero_and_stored_tail(self):
        torch.manual_seed(13)
        base = torch.randn(5, 4).to(torch.bfloat16).float()
        w = torch.randn(5, 4).to(torch.bfloat16).float()
        residual = w - base
        left, right = svd_factors(residual, 4)
        for rank in (0, 1, 3, 4):
            factors, tail, stats = rank_parts(residual, left, right, rank)
            self.assertAlmostEqual(stats["oracle_eta_uv"] + stats["oracle_eta_e"],
                                   1.0, places=4)
            restored = factors["u"].float() @ factors["v"].float()
            torch.testing.assert_close(w, base + restored + tail, atol=2e-6, rtol=1e-6)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "factors.pt"
                torch.save(factors, path)
                loaded = torch.load(path, weights_only=True)
                re_tail = residual - loaded["u"].float() @ loaded["v"].float()
                torch.testing.assert_close(tail, re_tail, atol=0, rtol=0)
        self.assertEqual(rank_parts(residual, left, right, 0)[2]["oracle_eta_uv"], 0)

    def _fixture(self, directory):
        model_dir = Path(directory) / "model"
        model_dir.mkdir()
        model = ToyModel()
        tensors = {}
        for expert, module in enumerate(model.model.layers[0].mlp.experts):
            for key in ("gate", "up", "down"):
                name = f"model.layers.0.mlp.experts.{expert}.{key}_proj.weight"
                tensors[name] = getattr(module, key + "_proj").weight.detach().cpu().contiguous()
        save_file(tensors, str(model_dir / "model-00001.safetensors"))
        (model_dir / "config.json").write_text(json.dumps({"num_hidden_layers": 1,
                                                           "num_experts": 3}))
        (model_dir / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "model-00001.safetensors" for name in tensors}}))
        cache = Path(directory) / "grouping.pt"
        torch.save({"group_state_dict": {"model.layers.0.mlp": torch.tensor([0, 0, 1])},
                    "saliency_state_dict": {"model.layers.0.mlp": torch.tensor([2., 1., 3.])}}, cache)
        return model_dir, cache, model

    def test_synthetic_single_layer_compress_reload_and_routing(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir, cache, model = self._fixture(directory)
            out = Path(directory) / "out"
            args = arguments(["--mode", "compress", "--model-path", str(model_dir),
                              "--grouping-cache", str(cache), "--layers", "0",
                              "--max-experts", "3", "--ranks", "0,1,2",
                              "--qjl-d", "16", "--backend", "reference",
                              "--device", "cpu", "--diagnostic-samples", "4",
                              "--output-dir", str(out)])
            result = analyze_or_compress(args)
            self.assertEqual(len(result["checkpoints"]), 9)
            self.assertTrue((out / "rank_vs_output_error.png").is_file())
            self.assertTrue((out / "rank_vs_svd_energy.png").is_file())
            for item in result["summary"]:
                if item["oracle_eta_uv"] is not None:
                    self.assertAlmostEqual(item["oracle_eta_uv"] +
                                           item["oracle_eta_e"], 1.0, places=4)
            with (out / "per_expert.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 27)
            self.assertTrue(all(r["input_source"] == "synthetic" for r in rows))
            self.assertTrue(all(math.isfinite(float(r["case_C_relative_l2"])) for r in rows))
            self.assertTrue(all(r["checkpoint_bytes"] > 0 and
                                r["bf16_compression_ratio"] > 0 for r in result["checkpoints"]))
            component_names = ("shared_base_bytes", "singleton_bf16_bytes",
                               "other_full_bf16_bytes", "lowrank_factor_bytes",
                               "qjl_packed_bytes", "qjl_row_norm_bytes",
                               "metadata_bytes")
            for checkpoint in result["checkpoints"]:
                self.assertEqual(checkpoint["checkpoint_bytes"],
                                 sum(checkpoint[name] for name in component_names))
            rank0_a = torch.load(out / "checkpoints/rank_0/A/layer_0.pt", weights_only=False)
            rank0_c = torch.load(out / "checkpoints/rank_0/C/layer_0.pt", weights_only=False)
            for projection in ("gate", "up", "down"):
                self.assertTrue(torch.equal(
                    rank0_a["experts"][0][projection]["qjl"]["packed"],
                    rank0_c["experts"][0][projection]["qjl"]["packed"]))
            x = torch.randn(5, 4).to(torch.bfloat16)
            for case in ("A", "B", "C"):
                loaded = copy.deepcopy(model)
                block = loaded.model.layers[0].mlp
                router, shared, shared_gate, singleton = (
                    block.gate, block.shared_expert, block.shared_expert_gate,
                    block.experts[2])
                before_logits = block(x)[1]
                manifest = out / f"checkpoints/rank_1/{case}/manifest.json"
                install_checkpoint(loaded, manifest, "reference")
                self.assertIs(block.gate, router)
                self.assertIs(block.shared_expert, shared)
                self.assertIs(block.shared_expert_gate, shared_gate)
                self.assertIs(block.experts[2], singleton)
                after, after_logits = block(x)
                torch.testing.assert_close(before_logits, after_logits, atol=0, rtol=0)
                self.assertEqual(after.shape, x.shape)
                self.assertTrue(torch.isfinite(after.float()).all())

    def test_analyze_writes_no_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir, cache, _ = self._fixture(directory)
            out = Path(directory) / "analyze"
            args = arguments(["--mode", "analyze", "--model-path", str(model_dir),
                              "--grouping-cache", str(cache), "--layers", "0",
                              "--ranks", "0,1", "--qjl-d", "16",
                              "--backend", "reference", "--device", "cpu",
                              "--output-dir", str(out)])
            result = analyze_or_compress(args)
            self.assertFalse((out / "checkpoints").exists())
            self.assertTrue((out / "summary.json").is_file())
            self.assertTrue(all(row["expert_count"] == 2 and
                                row["excluded_singleton_count"] == 1
                                for row in result["summary"]))
            main(["--mode", "summarize", "--output-dir", str(out)])
            self.assertEqual(json.loads((out / "summary.json").read_text())["summary"],
                             result["summary"])

    def test_separate_rescode_group_and_saliency_files(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir, _, _ = self._fixture(directory)
            group = Path(directory) / "group_state_dict.pt"
            saliency = Path(directory) / "saliency_state_dict.pt"
            key = "model.layers.0.mlp"
            torch.save({key: torch.tensor([0, 0, 1])}, group)
            torch.save({key: torch.tensor([2., 1., 3.])}, saliency)
            args = arguments(["--mode", "analyze", "--model-path", str(model_dir),
                              "--grouping-cache", str(group),
                              "--saliency-cache", str(saliency),
                              "--layers", "0", "--ranks", "0",
                              "--qjl-d", "16", "--backend", "reference",
                              "--device", "cpu", "--output-dir", str(Path(directory) / "out")])
            self.assertEqual(len(analyze_or_compress(args)["summary"]), 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/Triton unavailable")
    def test_existing_triton_matches_reference(self):
        device = torch.device("cuda")
        bank = ProjectionBank(64, 7)
        weight = torch.randn(17, 24, device=device)
        x = torch.randn(5, 24, device=device)
        packed, norms = pack_residual(weight, bank)
        z = x.to(bank.get(24, device).dtype) @ bank.get(24, device)
        expected = qjl_reference(z, packed, norms, 64)
        actual = qjl_triton(z, packed, norms, 64)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertIsNotNone(output_metrics(actual, expected)["cosine"])


if __name__ == "__main__":
    unittest.main()
