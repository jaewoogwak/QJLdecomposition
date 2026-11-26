"""Synthetic Qwen2-MoE calibration parity; no pretrained weights are loaded."""

from types import SimpleNamespace
import json
import sys
from pathlib import Path

import pytest

import torch
from transformers.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeModel

from rescode.qjl.activation_calibration import collect_layerwise, token_batches
from rescode.qjl.base_bundle_analysis import StreamingRoutedMoments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/QJL"))
import run_activation_bundle as runner


def tiny_backbone():
    torch.manual_seed(203)
    config = Qwen2MoeConfig(vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12,
        shared_expert_intermediate_size=24, max_position_embeddings=64,
        use_cache=False)
    return Qwen2MoeModel(config).eval()


def batches():
    return [
        {"input_ids": torch.tensor([[3, 7, 11, 5, 0], [2, 8, 9, 0, 0]]),
         "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])},
        {"input_ids": torch.tensor([[5, 13, 19, 23, 31]]),
         "attention_mask": torch.ones(1, 5, dtype=torch.long)},
    ]


def test_layerwise_matches_reference_covariance_and_layer_outputs(tmp_path):
    model = tiny_backbone()
    split_batches = batches()
    expected_outputs = {0: [], 1: []}
    hooks = [layer.register_forward_hook(
        lambda _module, _inputs, output, layer_id=i:
        expected_outputs[layer_id].append(output[0].detach().clone()))
        for i, layer in enumerate(model.layers)]
    reference = StreamingRoutedMoments(SimpleNamespace(model=SimpleNamespace(layers=model.layers)),
                                       [0, 1], probe_limit=0)
    try:
        with torch.no_grad():
            for batch in split_batches:
                reference.begin_batch(batch["attention_mask"])
                model(**batch, use_cache=False)
        expected = reference.snapshot()
    finally:
        reference.close()
        for hook in hooks:
            hook.remove()
    actual_outputs = {0: [], 1: []}
    hooks = [layer.register_forward_hook(
        lambda _module, _inputs, output, layer_id=i:
        actual_outputs[layer_id].append(output[0].detach().clone()))
        for i, layer in enumerate(model.layers)]
    try:
        collect_layerwise(model, {"calibration": split_batches}, [0, 1], tmp_path,
                          hidden_cache="cpu", progress=lambda *a, **k: None)
    finally:
        for hook in hooks:
            hook.remove()
    for layer in (0, 1):
        for original, optimized in zip(expected_outputs[layer], actual_outputs[layer]):
            assert torch.allclose(original, optimized, atol=1e-5, rtol=1e-5)
        actual = torch.load(tmp_path / f"calibration_layer_{layer}.pt",
                            map_location="cpu", weights_only=False)
        for expert in range(4):
            for projection in ("gate", "down"):
                expected_item = expected[layer]["experts"][expert][projection]
                actual_item = actual["experts"][expert][projection]
                assert actual_item["count"] == expected_item["count"]
                assert torch.allclose(actual_item["covariance"],
                                      expected_item["covariance"], atol=2e-5, rtol=2e-5)
            assert actual["experts"][expert]["up"] is actual["experts"][expert]["gate"]


def test_disk_hidden_resume_skips_completed_layer(tmp_path):
    model = tiny_backbone()
    split = {"calibration": batches()}
    collect_layerwise(model, split, [0], tmp_path, hidden_cache="disk",
                      progress=lambda *a, **k: None)
    assert (tmp_path / "hidden/calibration/layer_0/COMPLETE").exists()
    seen = {0: 0, 1: 0}
    hooks = [layer.register_forward_hook(
        lambda _module, _inputs, _output, layer_id=i:
        seen.__setitem__(layer_id, seen[layer_id] + 1))
        for i, layer in enumerate(model.layers)]
    try:
        profiles, mode = collect_layerwise(model, split, [0, 1], tmp_path,
            hidden_cache="disk", resume=True, progress=lambda *a, **k: None)
    finally:
        for hook in hooks:
            hook.remove()
    assert mode == "disk"
    assert seen == {0: 0, 1: len(split["calibration"])}
    assert (tmp_path / "calibration_layer_0.pt").exists()
    assert (tmp_path / "calibration_layer_1.pt").exists()
    assert not (tmp_path / "hidden/calibration/layer_0").exists()


def test_cache_manifest_reuse_and_invalidation(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"revision": 1}')
    data = tmp_path / "data"
    data.mkdir()
    cal, held = data / "cal.jsonl", data / "held.jsonl"
    cal.write_text('{"text":"first"}\n')
    held.write_text('{"text":"second"}\n')
    cache = tmp_path / "cache"
    cache.mkdir()
    args = runner.arguments(["--mode", "report", "--model-path", str(model_dir),
        "--output-dir", str(tmp_path / "out"), "--cache-dir", str(cache),
        "--calibration-jsonl", str(cal), "--heldout-jsonl", str(held),
        "--layers", "0", "--calibration-size", "1", "--heldout-size", "1"])
    runner.atomic_json(cache / "calibration_manifest.json", runner._cache_source(args, [0]))
    torch.save({"experts": {}}, cache / "calibration_layer_0.pt")
    assert runner.cache_complete(args, [0]) is False
    torch.save({"experts": {}}, cache / "heldout_layer_0.pt")
    assert runner.cache_complete(args, [0]) is True
    args_without_paths = runner.arguments(["--mode", "report", "--model-path", str(model_dir),
        "--output-dir", str(tmp_path / "out"), "--cache-dir", str(cache),
        "--layers", "0", "--calibration-size", "1", "--heldout-size", "1"])
    assert runner.cache_complete(args_without_paths, [0]) is True
    legacy = runner._cache_source(args, [0])
    legacy.pop("format")
    legacy.pop("model_identity")
    runner.atomic_json(cache / "calibration_manifest.json", legacy)
    assert runner.cache_complete(args, [0]) is True
    runner.atomic_json(cache / "calibration_manifest.json", runner._cache_source(args, [0]))
    held.write_text('{"text":"changed"}\n')
    with pytest.raises(ValueError, match="dataset differs"):
        runner.cache_complete(args, [0])
    with pytest.raises(ValueError, match="dataset differs"):
        runner.cache_complete(args_without_paths, [0])
    held.write_text('{"text":"second"}\n')
    (model_dir / "config.json").write_text('{"revision": 2}')
    with pytest.raises(ValueError, match="model revision"):
        runner.cache_complete(args, [0])
    (model_dir / "config.json").write_text('{"revision": 1}')
    altered = runner.arguments(["--mode", "report", "--model-path", str(model_dir),
        "--output-dir", str(tmp_path / "out"), "--cache-dir", str(cache),
        "--calibration-jsonl", str(cal), "--heldout-jsonl", str(held),
        "--layers", "0", "--calibration-size", "1", "--heldout-size", "1",
        "--calibration-batch-size", "4"])
    with pytest.raises(ValueError, match="calibration_batch_size"):
        runner.cache_complete(altered, [0])


def test_length_bucketing_keeps_samples_and_reduces_padding():
    class Tokenizer:
        def __call__(self, texts, **_kwargs):
            ids = [list(range(1, len(text.split()) + 1)) for text in texts]
            return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}

        def pad(self, records, **_kwargs):
            length = max(len(item["input_ids"]) for item in records)
            return {"input_ids": torch.tensor([item["input_ids"] +
                     [0] * (length - len(item["input_ids"])) for item in records]),
                    "attention_mask": torch.tensor([item["attention_mask"] +
                     [0] * (length - len(item["attention_mask"])) for item in records])}

    texts = ["a", "a b c d e f g h", "a b", "a b c d e f g", "a b c", "a b c d e f"]
    plain = token_batches(Tokenizer(), texts, batch_size=2, bucket=False)
    bucketed = token_batches(Tokenizer(), texts, batch_size=2, bucket=True)
    assert sorted(i for batch in bucketed for i in batch["sample_indices"]) == list(range(6))
    assert sum(batch["input_ids"].numel() for batch in bucketed) < sum(
        batch["input_ids"].numel() for batch in plain)
