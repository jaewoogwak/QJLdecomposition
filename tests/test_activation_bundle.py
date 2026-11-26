"""Small, model-free contracts for activation-aware QJL bundling."""

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rescode.qjl.activation_bundle_model import (
    FORMAT, activation_loss, checkpoint_sizes, compress_layer, install_layer,
    load_checkpoint, make_groups, refine_group, regularized_roots,
    svd_factors, unscaled_base, validate_groups, vote,
)
from rescode.qjl.base_bundle_analysis import optimized_scale
from rescode.qjl.codebook_model import bank_for
from rescode.qjl.core import pack_signs, unpack_signs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/QJL"))
import run_activation_bundle as runner
from benchmark_activation_bundle import refine_reference


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(6, 8, bias=False, dtype=torch.bfloat16)


class TinyModel(nn.Module):
    def __init__(self, n_experts=6):
        super().__init__()
        self.model = nn.Module()
        layer = nn.Module()
        block = nn.Module()
        block.gate = nn.Linear(8, n_experts, bias=False, dtype=torch.bfloat16)
        block.shared_expert = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
        block.shared_expert_gate = nn.Linear(8, 1, bias=False, dtype=torch.bfloat16)
        block.experts = nn.ModuleList(TinyExpert() for _ in range(n_experts))
        block.top_k = 2
        block.norm_topk_prob = False
        layer.mlp = block
        self.model.layers = nn.ModuleList([layer])


def config(cases=("A", "B", "C", "D", "E")):
    return {"format": FORMAT, "layers": [0], "qjl_d_factor": 2., "rank": 3,
            "group_size": 2, "pairing": "similarity", "projection_seed": 7,
            "pairing_seed": 11, "bundle_seed": 13, "ridge": 1e-4,
            "svd_method": "exact", "refine_sweeps": 1, "refine_block": 4,
            "conflict_only": True, "row_chunk": 3, "cases": list(cases)}


def calibration(n_experts=6):
    torch.manual_seed(123)
    experts = {}
    for i in range(n_experts):
        experts[i] = {}
        gate = torch.randn(8, 8)
        down = torch.randn(6, 6)
        experts[i]["gate"] = {"covariance": gate.T @ gate / 8 + torch.eye(8), "count": 9}
        experts[i]["up"] = experts[i]["gate"]
        experts[i]["down"] = {"covariance": down.T @ down / 6 + torch.eye(6), "count": 9}
    return {"experts": experts}


def test_groups_and_votes():
    signs = [torch.tensor([[True, False, True, False]]),
             torch.tensor([[True, True, False, False]])]
    assert vote(signs, 5)[0, [0, 3]].tolist() == [True, False]
    assert torch.equal(vote(signs, 5), vote(signs, 5))
    assert not torch.equal(vote(signs, 5, [torch.ones(1, 4), torch.full((1, 4), 3.)]),
                           signs[0])
    packed = {i: pack_signs(torch.rand(4, 8) > .5) for i in range(12)}
    for k in (2, 3, 4, 6):
        groups = make_groups(packed, 8, k, "similarity", 17)
        validate_groups(groups, 12, k)
        assert groups == make_groups(packed, 8, k, "similarity", 17)
        assert len(groups) == 12 // k
        assert len(make_groups(packed, 8, k, "random", 17)) == 12 // k
    with pytest.raises(ValueError):
        validate_groups([[0, 1], [1, 2], [3, 4]], 6, 2)


def test_weighted_vote_and_ties_exact():
    signs = [torch.tensor([[True, False, False]]),
             torch.tensor([[False, True, False]])]
    weights = [torch.tensor([[5., 1., 2.]]), torch.tensor([[1., 5., 2.]])]
    result = vote(signs, 99, weights)
    assert result[0, :2].tolist() == [True, True]
    assert result[0, 2].item() is False
    tied = vote(signs, 99, [torch.ones(1, 3), torch.ones(1, 3)])
    assert torch.equal(tied, vote(signs, 99, [torch.ones(1, 3), torch.ones(1, 3)]))


def test_covariance_loss_matches_direct_outputs_and_scale_is_optimal():
    torch.manual_seed(91)
    x = torch.randn(50, 8)
    c = x.T @ x / len(x)
    w = torch.randn(6, 8)
    b = torch.randn(6, 8)
    a = optimized_scale(w, b, c)
    approximate = a[:, None] * b
    direct = float((x @ (w - approximate).T).square().sum() /
                   (x @ w.T).square().sum())
    assert activation_loss(w, approximate, c) == pytest.approx(direct, abs=1e-5)
    assert activation_loss(w, approximate, c) <= activation_loss(w, b, c) + 1e-5
    assert activation_loss(w, approximate, c) <= activation_loss(w, torch.zeros_like(b), c) + 1e-5


def test_analytical_bit_delta_matches_bruteforce():
    torch.manual_seed(44)
    w = torch.randn(3, 5)
    g = torch.randn(5, 8).half()
    c = torch.randn(5, 5)
    c = c.T @ c + torch.eye(5)
    scale = torch.rand(3)
    h = torch.rand(3, 8) > .5
    old = scale[:, None] * unscaled_base(h, g)
    row, bit = 1, 2
    e = w - old
    alpha = math.sqrt(math.pi / 2) / g.shape[1]
    sign = 1 if h[row, bit] else -1
    expected = (4 * alpha * scale[row] * sign * (e[row] @ c @ g[:, bit].float()) +
                4 * alpha**2 * scale[row]**2 * (g[:, bit].float() @ c @ g[:, bit].float()))
    changed = h.clone()
    changed[row, bit] = ~changed[row, bit]
    new = scale[:, None] * unscaled_base(changed, g)
    brute = ((w - new) @ c * (w - new)).sum() - ((w - old) @ c * (w - old)).sum()
    assert float(expected) == pytest.approx(float(brute), abs=1e-4)


def test_covariance_svd_and_refinement():
    torch.manual_seed(3)
    c = torch.randn(8, 8)
    c = c.T @ c + torch.eye(8)
    root, inverse = regularized_roots(c, 1e-4)
    assert torch.allclose(root @ inverse, torch.eye(8), atol=1e-4)
    residual = torch.randn(6, 8)
    u, v, singular, energy = svd_factors(residual, 3, c)
    assert u.dtype == v.dtype == torch.bfloat16
    expected = torch.linalg.svdvals(residual @ root)
    assert torch.allclose(singular, expected, atol=1e-4)
    assert energy == pytest.approx(float(expected[:3].square().sum() / expected.square().sum()), abs=1e-5)
    random_u, random_v, _, _ = svd_factors(residual, 3, c, method="randomized")
    assert random_u.shape == (6, 3) and random_v.shape == (3, 8)
    zero_u, zero_v, _, zero_energy = svd_factors(residual, 0, c)
    assert zero_u.shape == (6, 0) and zero_v.shape == (0, 8) and zero_energy == 0
    g = torch.randn(8, 16).half()
    weights = [torch.randn(6, 8), torch.randn(6, 8)]
    cov = [c, c]
    scales = [torch.ones(6), torch.ones(6)]
    factors = [(torch.zeros(6, 0), torch.zeros(0, 8))] * 2
    signs = torch.rand(6, 16) > .5
    refined, stats = refine_group(signs, weights, cov, scales, factors, g,
                                  sweeps=2, block_size=4, row_chunk=3,
                                  conflict_only=False)
    assert stats["loss_after"] <= stats["loss_before"] + 1e-6
    assert refined.shape == signs.shape


def test_randomized_svd_is_seeded_and_uses_full_energy():
    torch.manual_seed(52)
    residual = torch.randn(48, 64)
    residual += torch.randn(48, 4) @ torch.randn(4, 64)
    covariance = torch.randn(64, 64)
    covariance = covariance.T @ covariance / 64 + torch.eye(64)
    roots = regularized_roots(covariance, 1e-4)
    exact_u, exact_v, _, exact_energy = svd_factors(residual, 8, covariance,
                                                     roots=roots, method="exact")
    random_a = svd_factors(residual, 8, covariance, roots=roots, method="randomized",
                           oversample=8, niter=2, seed=113)
    random_b = svd_factors(residual, 8, covariance, roots=roots, method="randomized",
                           oversample=8, niter=2, seed=113)
    assert torch.equal(random_a[0], random_b[0])
    assert torch.equal(random_a[1], random_b[1])
    assert 0 < random_a[3] <= exact_energy + 1e-4 <= 1.0001
    assert random_a[2].numel() <= 16 < min(residual.shape)
    target = residual @ roots[0]
    exact_error = ((residual - exact_u.float() @ exact_v.float()) @ roots[0]).norm() / target.norm()
    random_error = ((residual - random_a[0].float() @ random_a[1].float()) @ roots[0]).norm() / target.norm()
    assert random_error <= exact_error + .03


def test_conflict_only_preserves_unanimous_bits():
    torch.manual_seed(99)
    w = torch.randn(4, 5)
    g = torch.randn(5, 8).half()
    signs = torch.rand(4, 8) > .5
    factors = [(torch.zeros(4, 0), torch.zeros(0, 5))] * 2
    refined, stats = refine_group(signs, [w, w], [torch.eye(5)] * 2,
                                  [torch.ones(4)] * 2, factors, g,
                                  block_size=4, row_chunk=2, conflict_only=True)
    assert torch.equal(refined, signs)
    assert stats["accepted_flips"] == 0


def test_optimized_refinement_matches_reference_blocks():
    torch.manual_seed(92)
    rows, cols, d = 9, 8, 16
    g = torch.randn(cols, d).half()
    weights = [torch.randn(rows, cols) for _ in range(2)]
    covariances = []
    for _ in range(2):
        matrix = torch.randn(cols, cols)
        covariances.append(matrix.T @ matrix / cols + torch.eye(cols))
    scales = [torch.rand(rows) for _ in range(2)]
    factors = [(torch.zeros(rows, 0), torch.zeros(0, cols)) for _ in range(2)]
    signs = torch.rand(rows, d) > .5
    for block_size in (4, 16, 64, 128):
        options = {"sweeps": 2, "block_size": block_size,
                   "row_chunk": 0, "conflict_only": False}
        old, old_stats = refine_reference(signs, weights, covariances, scales,
                                          factors, g, **options)
        new, new_stats = refine_group(signs, weights, covariances, scales,
                                      factors, g, **options)
        assert torch.equal(new, old)
        assert new_stats["accepted_flips"] == old_stats["accepted_flips"]
        assert new_stats["loss_after"] <= new_stats["loss_before"] + 1e-6


def test_multi_rank_reuses_group_map_and_base():
    torch.manual_seed(17)
    cfg = config()
    cfg["ranks"] = [0, 2, 3]
    bases, factors, refined, rows, _ = compress_layer(TinyModel().model.layers[0].mlp,
                                                        0, calibration(), cfg, {})
    assert set(refined) == {0, 2, 3}
    for key in ("gate", "up", "down"):
        group_map = bases["A"]["projections"][key]["groups"]
        assert bases["B"]["projections"][key]["groups"] == group_map
        assert all(refined[rank]["projections"][key]["groups"] == group_map
                   for rank in cfg["ranks"])
        assert len(bases["A"]["projections"][key]["sketches"]) == 3
        assert len(bases["B"]["projections"][key]["sketches"]) == 3
        assert factors["C"]["projections"][key][0]["u"].shape[1] == 3
        assert factors["E"][0]["projections"][key][0]["u"].shape[1] == 0
    assert len(rows) == 6 * 3 * (2 + 3 * 3)
    zero_model = TinyModel()
    install_layer(zero_model.model.layers[0].mlp, refined[0], factors["E"][0],
                  cfg, {}, "reference")
    assert torch.isfinite(zero_model.model.layers[0].mlp(torch.randn(2, 8).bfloat16())[0]).all()


def test_all_group_sizes_store_one_sketch_per_group():
    for k in (2, 3, 4, 6):
        torch.manual_seed(27)
        cfg = config(("A", "B"))
        cfg["group_size"] = k
        bases, _, _, _, _ = compress_layer(TinyModel(12).model.layers[0].mlp,
                                           0, calibration(12), cfg, {})
        for case in ("A", "B"):
            for part in bases[case]["projections"].values():
                assert len(part["groups"]) == len(part["sketches"]) == 12 // k


def test_gate_up_covariance_geometry_is_reused(monkeypatch):
    import rescode.qjl.activation_bundle_model as optimized
    original = optimized.regularized_roots
    calls = []
    def counted(covariance, ridge):
        calls.append(covariance.shape)
        return original(covariance, ridge)
    monkeypatch.setattr(optimized, "regularized_roots", counted)
    cfg = config(("A", "B", "D"))
    compress_layer(TinyModel().model.layers[0].mlp, 0, calibration(), cfg, {})
    # Six gate/up covariances are shared; the six Down covariances differ.
    assert len(calls) == 12
    assert calls.count(torch.Size([8, 8])) == 6
    assert calls.count(torch.Size([6, 6])) == 6


@pytest.fixture(scope="module")
def encoded():
    torch.manual_seed(19)
    model = TinyModel()
    base, factors, refined, records, spectra = compress_layer(model.model.layers[0].mlp,
                                                               0, calibration(), config(), {})
    return model, base, factors, refined, records, spectra


def test_compress_storage_and_reload(encoded, tmp_path):
    source, bases, factors, refined, records, spectra = encoded
    assert len(records) == 6 * 3 * 5
    assert len(spectra) == 6 * 3 * 3
    for case in ("A", "B"):
        for key, part in bases[case]["projections"].items():
            validate_groups(part["groups"], 6, 2)
            assert len(part["sketches"]) == 3
            assert len(part["scales"]) == 6
            assert all(t.dtype == torch.float32 for t in part["scales"].values())
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(config()))
    for case in ("A", "B"):
        torch.save(bases[case], root / f"base_{case}_layer_0.pt")
    torch.save(refined[3], root / "base_E_r3_layer_0.pt")
    for case in ("C", "D", "E"):
        torch.save(factors[case][3] if case == "E" else factors[case],
                   root / f"factor_{case}_r3_layer_0.pt")
    for case in ("A", "B", "C", "D", "E"):
        model = copy.deepcopy(source)
        block = model.model.layers[0].mlp
        router, shared = block.gate, block.shared_expert
        load_checkpoint(model, root, case, "reference")
        assert block.gate is router and block.shared_expert is shared
        assert len(block._compact_experts) == 6
        first = block._compact_experts[0].bases["gate"].packed
        group = bases["A" if case == "A" else "B"]["projections"]["gate"]["group_id"][0]
        peer = bases["A" if case == "A" else "B"]["projections"]["gate"]["groups"][group][1]
        assert first.data_ptr() == block._compact_experts[peer].bases["gate"].packed.data_ptr()
        x = torch.randn(2, 8).bfloat16()
        assert torch.isfinite(block(x)[0]).all()
        size = checkpoint_sizes(root, config(), case)
        assert size["shared_sketch_bytes"] > 0
        assert size["effective_bpw"] > 0
    shared = checkpoint_sizes(root, config(), "A")["shared_sketch_bytes"]
    individual = 6 * sum(math.ceil(2 * n / 8) * rows for rows, n in ((6, 8), (6, 8), (8, 6)))
    assert shared == individual // 2


def test_case_b_common_base_and_scale(encoded):
    source, bases, factors, refined, _, _ = encoded
    for key, part in bases["B"]["projections"].items():
        for group_id, members in enumerate(part["groups"]):
            h = unpack_signs(part["sketches"][group_id], part["d"])
            bank = bank_for({}, part["shape"][1], 2., 7)
            g = bank.get(part["shape"][1], torch.device("cpu"))
            b = unscaled_base(h, g)
            for expert_id in members:
                original = getattr(source.model.layers[0].mlp.experts[expert_id], key + "_proj").weight.float()
                approximation = part["scales"][expert_id][:, None] * b
                residual = original - approximation
                assert torch.allclose(approximation + residual, original, atol=1e-6)
    for case in ("C", "D"):
        assert all(value["u"].dtype == torch.bfloat16 for part in
                   factors[case]["projections"].values() for value in part.values())
    assert all(value["u"].dtype == torch.bfloat16 for part in
               factors["E"][3]["projections"].values() for value in part.values())


def test_compress_reload_eval_report_smoke(tmp_path, monkeypatch):
    out = tmp_path / "result"
    cache = out / "cache"
    cache.mkdir(parents=True)
    cal = calibration()
    torch.save(cal, cache / "calibration_layer_0.pt")
    torch.save(cal, cache / "heldout_layer_0.pt")
    calibration_jsonl = tmp_path / "calibration.jsonl"
    heldout_jsonl = tmp_path / "heldout.jsonl"
    calibration_jsonl.write_text('{"text":"calibration"}\n')
    heldout_jsonl.write_text('{"text":"heldout"}\n')
    common = ["--model-path", str(tmp_path / "model"), "--output-dir", str(out),
              "--cache-dir", str(cache),
              "--calibration-jsonl", str(calibration_jsonl),
              "--heldout-jsonl", str(heldout_jsonl),
              "--layers", "0", "--group-size", "2", "--rank", "2",
              "--cases", "A,B,C,D,E", "--backend", "reference"]
    args = runner.arguments(["--mode", "compress", *common])
    runner.atomic_json(cache / "calibration_manifest.json", runner._cache_source(args, [0]))
    runner.compress(args, TinyModel(), [0])
    assert (out / "checkpoint" / "spectrum_layer_0.npz").exists()
    assert (out / "analysis_summary.json").exists()
    assert (out / "plots" / "cumulative_energy.png").exists()
    monkeypatch.setattr(runner, "evaluate_fewshot", lambda model, tokenizer, task, shots,
                        **kwargs: {"results": {}})
    monkeypatch.setattr(runner, "report_eval_summary", lambda *args, **kwargs:
                        {"avg": .5, "avg_norm": .4, "tasks": [("arc_easy", .5, .4)]})
    eval_args = runner.arguments(["--mode", "eval", "--case", "D", *common])
    runner.evaluate(eval_args, TinyModel(), None, [0])
    assert (out / "D" / "rank_2" / "eval_metrics.json").exists()
    assert (out / "summary.csv").exists()
    assert runner.completed_eval_matches(eval_args, "D", 2)
    changed = runner.arguments(["--mode", "eval", "--case", "D", *common,
                                "--task", "arc_challenge"])
    with pytest.raises(ValueError, match="saved evaluation settings differ"):
        runner.completed_eval_matches(changed, "D", 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton parity requires CUDA")
def test_reference_triton_parity(encoded):
    source, bases, factors, refined, _, _ = encoded
    reference = copy.deepcopy(source).cuda()
    accelerated = copy.deepcopy(source).cuda()
    install_layer(reference.model.layers[0].mlp, bases["B"], factors["D"],
                  config(), {}, "reference")
    install_layer(accelerated.model.layers[0].mlp, bases["B"], factors["D"],
                  config(), {}, "triton")
    x = torch.randn(9, 8, device="cuda", dtype=torch.bfloat16)
    expected = reference.model.layers[0].mlp(x)[0]
    actual = accelerated.model.layers[0].mlp(x)[0]
    assert torch.allclose(actual.float(), expected.float(), atol=.08, rtol=.08)
