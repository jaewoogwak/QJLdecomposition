"""Small synthetic checks only; these tests do not load Qwen weights."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.qjl.base_bundle_analysis import (PrefixProjectionBank, StreamingRoutedMoments, base_cases,
    base_metrics, conditioned_weight_cosine, covariance_error, covariance_terms,
    lowrank_covariance_errors, make_expert_cache, optimized_scale,
    perfect_matching, principal_basis, psd_sqrt, randomized_spectrum, spectrum_rows,
    subspace_overlap)
from rescode.qjl.core import ProjectionBank, pack_signs, unpack_signs
from rescode.qjl.residual_bundle_model import bundle_signs


class Expert(nn.Module):
    def __init__(self, n=4, hidden=6):
        super().__init__()
        self.gate_proj = nn.Linear(n, hidden, bias=False)
        self.up_proj = nn.Linear(n, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, experts=4):
        super().__init__()
        self.gate = nn.Linear(4, experts, bias=False)
        self.experts = nn.ModuleList(Expert() for _ in range(experts))
        self.top_k = 2

    def forward(self, hidden):
        logits = self.gate(hidden)
        selected = torch.topk(torch.softmax(logits.float(), -1), self.top_k, -1).indices
        out = torch.zeros_like(hidden)
        mask = F.one_hot(selected, len(self.experts)).permute(2, 1, 0)
        for i, expert in enumerate(self.experts):
            slot, token = torch.where(mask[i])
            if len(token):
                out[token] += expert(hidden[token])
        return out


def fake_model():
    return SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=Block())]))


def test_covariance_subspace_and_scaling():
    torch.manual_seed(4)
    x = torch.randn(20, 4)
    c = x.T @ x / len(x)
    w, b = torch.randn(6, 4), torch.randn(6, 4)
    direct = float(((x @ (w-b).T).norm() / (x @ w.T).norm()))
    assert abs(covariance_error(w, b, c) - direct) < 1e-6
    metrics = base_metrics(w, b, c, c, cal_terms=covariance_terms(w, c),
                           held_terms=covariance_terms(w, c))
    assert abs(metrics["calibration_error"] - direct) < 1e-5
    assert abs(metrics["heldout_error"] - direct) < 1e-5
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    assert abs(subspace_overlap(q[:, :2], q[:, :2]) - 1) < 1e-6
    assert abs(subspace_overlap(q[:, :2], q[:, 2:]) - 0) < 1e-6
    scaled = b * optimized_scale(w, b, c)[:, None]
    assert covariance_error(w, scaled, c) <= covariance_error(w, b, c) + 1e-6
    root = psd_sqrt(c)
    assert abs(conditioned_weight_cosine(w, b, c) -
               float(F.cosine_similarity((w @ root).flatten()[None], (b @ root).flatten()[None]))) < 1e-6


def test_binary_ties_decoder_reuse_identity():
    a = torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]], dtype=torch.bool)
    b = ~a
    pa, pb = pack_signs(a), pack_signs(b)
    assert torch.equal(unpack_signs(pa, 8), a)
    voted1 = bundle_signs(pa, pb, 8, "majority", 123)
    voted2 = bundle_signs(pa, pb, 8, "majority", 123)
    assert torch.equal(voted1, voted2)
    if torch.cuda.is_available():
        voted_gpu = bundle_signs(pa.cuda(), pb.cuda(), 8, "majority", 123)
        assert torch.equal(voted_gpu.cpu(), voted1)
    assert not torch.equal(bundle_signs(pa, pb, 8, "majority", 124), voted1)
    zi = torch.tensor([[5., -1, 5, -1, 5, -1, 5, -1]])
    zj = -torch.tensor([[1., -5, 1, -5, 1, -5, 1, -5]])
    mag = bundle_signs(pa, pb, 8, "magnitude-aware", 1,
                       projections=lambda start, end: (zi[start:end], zj[start:end]))
    assert torch.equal(unpack_signs(mag, 8), torch.ones_like(a))
    if torch.cuda.is_available():
        mag_gpu = bundle_signs(pa.cuda(), pb.cuda(), 8, "magnitude-aware", 1,
                               projections=lambda start, end: (zi[start:end].cuda(), zj[start:end].cuda()))
        assert torch.equal(mag_gpu.cpu(), mag)
    bank = ProjectionBank(8, 0)
    half = PrefixProjectionBank(bank, 4)
    w1, w2 = torch.randn(6, 4), torch.randn(6, 4)
    c = torch.eye(4)
    e1, e2 = make_expert_cache(w1, c, bank, half, [1, 2]), make_expert_cache(w2, c, bank, half, [1, 2])
    cases = base_cases(e1, e2, bank, half, 1, 1e-5)
    if torch.cuda.is_available():
        gpu_bank = ProjectionBank(8, 0)
        gpu_half = PrefixProjectionBank(gpu_bank, 4)
        g1 = make_expert_cache(w1.cuda(), c.cuda(), gpu_bank, gpu_half, [1, 2])
        g2 = make_expert_cache(w2.cuda(), c.cuda(), gpu_bank, gpu_half, [1, 2])
        gpu_cases = base_cases(g1, g2, gpu_bank, gpu_half, 1, 1e-5)
        assert set(gpu_cases) == set(cases)
        for case in cases:
            assert torch.allclose(gpu_cases[case][0].cpu(), cases[case][0], atol=2e-2)
    for case, (bi, bj, _) in cases.items():
        assert float((w1 - (bi + (w1-bi))).abs().max()) < 1e-6
        assert float((w2 - (bj + (w2-bj))).abs().max()) < 1e-6
    # E and F share exactly one decoded majority code before row scaling.
    be, bf = cases["E"][0], cases["F"][0]
    common = be / e1.norm[:, None]
    assert torch.allclose(bf, common * optimized_scale(w1, common, c)[:, None], atol=1e-5)
    assert covariance_error(w1, bf, c) <= covariance_error(w1, be, c) + 1e-6


def test_svd_tail_and_corrections():
    torch.manual_seed(7)
    w, b = torch.randn(8, 5), torch.randn(8, 5) * .1
    r = w-b
    u, s, v = randomized_spectrum(r, 4, 3, exact=True)
    for rank in (0, 1, 3):
        approx = (u[:, :rank] * s[:rank]) @ v[:, :rank].T
        tail = math.sqrt(max(0, 1 - float(s[:rank].square().sum() / r.square().sum())))
        assert abs(float((r-approx).norm()/r.norm()) - tail) < 1e-5
    large = torch.randn(20, 3) @ torch.randn(3, 12) + .001 * torch.randn(20, 12)
    _, exact_s, _ = randomized_spectrum(large, 3, 3, exact=True)
    _, random_s, _ = randomized_spectrum(large, 3, 3, oversample=2, niter=2)
    assert abs(float(random_s[:3].square().sum() / exact_s[:3].square().sum()) - 1) < .01
    c = torch.diag(torch.tensor([3., 2., 1., .5, .1]))
    rows = spectrum_rows(w, b, c, c, [0, 1, 3], 3, method="exact")
    assert {r["space"] for r in rows} == {"weight", "activation"}
    assert all(math.isfinite(r["actual_heldout_error"]) for r in rows)
    assert all(r["actual_heldout_error"] < 2 for r in rows)
    root, invroot = psd_sqrt(c), psd_sqrt(c, regularization=1e-5, inverse=True)
    for space in ("weight", "activation"):
        matrix = r if space == "weight" else r @ root
        uu, ss, vv = randomized_spectrum(matrix, 3, 3, exact=True)
        for rank in (0, 1, 3):
            left = (uu[:, :rank] * ss[:rank]).to(torch.bfloat16).float()
            right = (vv[:, :rank].T @ invroot if space == "activation" else
                     vv[:, :rank].T).to(torch.bfloat16).float()
            expected = covariance_error(w, b + left @ right, c)
            observed = next(x["actual_calibration_error"] for x in rows
                            if x["space"] == space and x["rank"] == rank)
            assert abs(expected - observed) < 1e-5
    left, right = torch.randn(8, 3), torch.randn(3, 5)
    fast = lowrank_covariance_errors(r, w, c, left, right, [0, 1, 3])
    for rank in (0, 1, 3):
        correction = left[:, :rank] @ right[:rank]
        assert abs(fast[rank] - covariance_error(w, b + correction, c)) < 1e-5


def test_hook_snapshots_padding_and_matching():
    torch.manual_seed(0)
    model = fake_model()
    collector = StreamingRoutedMoments(model, [0], probe_limit=2)
    try:
        x = torch.randn(3, 4)
        mask = torch.tensor([[1, 1, 0]])
        collector.begin_batch(mask)
        model.model.layers[0].mlp(x)
        first = collector.snapshot()[0]
        assert int(first["coroute"].diagonal().sum()) == 4
        for expert in first["experts"].values():
            assert expert["gate"] is expert["up"]
        collector.begin_batch(mask)
        model.model.layers[0].mlp(x)
        second = collector.snapshot()[0]
        assert int(second["coroute"].diagonal().sum()) == 8
        for eid, expert in first["experts"].items():
            assert second["experts"][eid]["gate"]["count"] == 2 * expert["gate"]["count"]
    finally:
        collector.close()
    resumed = StreamingRoutedMoments(model, [0], probe_limit=2)
    try:
        resumed.restore({0: first})
        resumed.begin_batch(mask)
        model.model.layers[0].mlp(x)
        replay = resumed.snapshot()[0]
        assert torch.equal(replay["coroute"], second["coroute"])
    finally:
        resumed.close()
    rows = [{"expert_i": i, "expert_j": j, "score": float(i+j)}
            for i in range(4) for j in range(i+1, 4)]
    pairs = perfect_matching(rows, "score", n_experts=4)
    assert sorted(x for pair in pairs for x in pair) == list(range(4))


def test_small_pipeline_outputs(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "QJL"))
    import analyze_base_bundling as script
    torch.manual_seed(8)
    model = fake_model()
    calfile, heldfile = tmp_path / "cal.jsonl", tmp_path / "held.jsonl"
    calfile.write_text('{"text":"one"}\n{"text":"two"}\n')
    heldfile.write_text('{"text":"three"}\n{"text":"four"}\n')
    args = script.arguments(["--stage", "report", "--model-path", str(tmp_path),
                             "--calibration-jsonl", str(calfile), "--heldout-jsonl", str(heldfile),
                             "--output-dir", str(tmp_path / "out"), "--layers", "0",
                             "--projections", "gate,up", "--calibration-sizes", "2",
                             "--heldout-size", "2", "--input-ranks", "1,2",
                             "--residual-ranks", "0,1", "--spectrum-cases", "F,C2",
                             "--matching-rank", "1", "--representative-count", "1"])
    script.prepare(args)
    cache = args.output_dir / "cache"
    cache.mkdir(exist_ok=True)
    def snap(seed):
        torch.manual_seed(seed)
        experts = {}
        for i in range(4):
            x = torch.randn(8, 4)
            experts[i] = {"gate": {"covariance": x.T@x/8 + torch.eye(4)*.1,
                                   "count": 8, "probe": x[:2]}}
            experts[i]["up"] = experts[i]["gate"]
        return {0: {"experts": experts, "coroute": torch.eye(4, dtype=torch.int64)}}
    torch.save(snap(1), cache / "calibration_2.pt")
    torch.save(snap(2), cache / "heldout_2.pt")
    script.similarity(args, model)
    script.base(args, model)
    script.spectrum(args, model)
    script.report(args)
    individual_files = list((args.output_dir / "spectra").glob("*individual*_C2_*.npz"))
    assert len(individual_files) == 4 * 2 * 2  # experts × spaces × projections
    count_before = len(script.read_csv(args.output_dir / "base_metrics.csv"))
    args.resume = True
    script.similarity(args, model)
    script.base(args, model)
    script.spectrum(args, model)
    assert len(script.read_csv(args.output_dir / "base_metrics.csv")) == count_before
    # Report can be regenerated from cached CSV even after source JSONL moves.
    calfile.unlink()
    heldfile.unlink()
    args.calibration_jsonl = None
    args.heldout_jsonl = None
    script.prepare(args)
    script.report(args)
    for name in ("config.json", "pairwise_metrics.csv", "base_metrics.csv",
                 "residual_metrics.csv", "grouping_summary.csv", "correlation.csv",
                 "summary.json", "REPORT.md"):
        assert (args.output_dir / name).is_file(), name
    assert len(script.read_csv(args.output_dir / "pairwise_metrics.csv")) == 12
