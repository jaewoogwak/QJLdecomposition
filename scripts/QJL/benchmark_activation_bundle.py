#!/usr/bin/env python3
"""Tiny synthetic benchmark of calibration, SVD, and refinement backends."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers.models.qwen2_moe import Qwen2MoeConfig, Qwen2MoeModel

from rescode.qjl.activation_calibration import collect_layerwise
from rescode.qjl.activation_bundle_model import (
    SQRT_PI_HALF, activation_loss, refine_group, refinement_geometry,
    regularized_roots, svd_factors, unscaled_base,
)
from rescode.qjl.base_bundle_analysis import StreamingRoutedMoments


def refine_reference(signs, weights, covariances, scales, factors, g, *,
                     sweeps=1, block_size=32, row_chunk=64, conflict_only=True,
                     geometry=None):
    """The pre-optimization block loop, retained only for comparison."""
    if sweeps == 0:
        return signs, {"accepted_flips": 0, "candidate_flips": 0}
    h = signs.clone()
    rows, d = h.shape
    row_chunk = row_chunk or rows
    a = SQRT_PI_HALF / d
    geometry = geometry if geometry is not None else refinement_geometry(weights, covariances, g)
    cg, kg = geometry["cg"], geometry["gram"]
    denominator, individual = geometry["denominator"], geometry["individual"]
    target = [w.float() - u.float() @ v.float() for w, (u, v) in zip(weights, factors)]
    candidates = accepted = 0
    for _ in range(sweeps):
        for start in range(0, rows, row_chunk):
            end = min(start + row_chunk, rows)
            old_sign = h[start:end].float().mul(2).sub(1)
            current_b = a * old_sign @ g.float().T
            ecg = [(t[start:end] - s[start:end, None] * current_b) @ x
                   for t, s, x in zip(target, scales, cg)]
            for lo in range(0, d, block_size):
                hi = min(lo + block_size, d)
                dh = -2 * old_sign[:, lo:hi]
                delta = torch.zeros_like(dh)
                for s, ec, gram, den in zip(scales, ecg, kg, denominator):
                    amplitude = a * s[start:end, None] * dh
                    delta += (-2 * amplitude * ec[:, lo:hi] +
                              amplitude.square() * gram.diag()[lo:hi]) / den
                mask = delta < 0
                if conflict_only:
                    conflict = (individual[:, start:end, lo:hi] !=
                                individual[0:1, start:end, lo:hi]).any(0)
                    mask &= conflict
                candidates += int(mask.sum())
                if not mask.any():
                    continue
                proposal = dh * mask
                total = torch.zeros(end - start, device=signs.device)
                for s, ec, gram, den in zip(scales, ecg, kg, denominator):
                    amp = a * s[start:end, None] * proposal
                    total += (-2 * (amp * ec[:, lo:hi]).sum(1) +
                              (amp @ gram[lo:hi, lo:hi] * amp).sum(1)) / den
                take = (total < -1e-10) & mask.any(1)
                proposal *= take[:, None]
                accepted += int((proposal != 0).sum())
                old_sign[:, lo:hi] += proposal
                h[start:end, lo:hi] = old_sign[:, lo:hi] > 0
                for e, s in enumerate(scales):
                    ecg[e] -= (a * s[start:end, None] * proposal) @ kg[e][lo:hi, :]
    return h, {"accepted_flips": accepted, "candidate_flips": candidates}


def timed(function, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def tiny_model(device):
    config = Qwen2MoeConfig(vocab_size=256, hidden_size=48, intermediate_size=96,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        num_experts=6, num_experts_per_tok=2, moe_intermediate_size=64,
        shared_expert_intermediate_size=96, max_position_embeddings=128,
        use_cache=False)
    return Qwen2MoeModel(config).eval().to(device)


def benchmark_calibration(device):
    torch.manual_seed(314)
    model = tiny_model(device)
    batches = []
    for _ in range(3):
        ids = torch.randint(0, 256, (4, 48))
        mask = torch.ones(4, 48, dtype=torch.long)
        mask[0, -8:] = 0
        mask[1, -16:] = 0
        batches.append({"input_ids": ids, "attention_mask": mask})
    reference = StreamingRoutedMoments(SimpleNamespace(model=SimpleNamespace(layers=model.layers)),
                                       [0, 1], probe_limit=0, profile=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        def reference_run():
            with torch.no_grad():
                for batch in batches:
                    reference.begin_batch(batch["attention_mask"])
                    model(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device), use_cache=False)
            return reference.snapshot()
        ref_state, ref_seconds = timed(reference_run, device)
    finally:
        reference.close()
    ref_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory(prefix="qjl-calibration-bench-") as temporary:
        def optimized_run():
            collect_layerwise(model, {"calibration": batches}, [0, 1], temporary,
                hidden_cache="gpu" if device.type == "cuda" else "cpu",
                profile=True, progress=lambda *args, **kwargs: None)
            return [torch.load(Path(temporary) / f"calibration_layer_{layer}.pt",
                               map_location="cpu", weights_only=False) for layer in (0, 1)]
        new_state, new_seconds = timed(optimized_run, device)
    new_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    max_diff = 0.
    equal_counts = True
    for layer in (0, 1):
        for expert in range(6):
            for projection in ("gate", "down"):
                old = ref_state[layer]["experts"][expert][projection]
                new = new_state[layer]["experts"][expert][projection]
                equal_counts &= old["count"] == new["count"]
                max_diff = max(max_diff, float((old["covariance"] -
                                               new["covariance"]).abs().max()))
    return {"reference_seconds": ref_seconds, "layerwise_seconds": new_seconds,
            "reference_peak_gpu_bytes": int(ref_peak),
            "layerwise_peak_gpu_bytes": int(new_peak),
            "reference_profile": reference.profile,
            "equal_routed_counts": bool(equal_counts), "max_covariance_abs_diff": max_diff}


def benchmark_svd(device):
    torch.manual_seed(128)
    rows, cols, rank = 192, 256, 48
    residual = (torch.randn(rows, 24, device=device) @
                torch.randn(24, cols, device=device) + .05 * torch.randn(rows, cols, device=device))
    matrix = torch.randn(cols, cols, device=device)
    covariance = matrix.T @ matrix / cols + .5 * torch.eye(cols, device=device)
    roots, root_seconds = timed(lambda: regularized_roots(covariance, 1e-4), device)
    results = {}
    for method in ("exact", "randomized"):
        (u, v, singular, energy), seconds = timed(
            lambda: svd_factors(residual, rank, covariance, method=method,
                                roots=roots, oversample=16, niter=2, seed=9), device)
        approximation = u.float() @ v.float()
        results[method] = {"seconds": seconds, "singular_values_computed": singular.numel(),
                           "energy_retention": energy,
                           "weight_relative_error": float((residual - approximation).norm() /
                                                          residual.norm()),
                           "activation_relative_error": math.sqrt(activation_loss(
                               residual, approximation, covariance))}
    return {"covariance_roots_seconds": root_seconds, "rank": rank,
            "matrix_shape": [rows, cols], "methods": results}


def benchmark_refinement(device):
    torch.manual_seed(77)
    rows, cols, d = 48, 48, 96
    g = torch.randn(cols, d, device=device).half()
    weights = [torch.randn(rows, cols, device=device) for _ in range(2)]
    covariances = []
    for _ in range(2):
        matrix = torch.randn(cols, cols, device=device)
        covariances.append(matrix.T @ matrix / cols + torch.eye(cols, device=device))
    scales = [torch.ones(rows, device=device) for _ in range(2)]
    signs = torch.rand(rows, d, device=device) > .5
    factors = [(torch.zeros(rows, 0, device=device),
                torch.zeros(0, cols, device=device)) for _ in range(2)]
    geometry = refinement_geometry(weights, covariances, g)
    options = dict(sweeps=1, block_size=32, row_chunk=48, conflict_only=True,
                   geometry=geometry)
    (old_signs, old_stats), old_seconds = timed(lambda: refine_reference(
        signs, weights, covariances, scales, factors, g, **options), device)
    (new_signs, new_stats), new_seconds = timed(lambda: refine_group(
        signs, weights, covariances, scales, factors, g, **options), device)
    old_error = sum(activation_loss(w, s[:, None] * unscaled_base(old_signs, g), c)
                    for w, s, c in zip(weights, scales, covariances))
    new_error = sum(activation_loss(w, s[:, None] * unscaled_base(new_signs, g), c)
                    for w, s, c in zip(weights, scales, covariances))
    return {"reference_seconds": old_seconds, "optimized_seconds": new_seconds,
            "bit_agreement": float((old_signs == new_signs).float().mean()),
            "reference_accepted_flips": old_stats["accepted_flips"],
            "optimized_accepted_flips": new_stats["accepted_flips"],
            "reference_final_loss": old_error, "optimized_final_loss": new_error,
            "optimized_accepted_flip_ratio": new_stats["accepted_flips"] / (rows * d)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/activation_bundle/benchmark"))
    args = parser.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    result = {"device": str(device), "calibration": benchmark_calibration(device),
              "svd": benchmark_svd(device), "refinement": benchmark_refinement(device),
              "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(json.dumps(result, indent=2))
    lines = ["# Synthetic activation bundle benchmark", "",
             "All timings are measured on randomly initialized tiny models or tensors.", "",
             f"Device: {device}", "",
             f"Calibration reference: {result['calibration']['reference_seconds']:.4f} s; "
             f"layerwise: {result['calibration']['layerwise_seconds']:.4f} s.",
             f"Maximum covariance difference: {result['calibration']['max_covariance_abs_diff']:.6g}.",
             f"SVD exact: {result['svd']['methods']['exact']['seconds']:.4f} s; "
             f"randomized: {result['svd']['methods']['randomized']['seconds']:.4f} s.",
             f"Refinement reference: {result['refinement']['reference_seconds']:.4f} s; "
             f"optimized: {result['refinement']['optimized_seconds']:.4f} s.",
             f"Refinement bit agreement: {result['refinement']['bit_agreement']:.6f}.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
