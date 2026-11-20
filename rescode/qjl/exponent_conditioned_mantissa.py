"""Independent weight-only diagnostics for exponent-conditioned mantissa scales.

This module does not create a checkpoint or change the existing Base encoder.
Only present finite, nonzero BF16 exponent groups receive sparse beta entries.
"""

from __future__ import annotations

import math

import torch

from .exponent_mantissa_base import (
    _scale_from_bits, encode_exponent_mantissa_base,
    exponent_mantissa_base_bytes, fit_row_beta, reconstruct_base_from_weight,
)

METHODS = ("A", "B1", "B2")


def _fields(weight: torch.Tensor) -> dict:
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("expected a 2D BF16 weight matrix")
    bits = weight.contiguous().view(torch.uint16).to(torch.int32)
    exponent = (bits >> 7) & 255
    mantissa = bits & 127
    normal = (exponent > 0) & (exponent < 255)
    subnormal = (exponent == 0) & (mantissa != 0)
    valid = normal | subnormal
    return {"bits": bits, "exponent": exponent, "mantissa": mantissa,
            "valid": valid, "subnormal": subnormal,
            "zero": (exponent == 0) & (mantissa == 0),
            "inf": (exponent == 255) & (mantissa == 0),
            "nan": (exponent == 255) & (mantissa != 0)}


def fit_exponent_beta(weight: torch.Tensor, row_beta: torch.Tensor,
                      min_exponent_count: int = 1) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Fit FP32 beta by row and original BF16 exponent; absent groups stay empty.

    The dense [rows, 256] arrays are analysis temporaries. Storage estimates
    count only present entries in a sparse row-offset/exponent/beta table.
    """
    if min_exponent_count not in (1, 4, 16, 64):
        raise ValueError("min_exponent_count must be 1, 4, 16, or 64")
    fields = _fields(weight)
    rows = weight.shape[0]
    exponent = fields["exponent"].long()
    valid = fields["valid"]
    fraction = fields["mantissa"].float() / 128.0
    magnitude = torch.where(fields["subnormal"], fraction, 1.0 + fraction)
    counts = torch.zeros((rows, 256), dtype=torch.int32, device=weight.device)
    sums = torch.zeros((rows, 256), dtype=torch.float32, device=weight.device)
    counts.scatter_add_(1, exponent, valid.int())
    sums.scatter_add_(1, exponent, torch.where(valid, magnitude, 0.0))
    present = counts > 0
    fitted = sums / counts.clamp_min(1).float()
    fallback = present & (counts < min_exponent_count)
    beta = torch.where(fallback, row_beta.float()[:, None], fitted)
    beta = torch.where(present, beta, 0.0)
    return beta, counts, fields


def _reconstruct(weight: torch.Tensor, beta: torch.Tensor,
                 fields: dict, special_base: torch.Tensor) -> torch.Tensor:
    """Reconstruct finite nonzero entries; preserve legacy special handling."""
    selected = beta.gather(1, fields["exponent"].long())
    sign = torch.where((fields["bits"] & 0x8000) != 0, -1.0, 1.0)
    value = sign * _scale_from_bits(fields["bits"]) * selected
    return torch.where(fields["valid"], value, special_base)


def _metrics(weight: torch.Tensor, base: torch.Tensor) -> tuple[dict, torch.Tensor]:
    original = weight.float()
    finite = torch.isfinite(original)
    residual = torch.where(finite, original - base, 0.0)
    if not torch.allclose(base[finite] + residual[finite], original[finite],
                          rtol=1e-6, atol=1e-6):
        raise AssertionError("W = B + R reconstruction failed on finite weights")
    weight_sq_rows = torch.where(finite, original, 0.0).double().square().sum(1)
    residual_sq_rows = residual.double().square().sum(1)
    weight_sq = float(weight_sq_rows.sum())
    residual_sq = float(residual_sq_rows.sum())
    nonzero = weight_sq_rows > 0
    ratios = (residual_sq_rows[nonzero] / weight_sq_rows[nonzero]).sqrt().float()
    result = {
        "weight_squared_norm": weight_sq,
        "residual_squared_norm": residual_sq,
        "rho": math.sqrt(residual_sq / weight_sq) if weight_sq else None,
        "row_ratio_mean": float(ratios.mean()) if ratios.numel() else None,
        "row_ratio_median": float(torch.quantile(ratios, 0.5)) if ratios.numel() else None,
        "row_ratio_p90": float(torch.quantile(ratios, 0.9)) if ratios.numel() else None,
        "row_ratio_max": float(ratios.max()) if ratios.numel() else None,
        "nonzero_row_count": int(nonzero.sum()),
    }
    return result, residual_sq_rows


def _storage(state: dict, rows: int, cols: int, exponent_groups: int) -> dict:
    def nbytes(key):
        value = state[key]
        return value.numel() * value.element_size()

    sign = nbytes("codes")
    exponent = sum(nbytes(key) for key in ("exp_data", "exp_offsets", "exp_modes"))
    specials = nbytes("special_indices") + nbytes("special_bits")
    beta = 2 * exponent_groups
    ids = exponent_groups  # One uint8 original exponent ID per present group.
    offsets = 4 * (rows + 1)  # CSR-style uint32 row offsets.
    new_base = sign + exponent + specials + beta + ids + offsets
    existing = exponent_mantissa_base_bytes(state)
    qjl_n = rows * ((cols + 7) // 8 + 4)
    qjl_2n = rows * ((2 * cols + 7) // 8 + 4)
    return {"packed_sign_bytes": sign, "packed_exponent_bytes": exponent,
            "special_table_bytes": specials, "fp16_beta_table_bytes": beta,
            "exponent_id_bytes": ids, "row_offset_bytes": offsets,
            "B2_estimated_base_bytes": new_base, "A_measured_base_bytes": existing,
            "qjl_d_n_projected_bytes": qjl_n, "qjl_d_2n_projected_bytes": qjl_2n}


def analyze_matrix(weight: torch.Tensor, min_exponent_count: int = 1) -> tuple[dict, dict]:
    """Return A/B1/B2 residuals, sparse storage estimates, and exponent stats."""
    fields = _fields(weight)
    state = encode_exponent_mantissa_base(weight)
    stored_row_beta = state["beta"].to(weight.device)
    row_beta_fp32 = fit_row_beta(weight)
    base_a = reconstruct_base_from_weight(weight, stored_row_beta)
    beta, counts, fields = fit_exponent_beta(weight, row_beta_fp32,
                                             min_exponent_count)
    base_b1 = _reconstruct(weight, beta, fields, base_a)
    base_b2 = _reconstruct(weight, beta.half().float(), fields, base_a)
    base_a_fp32 = _reconstruct(weight, row_beta_fp32[:, None].expand(-1, 256),
                               fields, base_a)
    metrics_a, rows_a = _metrics(weight, base_a)
    metrics_b1, rows_b1 = _metrics(weight, base_b1)
    metrics_b2, rows_b2 = _metrics(weight, base_b2)
    oracle_a, _ = _metrics(weight, base_a_fp32)
    if metrics_b1["residual_squared_norm"] > (
            oracle_a["residual_squared_norm"] * (1 + 1e-6) + 1e-60):
        raise AssertionError("FP32 exponent-conditioned fit worsened FP32 row-wise optimum")

    present = counts > 0
    sizes = counts[present].float()
    group_count = int(present.sum())
    storage = _storage(state, *weight.shape, group_count)
    baseline_sq = metrics_a["residual_squared_norm"]
    worse_rows = int((rows_b2 > rows_a * (1 + 1e-10) + 1e-60).sum())
    worse_matrix = metrics_b2["residual_squared_norm"] > baseline_sq * (1 + 1e-10) + 1e-60

    def improvements(candidate: dict) -> dict:
        rho_a, rho_b = metrics_a["rho"], candidate["rho"]
        return {
            "absolute_rho_improvement_vs_A": rho_a - rho_b if rho_a is not None else None,
            "relative_rho_improvement_vs_A": (1 - rho_b / rho_a) if rho_a else None,
            "residual_energy_ratio_vs_A": (candidate["residual_squared_norm"] / baseline_sq)
            if baseline_sq else None,
        }

    records = {}
    for method, metrics in (("A", metrics_a), ("B1", metrics_b1), ("B2", metrics_b2)):
        base_bytes = storage["A_measured_base_bytes"] if method == "A" else (
            storage["B2_estimated_base_bytes"] if method == "B2" else None)
        elements = weight.numel()
        records[method] = {"method": method, "rows": weight.shape[0],
                           "cols": weight.shape[1], "elements": elements, **metrics,
                           **improvements(metrics), "base_bytes": base_bytes,
                           "base_bits_per_weight": 8 * base_bytes / elements if base_bytes is not None else None,
                           "projected_total_bits_per_weight_d_n": (
                               8 * (base_bytes + storage["qjl_d_n_projected_bytes"]) / elements)
                           if base_bytes is not None else None,
                           "projected_total_bits_per_weight_d_2n": (
                               8 * (base_bytes + storage["qjl_d_2n_projected_bytes"]) / elements)
                           if base_bytes is not None else None,
                           "projected_memory_reduction_vs_bf16_d_n": (
                               1 - (base_bytes + storage["qjl_d_n_projected_bytes"]) /
                               (2 * elements)) if base_bytes is not None else None,
                           "projected_memory_reduction_vs_bf16_d_2n": (
                               1 - (base_bytes + storage["qjl_d_2n_projected_bytes"]) /
                               (2 * elements)) if base_bytes is not None else None,
                           "qjl_d_n_projected_bytes": storage["qjl_d_n_projected_bytes"],
                           "qjl_d_2n_projected_bytes": storage["qjl_d_2n_projected_bytes"],
                           "fp16_worse_row_count_vs_A": worse_rows if method == "B2" else None,
                           "fp16_worse_matrix_vs_A": bool(worse_matrix) if method == "B2" else None,
                           "storage_kind": ("measured_existing_base" if method == "A" else
                                            "estimated_sparse_base" if method == "B2" else
                                            "oracle_only_no_storage_claim")}

    unique_per_row = present.sum(1).float()
    beta_difference = (beta - row_beta_fp32[:, None]).abs()[present]
    beta_values = beta[present]
    group_counts_per_row = present.sum(1).clamp_min(1)
    row_beta_means = beta.sum(1) / group_counts_per_row
    row_beta_std = ((((beta - row_beta_means[:, None]).square() * present).sum(1)
                     / group_counts_per_row).sqrt())
    multi_group_rows = present.sum(1) > 1
    valid = fields["valid"]
    positive_zero = fields["zero"] & ((fields["bits"] & 0x8000) == 0)
    negative_zero = fields["zero"] & ((fields["bits"] & 0x8000) != 0)
    histogram = {
        "1": int((sizes == 1).sum()), "2-3": int(((sizes >= 2) & (sizes <= 3)).sum()),
        "4-15": int(((sizes >= 4) & (sizes <= 15)).sum()),
        "16-63": int(((sizes >= 16) & (sizes <= 63)).sum()),
        "64+": int((sizes >= 64).sum()),
    }
    statistics = {
        "unique_exponents_per_row_mean": float(unique_per_row.mean()),
        "unique_exponents_per_row_median": float(torch.quantile(unique_per_row, 0.5)),
        "unique_exponents_per_row_min": int(unique_per_row.min()),
        "unique_exponents_per_row_max": int(unique_per_row.max()),
        "distinct_exponent_groups": group_count,
        "group_size_histogram": histogram,
        "group_size_mean": float(sizes.mean()) if sizes.numel() else None,
        "group_size_median": float(torch.quantile(sizes, 0.5)) if sizes.numel() else None,
        "group_size_p90": float(torch.quantile(sizes, 0.9)) if sizes.numel() else None,
        "group_size_max": int(sizes.max()) if sizes.numel() else None,
        "beta_across_groups_std": float(beta_values.std(unbiased=False)) if beta_values.numel() else None,
        "beta_across_groups_range": float(beta_values.max() - beta_values.min()) if beta_values.numel() else None,
        "within_row_beta_std_mean": (float(row_beta_std[multi_group_rows].mean())
                                     if multi_group_rows.any() else 0.0),
        "within_row_beta_std_max": (float(row_beta_std[multi_group_rows].max())
                                    if multi_group_rows.any() else 0.0),
        "beta_vs_row_mean_abs": float(beta_difference.mean()) if beta_difference.numel() else None,
        "beta_vs_row_max_abs": float(beta_difference.max()) if beta_difference.numel() else None,
        "beta_vs_row_weighted_mean_abs": (float((beta_difference * sizes).sum() / sizes.sum())
                                          if sizes.numel() else None),
        "rare_group_fallback_count": int(((counts > 0) & (counts < min_exponent_count)).sum()),
        "normal_count": int((valid & ~fields["subnormal"]).sum()),
        "subnormal_count": int(fields["subnormal"].sum()),
        "positive_zero_count": int(positive_zero.sum()),
        "negative_zero_count": int(negative_zero.sum()),
        "inf_count": int(fields["inf"].sum()),
        "nan_count": int(fields["nan"].sum()),
        **storage,
    }
    return records, statistics
