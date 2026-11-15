"""Residual-only compression methods for the independent A–E ablation.

The GPTQ update follows the column-wise Hessian error feedback in
Frantar et al., GPTQ (2022), and the public IST-DASLab/gptq reference.
Its quantizer is asymmetric, per output row, with one FP16 scale and
one FP16 offset per row.  There is deliberately no packed INT2 kernel.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rescode.merging.lowrank_init import compute_covariance_init


def packed_int2_bytes(rows: int, cols: int) -> int:
    return rows * ((cols + 3) // 4) + 4 * rows


def pack_int2(codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 2 or codes.min() < 0 or codes.max() > 3:
        raise ValueError("INT2 codes must be a [rows, cols] tensor in 0..3")
    rows, cols = codes.shape
    pad = (-cols) % 4
    if pad:
        codes = F.pad(codes, (0, pad))
    values = codes.to(torch.uint8).reshape(rows, -1, 4)
    return (values[:, :, 0] | (values[:, :, 1] << 2) |
            (values[:, :, 2] << 4) | (values[:, :, 3] << 6)).contiguous()


def unpack_int2(packed: torch.Tensor, cols: int) -> torch.Tensor:
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] != (cols + 3) // 4:
        raise ValueError("invalid packed INT2 shape")
    return torch.stack(tuple((packed >> shift) & 3 for shift in (0, 2, 4, 6)),
                       dim=-1).reshape(packed.shape[0], -1)[:, :cols]


def decode_int2(state: dict, device=None) -> torch.Tensor:
    packed = state["packed"].to(device=device)
    cols = int(state["shape"][1])
    scale = state["scale"].to(device=packed.device, dtype=torch.float32)
    offset = state["offset"].to(device=packed.device, dtype=torch.float32)
    return unpack_int2(packed, cols).float() * scale[:, None] + offset[:, None]


@torch.no_grad()
def gptq_int2(residual: torch.Tensor, calibration: torch.Tensor,
              damp_percent: float = 0.01, block_size: int = 128) -> dict:
    """Hessian-based sequential GPTQ on R, retaining only packed INT2 data.

    H = X.T X; columns are processed in original order.  A diagonal damping
    term makes the inverse defined for limited calibration samples.  A block
    update applies the accumulated error to later columns exactly as in GPTQ.
    Quantizer metadata is rounded to FP16 *before* the feedback updates, so
    fitting targets the matrix that the saved checkpoint actually decodes.
    """
    if residual.ndim != 2 or calibration.ndim != 2 or residual.shape[1] != calibration.shape[1]:
        raise ValueError("residual/calibration dimensions differ")
    if calibration.shape[0] < 2 or block_size < 1 or not 0 <= damp_percent < 1:
        raise ValueError("GPTQ needs at least two samples and valid settings")
    device = residual.device
    w = residual.float().clone()
    x = calibration.to(device=device, dtype=torch.float32)
    rows, cols = w.shape
    minimum = torch.minimum(w.amin(dim=1), torch.zeros(rows, device=device))
    maximum = torch.maximum(w.amax(dim=1), torch.zeros(rows, device=device))
    # A constant row has no quantization error and uses an exact offset.
    scale = ((maximum - minimum) / 3).clamp_min(
        torch.finfo(torch.float16).tiny).to(torch.float16)
    offset = minimum.to(torch.float16)
    if not (torch.isfinite(scale).all() and torch.isfinite(offset).all()):
        raise ValueError("GPTQ metadata is not finite in FP16")
    sf, of = scale.float(), offset.float()
    h = (x.T @ x) / x.shape[0]
    damp = max(float(torch.diag(h).mean()) * damp_percent, 1e-6)
    h.diagonal().add_(damp)
    chol = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(chol)
    # Upper triangular error propagation matrix in GPTQ's block algorithm.
    upper = torch.linalg.cholesky(inverse, upper=True)
    codes = torch.empty((rows, cols), dtype=torch.uint8, device=device)
    for start in range(0, cols, block_size):
        end = min(cols, start + block_size)
        width = end - start
        block = w[:, start:end].clone()
        errors = torch.empty_like(block)
        for j in range(width):
            absolute = start + j
            q = torch.round((block[:, j] - of) / sf).clamp_(0, 3).to(torch.uint8)
            decoded = q.float() * sf + of
            codes[:, absolute] = q
            error = (block[:, j] - decoded) / upper[absolute, absolute]
            errors[:, j] = error
            if j + 1 < width:
                block[:, j + 1:] -= error[:, None] * upper[absolute, absolute + 1:end][None, :]
        if end < cols:
            w[:, end:] -= errors @ upper[start:end, end:]
    state = {"packed": pack_int2(codes).cpu(), "scale": scale.cpu(),
             "offset": offset.cpu(), "shape": (rows, cols),
             "damp_percent": damp_percent, "block_size": block_size,
             "calibration_samples": int(x.shape[0])}
    if state["packed"].numel() + 2 * rows * 2 != packed_int2_bytes(rows, cols):
        raise AssertionError("INT2 physical byte count mismatch")
    return state


class PackedINT2Residual(nn.Module):
    """Packed storage; decode plus FP32 matmul is timed separately."""

    def __init__(self, state: dict, device="cpu"):
        super().__init__()
        self.shape = tuple(state["shape"])
        for key in ("packed", "scale", "offset"):
            self.register_buffer(key, state[key].to(device))

    def decode(self):
        return decode_int2({"packed": self.packed, "scale": self.scale,
                            "offset": self.offset, "shape": self.shape})

    def forward(self, x):
        return F.linear(x.float(), self.decode()).to(x.dtype)


def lowrank_group_bytes(k: int, m: int, n: int, rank: int) -> int:
    return 2 * rank * (m + n + k)


def matched_rank(k: int, m: int, n: int, d: int) -> tuple[int, int]:
    qjl_bytes = k * m * (math.ceil(d / 8) + 4)
    rank = min(m, n, qjl_bytes // (2 * (m + n + k)))
    if rank < 1:
        raise ValueError("QJL storage budget cannot hold rank one")
    return rank, qjl_bytes


@torch.no_grad()
def fit_shared_lowrank(residuals: torch.Tensor, rank: int, seed: int,
                       iterations: int = 0, freqs: torch.Tensor | None = None,
                       alpha: float = 2.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ResCode's frequency-weighted covariance initializer to our R.

    No output-guided fitting or optimization follows this initialization.
    ResCode returns left, right, and codes, with z scaled by 1/alpha.
    """
    if residuals.ndim != 3 or not torch.isfinite(residuals).all():
        raise ValueError("expected finite [experts, rows, cols] residuals")
    k, m, n = residuals.shape
    if not 1 <= rank <= min(m, n) or iterations != 0:
        raise ValueError("invalid rank or requested post-initialization fitting")
    if freqs is None:
        raise ValueError("ResCode frequency weights are required")
    left, right, codes = compute_covariance_init(
        [residuals[i].float() for i in range(k)], freqs.float(), rank, alpha)
    return left.to(torch.bfloat16), right.to(torch.bfloat16), codes.to(torch.bfloat16)


class SharedLowRankResidual(nn.Module):
    def __init__(self, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor,
                 alpha: float = 2.0):
        super().__init__()
        self.register_buffer("a", a)
        self.register_buffer("b", b)
        self.register_buffer("z", z)
        self.alpha = float(alpha)

    def decode(self):
        return self.alpha * (self.a.float() * self.z.float()[None, :]) @ self.b.float()

    def forward(self, x):
        hidden = F.linear(x.float(), self.b.float()) * self.z.float()
        return (self.alpha * F.linear(hidden, self.a.float())).to(x.dtype)
