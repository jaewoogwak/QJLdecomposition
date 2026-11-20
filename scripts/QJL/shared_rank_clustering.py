"""Weight-only grouping and exact rank-constrained Frobenius diagnostics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Group:
    members: tuple[int, ...]
    weight_sum: torch.Tensor
    energy: float


def _validate(weights: torch.Tensor, rank: int) -> None:
    if weights.ndim != 3 or weights.shape[0] < 1:
        raise ValueError("weights must have shape (experts, rows, columns)")
    if weights.dtype != torch.float32 or not torch.isfinite(weights).all():
        raise ValueError("weights must be finite FP32 matrices")
    if rank < 0:
        raise ValueError("rank must be nonnegative")


def captured_energy(mean: torch.Tensor, rank: int) -> float:
    """Squared top singular values of a group mean, with effective rank clamped."""
    effective = min(rank, *mean.shape)
    if effective == 0:
        return 0.0
    singular = torch.linalg.svdvals(mean.float())[:effective]
    return float(singular.square().sum(dtype=torch.float64).item())


def truncated_base(mean: torch.Tensor, rank: int) -> torch.Tensor:
    effective = min(rank, *mean.shape)
    if effective == 0:
        return torch.zeros_like(mean)
    u, s, vh = torch.linalg.svd(mean.float(), full_matrices=False)
    return (u[:, :effective] * s[:effective]) @ vh[:effective]


class ResidualCosts:
    """Active group sums and memoized rank-constrained costs for one rank."""

    def __init__(self, weights: torch.Tensor, rank: int):
        _validate(weights, rank)
        self.weights = weights
        self.rank = min(rank, *weights.shape[1:])
        self.energies = weights.square().flatten(1).sum(dim=1, dtype=torch.float64).tolist()
        self.groups = {
            (i,): Group((i,), weights[i], self.energies[i])
            for i in range(len(weights))
        }
        self._costs: dict[tuple[int, ...], float] = {}
        flat = weights.flatten(1)
        self.gram = (flat @ flat.T).double().cpu()

    def cost(self, group: Group) -> float:
        key = group.members
        if key not in self._costs:
            if self.rank == 0:
                value = group.energy
            else:
                value = group.energy - len(key) * captured_energy(
                    group.weight_sum / len(key), self.rank)
            # Subtraction of large near-equal FP32 quantities can be slightly negative.
            if value < -1e-4 * max(group.energy, 1.0):
                raise ArithmeticError(f"negative residual cost for {key}: {value}")
            self._costs[key] = max(value, 0.0)
        return self._costs[key]

    @staticmethod
    def union(a: Group, b: Group) -> Group:
        return Group(tuple(sorted(a.members + b.members)),
                     a.weight_sum + b.weight_sum, a.energy + b.energy)

    def delta(self, a: Group, b: Group) -> float:
        merged = self.union(a, b)
        delta = self.cost(merged) - self.cost(a) - self.cost(b)
        if delta < -1e-4 * max(merged.energy, 1.0):
            raise ArithmeticError(f"negative merge cost for {merged.members}: {delta}")
        return max(delta, 0.0)

    def cosine_distance(self, a: Group, b: Group) -> float:
        ia, ib = list(a.members), list(b.members)
        dot = float(self.gram[ia][:, ib].sum())
        na = float(self.gram[ia][:, ia].sum())
        nb = float(self.gram[ib][:, ib].sum())
        if na <= 0 or nb <= 0:
            return 0.0 if na <= 0 and nb <= 0 else 1.0
        return 1.0 - max(-1.0, min(1.0, dot / math.sqrt(na * nb)))

    def cluster(self, count: int, method: str, candidate_shortlist: int = 8
                ) -> tuple[tuple[int, ...], ...]:
        """Agglomerate by cosine or residual cost; shortlist=0 is exhaustive."""
        if method not in ("cosine", "rank_aware"):
            raise ValueError("method must be cosine or rank_aware")
        if not 1 <= count <= len(self.weights) or candidate_shortlist < 0:
            raise ValueError("invalid group count or shortlist size")
        active = dict(self.groups)
        while len(active) > count:
            keys = sorted(active)
            pairs = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]]
            by_cosine = sorted(pairs, key=lambda pair: (
                self.cosine_distance(active[pair[0]], active[pair[1]]), pair))
            if method == "rank_aware":
                candidates = (by_cosine[:candidate_shortlist] if candidate_shortlist
                              else by_cosine)
                a, b = min(candidates, key=lambda pair: (
                    self.delta(active[pair[0]], active[pair[1]]), pair))
            else:
                a, b = by_cosine[0]
            merged = self.union(active.pop(a), active.pop(b))
            active[merged.members] = merged
        return tuple(sorted(active))


def random_groups(expert_count: int, group_count: int, seed: int,
                  sizes: tuple[int, ...] | None = None) -> tuple[tuple[int, ...], ...]:
    if not 1 <= group_count <= expert_count:
        raise ValueError("invalid group count")
    rng = random.Random(seed)
    order = list(range(expert_count))
    rng.shuffle(order)
    if sizes is None:
        groups = [order[i::group_count] for i in range(group_count)]
    else:
        if len(sizes) != group_count or any(s < 1 for s in sizes) or sum(sizes) != expert_count:
            raise ValueError("sizes must partition the experts into nonempty groups")
        groups, offset = [], 0
        for size in sizes:
            groups.append(order[offset:offset + size])
            offset += size
    return tuple(sorted(tuple(sorted(group)) for group in groups))


def group_metrics(weights: torch.Tensor, groups: tuple[tuple[int, ...], ...], rank: int):
    """Use explicit FP32 residuals for per-expert results and global squared sums."""
    _validate(weights, rank)
    if sorted(i for group in groups for i in group) != list(range(len(weights))):
        raise ValueError("groups must partition all experts exactly once")
    rows, cols = weights.shape[1:]
    effective = min(rank, rows, cols)
    per_expert = [None] * len(weights)
    for group_index, members in enumerate(groups):
        mean = weights[list(members)].mean(dim=0)
        base = truncated_base(mean, effective)
        for expert in members:
            weight_energy = float(weights[expert].square().sum(dtype=torch.float64))
            residual_energy = float((weights[expert] - base).square().sum(dtype=torch.float64))
            per_expert[expert] = {
                "expert": expert, "group": group_index,
                "weight_squared_norm": weight_energy,
                "residual_squared_norm": residual_energy,
                "residual_ratio": math.sqrt(residual_energy / weight_energy)
                if weight_energy else None,
            }
    total_weight = sum(item["weight_squared_norm"] for item in per_expert)
    total_residual = sum(item["residual_squared_norm"] for item in per_expert)
    ratios = sorted(item["residual_ratio"] for item in per_expert
                    if item["residual_ratio"] is not None)
    def percentile(q):
        if not ratios:
            return None
        position = (len(ratios) - 1) * q
        low = int(position)
        return ratios[low] + (ratios[min(low + 1, len(ratios) - 1)] - ratios[low]) * (position - low)
    summary = {
        "weight_squared_norm": total_weight,
        "residual_squared_norm": total_residual,
        "rho_global": math.sqrt(total_residual / total_weight) if total_weight else None,
        "captured_energy_fraction": 1 - total_residual / total_weight if total_weight else None,
        "expert_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
        "expert_ratio_median": percentile(0.5),
        "expert_ratio_p90": percentile(0.9),
        "expert_ratio_max": ratios[-1] if ratios else None,
        "effective_rank": effective,
        "bf16_base_bits": len(groups) * 16 * effective * (rows + cols),
        "bf16_original_bits": len(weights) * 16 * rows * cols,
    }
    return summary, per_expert
