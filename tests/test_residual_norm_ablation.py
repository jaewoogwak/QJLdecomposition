"""Small deterministic checks for the BF16 residual norm ablation."""

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/QJL/analyze_residual_norm_ablation.py"
spec = importlib.util.spec_from_file_location("residual_norm_ablation", SCRIPT)
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


def test_known_shared_base_and_singletons():
    w0 = torch.tensor([[1., 2., -4.], [0.5, -1., 2.]], dtype=torch.bfloat16)
    w1 = torch.tensor([[3., 4., -2.], [1.5, -3., 6.]], dtype=torch.bfloat16)
    weights = {
        0: {p: w0.clone() for p in ablation.PROJECTIONS},
        1: {p: w1.clone() for p in ablation.PROJECTIONS},
    }
    originals = {e: {p: w.clone() for p, w in matrices.items()}
                 for e, matrices in weights.items()}
    saliency = torch.tensor([1., 3.])
    bases = ablation.rescode_shared_bases(weights, [0, 1], saliency, 0.)
    for p in ablation.PROJECTIONS:
        expected = (0.25 * w0.float() + 0.75 * w1.float()).to(torch.bfloat16)
        assert torch.equal(bases[p], expected)
    rows, checks = ablation.analyze_group(0, 0, [0, 1], weights, saliency, 0.,
                                          verify_packed=True)
    assert checks == 3
    assert len(rows) == 6
    assert all(not row["singleton"] for row in rows)
    assert all(row["r_rescode"] >= 0 and row["r_mantissa"] >= 0 for row in rows)
    for e, matrices in originals.items():
        for p, w in matrices.items():
            assert torch.equal(weights[e][p], w)
    singleton, _ = ablation.analyze_group(0, 1, [0], {0: weights[0]}, saliency, 0.)
    assert len(singleton) == 3
    assert all(row["singleton"] and row["r_rescode"] is None for row in singleton)
    summary = ablation.describe(rows + singleton)
    assert summary["count"] == 6 and summary["singleton_count"] == 3


def test_residual_identity_and_bf16_storage():
    w = torch.tensor([[0.125, -0.75, 2.0], [0., 1.25, -3.]], dtype=torch.bfloat16)
    beta = ablation.fit_row_beta(w).to(torch.float16)
    base = ablation.reconstruct_base_from_weight(w, beta)
    ratio, w2, r2 = ablation.checked_ratio(w, base)
    residual = w.float() - base
    assert torch.equal(base + residual, w.float())
    assert abs(ratio - (r2 / w2) ** 0.5) < 1e-12
    assert torch.all(torch.isfinite(residual))
    try:
        ablation.checked_ratio(w.float(), base)
    except ValueError:
        pass
    else:
        raise AssertionError("non-BF16 original weight was accepted")


def test_cluster_settings_match_rescode_budget_path():
    from rescode.merging.expert_clusters import form_clusters, residual_params_fn_for
    stats = {layer: {"saliency": torch.tensor([5., 1., 1., 0.]),
                     "dist": torch.tensor([[0., 3., 3., 3.], [3., 0., 1., 2.],
                                           [3., 1., 0., 2.], [3., 2., 2., 0.]])}
             for layer in range(2)}
    groups, _ = form_clusters(stats, num_groups=3, singleton_ratio=0.25,
                              expert_params=24,
                              residual_params_fn=residual_params_fn_for(2, 2, 1, 3),
                              linkage="average", budget_aware=True)
    assert all([0] in groups[layer] for layer in groups)
    assert all(sorted(e for g in groups[layer] for e in g) == list(range(4))
               for layer in groups)


if __name__ == "__main__":
    test_known_shared_base_and_singletons()
    test_residual_identity_and_bf16_storage()
    test_cluster_settings_match_rescode_budget_path()
    print("3 residual norm ablation tests passed")
