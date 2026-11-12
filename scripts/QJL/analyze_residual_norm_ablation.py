#!/usr/bin/env python3
"""Compare ResCode and exponent/mantissa Base residuals on original Qwen BF16 weights.

The ResCode cluster assignment and weighted merge are imported unchanged.  Only
calibration statistics may require a full model; the ablation itself reads one
group at a time from the original safetensors, before any merge can alter them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rescode.merging.expert_clusters import form_clusters, residual_params_fn_for
from rescode.merging.grouping_qwen import _merge_mlp_experts_by_usage_frequency_weighting
from rescode.qjl.exponent_mantissa_base import (
    PackedExponentMantissaBase,
    encode_exponent_mantissa_base,
    fit_row_beta,
    reconstruct_base_from_weight,
)
from rescode.utils.constants import FP32_EPS

PROJECTIONS = ("gate", "up", "down")
FIELDS = ("layer", "group", "expert", "projection", "group_size", "singleton",
          "r_rescode", "r_mantissa", "weight_sq_norm", "rescode_residual_sq_norm",
          "mantissa_residual_sq_norm")


def parse_layers(value: str, count: int) -> list[int]:
    if value == "all":
        return list(range(count))
    result = sorted({int(part.strip()) for part in value.split(",")})
    if not result or result[0] < 0 or result[-1] >= count:
        raise ValueError(f"layers must be within 0..{count - 1}")
    return result


def checked_ratio(weight: torch.Tensor, base: torch.Tensor) -> tuple[float, float, float]:
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("original weight must be a BF16 matrix")
    if base.shape != weight.shape or base.dtype != torch.float32:
        raise ValueError("base must be reconstructed as an FP32 matrix of the same shape")
    original = weight.float()
    denominator = float(original.square().sum(dtype=torch.float64))
    residual = float((original - base).square().sum(dtype=torch.float64))
    ratio = math.sqrt(residual / denominator) if denominator else (0.0 if residual == 0 else math.inf)
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("non-finite or negative residual ratio")
    return ratio, denominator, residual


@torch.no_grad()
def rescode_shared_bases(weights: dict[int, dict[str, torch.Tensor]],
                         members: list[int], saliency: torch.Tensor,
                         delta: float) -> dict[str, torch.Tensor]:
    """Call ResCode's exact weighted merge on local group wrappers.

    Its representative weight is cloned; all original BF16 matrices stay intact.
    ResCode performs an FP32 weighted sum and copies it into BF16 storage.
    """
    if not members or len(set(members)) != len(members):
        raise ValueError("group members must be unique and nonempty")
    if set(members) != set(weights):
        raise ValueError("loaded experts do not match the cluster membership")
    frequency = saliency[members].float() + float(delta)
    if not torch.isfinite(frequency).all() or (frequency < 0).any():
        raise ValueError("invalid saliency weights")
    coefficients = frequency / (frequency.sum() + FP32_EPS)
    # ResCode adds FP32_EPS to the denominator, so the sum can be just below 1.
    expected_sum = frequency.sum() / (frequency.sum() + FP32_EPS)
    if not torch.isclose(coefficients.sum(), expected_sum, atol=1e-7):
        raise AssertionError("shared-base coefficients differ from ResCode")
    wrappers = []
    for i, expert in enumerate(members):
        matrices = weights[expert]
        if set(matrices) != set(PROJECTIONS):
            raise ValueError("each expert must have Gate/Up/Down matrices")
        wrappers.append(SimpleNamespace(**{
            f"{projection}_proj": SimpleNamespace(weight=(matrices[projection].clone()
                                                      if i == 0 else matrices[projection]))
            for projection in PROJECTIONS
        }))
    ffn = SimpleNamespace(experts=wrappers)
    _merge_mlp_experts_by_usage_frequency_weighting(
        ffn, torch.zeros(len(members), dtype=torch.long), frequency.cpu())
    return {projection: getattr(ffn.experts[0], f"{projection}_proj").weight
            for projection in PROJECTIONS}


@torch.no_grad()
def analyze_group(layer: int, group_id: int, members: list[int],
                  weights: dict[int, dict[str, torch.Tensor]],
                  saliency: torch.Tensor, delta: float,
                  verify_packed: bool = False) -> tuple[list[dict], int]:
    bases = rescode_shared_bases(weights, members, saliency, delta)
    rows = []
    packed_checks = 0
    for projection in PROJECTIONS:
        base_bf16 = bases[projection]
        if base_bf16.dtype != torch.bfloat16:
            raise AssertionError("ResCode shared base must be stored as BF16")
        base = base_bf16.float()
        shapes = {tuple(weights[e][projection].shape) for e in members}
        if len(shapes) != 1 or base.shape != weights[members[0]][projection].shape:
            raise ValueError("inconsistent projection shapes in a group")
        # Verify the exact ResCode operation and BF16 storage on one matrix.
        if len(members) > 1:
            coeff = (saliency[members].float() + delta)
            coeff = coeff / (coeff.sum() + FP32_EPS)
            reference = sum(coeff[k] * weights[e][projection].float()
                            for k, e in enumerate(members)).to(torch.bfloat16)
            if not torch.equal(reference, base_bf16):
                raise AssertionError("shared base differs from ResCode weighted merge")
        for j, expert in enumerate(members):
            weight = weights[expert][projection]
            beta = fit_row_beta(weight).to(torch.float16)
            if not torch.isfinite(beta).all():
                raise ValueError("FP16 beta is not finite")
            mantissa_base = reconstruct_base_from_weight(weight, beta)
            if mantissa_base.shape != weight.shape or mantissa_base.dtype != torch.float32:
                raise AssertionError("mantissa Base shape/dtype mismatch")
            if verify_packed and j == 0:
                state = encode_exponent_mantissa_base(weight)
                if not torch.equal(state["beta"], beta.cpu()):
                    raise AssertionError("stored beta differs from encoder")
                decoded = PackedExponentMantissaBase(state, device="cpu").decode()
                if not torch.equal(decoded, mantissa_base.cpu()):
                    raise AssertionError("mantissa Base differs from packed decoder")
                packed_checks += 1
                del decoded, state
            r_mantissa, w_sq, m_sq = checked_ratio(weight, mantissa_base)
            if len(members) > 1:
                r_rescode, w_sq_again, s_sq = checked_ratio(weight, base)
                if w_sq != w_sq_again:
                    raise AssertionError("methods used different original weights")
            else:
                r_rescode, s_sq = None, None
            rows.append({"layer": layer, "group": group_id, "expert": expert,
                         "projection": projection, "group_size": len(members),
                         "singleton": len(members) == 1, "r_rescode": r_rescode,
                         "r_mantissa": r_mantissa, "weight_sq_norm": w_sq,
                         "rescode_residual_sq_norm": s_sq,
                         "mantissa_residual_sq_norm": m_sq})
    return rows, packed_checks


def describe(rows: list[dict]) -> dict:
    selected = [r for r in rows if not r["singleton"]]
    result = {"count": len(selected), "singleton_count": len(rows) - len(selected)}
    for field in ("r_rescode", "r_mantissa"):
        values = np.asarray([r[field] for r in selected], dtype=np.float64)
        result[field] = ({"mean": float(values.mean()), "median": float(np.median(values)),
                          "p90": float(np.percentile(values, 90))} if len(values) else None)
    if selected:
        weight_sq = sum(r["weight_sq_norm"] for r in selected)
        for name, residual in (("r_rescode", "rescode_residual_sq_norm"),
                               ("r_mantissa", "mantissa_residual_sq_norm")):
            result[name]["aggregate_frobenius_ratio"] = math.sqrt(
                sum(r[residual] for r in selected) / weight_sq)
        result["mean_difference_rescode_minus_mantissa"] = float(np.mean(
            [r["r_rescode"] - r["r_mantissa"] for r in selected]))
    return result


def model_snapshot(model: str, path: Path | None) -> Path:
    def complete(root: Path) -> bool:
        index_path = root / "model.safetensors.index.json"
        if not (root / "config.json").is_file() or not index_path.is_file():
            return False
        shards = set(json.loads(index_path.read_text())["weight_map"].values())
        return all((root / shard).is_file() for shard in shards)

    if path is not None:
        root = path.expanduser().resolve()
    else:
        patterns = ["config.json", "generation_config.json", "*.safetensors",
                    "*.safetensors.index.json", "tokenizer*", "vocab*", "merges*",
                    "special_tokens_map.json", "added_tokens.json"]
        try:
            root = Path(snapshot_download(repo_id=model, allow_patterns=patterns,
                                          local_files_only=True))
        except LocalEntryNotFoundError:
            root = None
        if root is None or not complete(root):
            root = Path(snapshot_download(repo_id=model, allow_patterns=patterns))
    if not complete(root):
        raise FileNotFoundError(f"incomplete local Qwen snapshot: {root}")
    return root


def calibration_stats(args: argparse.Namespace, model_dir: Path) -> dict:
    if args.cache_path.is_file():
        print(f"[ablation] loading ResCode calibration cache: {args.cache_path}", flush=True)
        return torch.load(args.cache_path, map_location="cpu", weights_only=False)
    if not args.calibration_path.is_file():
        if args.calib_dataset not in ("dclm", "wikipedia", "slimpajama", "starcoder"):
            raise FileNotFoundError(f"calibration file missing: {args.calibration_path}")
        default_path = ROOT / "rescode/data" / f"{args.calib_dataset}-calib.jsonl"
        if args.calibration_path.resolve() != default_path.resolve():
            raise FileNotFoundError(f"calibration file missing: {args.calibration_path}")
        subprocess.run([sys.executable, str(ROOT / "scripts/RESCODE/prepare_calib_datasets.py"),
                        "--dataset", args.calib_dataset, "--n_docs",
                        str(max(40000, args.n_sentences * 17))], check=True, cwd=ROOT)
    print("[ablation] collecting ResCode calibration statistics; this may take hours", flush=True)
    from transformers import AutoTokenizer, Qwen2MoeForCausalLM
    from rescode.evaluation import get_calib_dataloder
    from rescode.merging.calib_stats import load_or_collect_calib_stats

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2MoeForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    if not torch.cuda.is_available() or model.get_input_embeddings().weight.device.type != "cuda":
        raise RuntimeError("ResCode calibration requires the model's input embedding on CUDA")
    loader = get_calib_dataloder(
        dataset="c4", tokenizer=tokenizer, max_block_size=2048,
        n_blocks_for_stat=args.n_sentences, batch_size=1, num_workers=4,
        seed=args.calib_seed, data_path=str(args.calibration_path), shuffle_docs=True)
    stats = load_or_collect_calib_stats(model, loader, "qwen", cache_path=str(args.cache_path))
    del model, loader, tokenizer
    torch.cuda.empty_cache()
    return stats


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen1.5-MoE-A2.7B-Chat")
    p.add_argument("--model-path", type=Path)
    p.add_argument("--calib-dataset", default="dclm")
    p.add_argument("--calibration-path", type=Path, default=ROOT / "rescode/data/dclm-calib.jsonl")
    p.add_argument("--cache-path", type=Path,
                   default=ROOT / "results/qwen/calib_stats/dclm-n2400-seed42.pt")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/residual_norm_ablation")
    p.add_argument("--layers", default="all")
    p.add_argument("--num-groups", type=int, default=45)
    p.add_argument("--singleton-ratio", type=float, default=0.15)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--n-sentences", type=int, default=2400)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--max-groups", type=int, default=None,
                   help="Only for a selected-layer smoke test; still cluster using all-layer statistics")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-test-shapes", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not (0 <= args.singleton_ratio < 1) or args.num_groups < 1 or args.delta < 0:
        p.error("invalid grouping or delta setting")
    if args.max_groups is not None and args.max_groups < 1:
        p.error("--max-groups must be positive")
    return args


def main() -> None:
    args = arguments()
    model_dir = model_snapshot(args.model, args.model_path)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("num_hidden_layers") != 24 or config.get("num_experts") != 60:
        raise ValueError("expected original Qwen1.5-MoE-A2.7B with 24 layers and 60 routed experts")
    d_model, d_ff = int(config["hidden_size"]), int(config["moe_intermediate_size"])
    if not args.allow_test_shapes and (d_model, d_ff) != (2048, 1408):
        raise ValueError("expected Qwen1.5-MoE-A2.7B expert shapes (2048, 1408)")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    stats = calibration_stats(args, model_dir)
    if set(stats) != set(range(24)):
        raise ValueError("calibration cache must contain all 24 layers for budget-aware grouping")
    for layer, st in stats.items():
        if st["saliency"].shape != (60,) or st["dist"].shape != (60, 60):
            raise ValueError(f"invalid calibration statistics in layer {layer}")
        if st.get("tokens") != args.n_sentences * 2048:
            raise ValueError(f"calibration cache token count mismatch in layer {layer}; "
                             f"expected {args.n_sentences * 2048}, found {st.get('tokens')}")
    groups, budget = form_clusters(
        stats, num_groups=args.num_groups, singleton_ratio=args.singleton_ratio,
        expert_params=3 * d_model * d_ff,
        residual_params_fn=residual_params_fn_for(d_model, d_ff, args.rank, 3),
        linkage="average", budget_aware=True)
    layers = parse_layers(args.layers, 24)
    for layer, layer_groups in groups.items():
        if sorted(e for group in layer_groups for e in group) != list(range(60)):
            raise AssertionError(f"ResCode groups do not partition the 60 experts in layer {layer}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, checks = [], 0
    with ExitStack() as stack:
        handles = {}

        def load(layer: int, expert: int, projection: str) -> torch.Tensor:
            key = f"model.layers.{layer}.mlp.experts.{expert}.{projection}_proj.weight"
            if key not in index:
                raise KeyError(f"original expert weight missing: {key}")
            shard = index[key]
            if shard not in handles:
                path = model_dir / shard
                if not path.is_file():
                    raise FileNotFoundError(path)
                handles[shard] = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            tensor = handles[shard].get_tensor(key)
            expected = (d_model, d_ff) if projection == "down" else (d_ff, d_model)
            if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected:
                raise ValueError(f"wrong original weight shape/dtype for {key}: {tensor.shape}, {tensor.dtype}")
            return tensor.to(device)

        for layer in layers:
            layer_groups = groups[layer][:args.max_groups] if args.max_groups else groups[layer]
            saliency = stats[layer]["saliency"].to(device)
            checked_non_singleton = False
            for group_id, members in enumerate(layer_groups):
                source = {expert: {projection: load(layer, expert, projection)
                                   for projection in PROJECTIONS} for expert in members}
                verify_packed = group_id == 0 or (len(members) > 1 and not checked_non_singleton)
                analyzed, n_checks = analyze_group(
                    layer, group_id, members, source, saliency,
                    args.delta, verify_packed=verify_packed)
                if len(members) > 1:
                    checked_non_singleton = True
                rows.extend(analyzed)
                checks += n_checks
                print(f"[ablation] layer={layer} group={group_id} size={len(members)}", flush=True)
                del source, analyzed
            if device.type == "cuda":
                torch.cuda.empty_cache()
    path = args.output_dir / "per_expert.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model": args.model, "model_path": str(model_dir), "calibration_cache": str(args.cache_path),
        "calibration_dataset": args.calib_dataset, "calibration_path": str(args.calibration_path),
        "n_sentences": args.n_sentences, "calibration_seed": args.calib_seed,
        "layers": layers, "num_groups": args.num_groups, "singleton_ratio": args.singleton_ratio,
        "linkage": "average", "budget_aware": True, "delta": args.delta, "rank": args.rank,
        "analysis_complete": len(layers) == 24 and args.max_groups is None,
        "max_groups": args.max_groups,
        "packed_decoder_checks": checks, "budget": budget,
        "overall": describe(rows),
        "projections": {projection: describe([r for r in rows if r["projection"] == projection])
                        for projection in PROJECTIONS},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for projection in (*PROJECTIONS, "overall"):
        item = summary["overall"] if projection == "overall" else summary["projections"][projection]
        if item["count"]:
            print(f"[ablation] {projection}: n={item['count']} "
                  f"ResCode median={item['r_rescode']['median']:.6f} "
                  f"Mantissa median={item['r_mantissa']['median']:.6f} "
                  f"ResCode aggregate={item['r_rescode']['aggregate_frobenius_ratio']:.6f} "
                  f"Mantissa aggregate={item['r_mantissa']['aggregate_frobenius_ratio']:.6f}")
    print(f"[ablation] wrote {path} and {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
