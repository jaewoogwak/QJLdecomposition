#!/usr/bin/env python3
"""Independent output and bit analysis of a saved residual bundle checkpoint."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from rescode.qjl.codebook_model import bank_for, restore_base
from rescode.qjl.core import pack_residual, qjl_reference, unpack_signs
from rescode.qjl.exponent_model import PROJECTIONS
from rescode.qjl.residual_bundle_model import (FORMAT, bundle_signs,
    encode_projection_rows, validate_pairs, _normalized_projection, _tie_generator)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--projections", default="gate,up,down")
    p.add_argument("--pairs", default="all", help="Pair indices to analyze; all by default")
    p.add_argument("--synthetic-samples", type=int, default=16)
    p.add_argument("--input-seed", type=int, default=0)
    p.add_argument("--activation-path", type=Path,
                   help="Optional torch dict: layer -> projection -> expert -> routed input tensor")
    p.add_argument("--row-chunk", type=int, default=128)
    return p.parse_args(argv)


def relative(a, b):
    return (torch.linalg.vector_norm(a.float() - b.float()) /
            torch.linalg.vector_norm(b.float()).clamp_min(1e-30)).item()


def cosine(a, b):
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return F.cosine_similarity(a[None], b[None]).item()


def _aggregate(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    result = []
    metrics = ("residual_norm_ratio", "row_cosine", "bit_agreement",
               "bit_preservation", "pair_preservation_disparity",
               "pair_preservation_min", "residual_output_relative_l2",
               "weight_output_relative_l2", "output_cosine")
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        record = dict(zip(group_keys, group))
        record["expert_count"] = len(values)
        for metric in metrics:
            available = [v[metric] for v in values if v.get(metric) is not None]
            record[metric + "_mean"] = sum(available) / len(available) if available else None
            record[metric + "_min"] = min(available) if available else None
        result.append(record)
    return result


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def analyze(model, payload, args, activations=None):
    config = payload["config"]
    banks = {}
    rows = []
    selected = (sorted(int(i) for i in payload["layers"])
                if args.layers == "all" else [int(i) for i in args.layers.split(",")])
    for layer in selected:
        state = payload["layers"][layer]
        block = model.model.layers[layer].mlp
        for key in args.projections.split(","):
            if key not in PROJECTIONS:
                raise ValueError(f"unknown projection: {key}")
            attr = PROJECTIONS[key]
            pairs = state["pairs"][key]
            validate_pairs(pairs, len(state["experts"]))
            for pair_id, (i, j) in enumerate(pairs):
                if args.pairs != "all" and pair_id not in {
                        int(value) for value in args.pairs.split(",")}:
                    continue
                weights = {k: getattr(block.experts[k], attr).weight.detach()
                           for k in (i, j)}
                bank = bank_for(banks, weights[i].shape[1], config["qjl_d_factor"],
                                config["projection_seed"])
                bases, residuals, originals, norms = {}, {}, {}, {}
                for k in (i, j):
                    part = state["experts"][k][key]
                    base = restore_base(part["base"], weights[k].device, "exponent-mantissa")
                    bases[k] = base
                    originals[k] = weights[k].float()
                    residuals[k] = originals[k] - base.decode()
                    norms[k] = part["norms"]
                packed = {k: pack_residual(residuals[k], bank,
                                           row_chunk=args.row_chunk)[0].cpu() for k in (i, j)}
                ba = unpack_signs(packed[i], bank.d)
                bb = unpack_signs(packed[j], bank.d)
                agreement = (ba == bb).float().mean().item()
                row_cosine = F.cosine_similarity(residuals[i], residuals[j], dim=1,
                                                  eps=1e-30).mean().item()
                methods = {"none": packed}
                for method in ("majority", "magnitude-aware"):
                    if method == config["bundle_method"]:
                        shared = state["sketches"][key][pair_id]
                    else:
                        projection = None
                        if method == "magnitude-aware":
                            projection = lambda start, end: (
                                _normalized_projection(weights[i], state["experts"][i][key]["base"]["beta"],
                                                       bank, start, end, norms[i]),
                                _normalized_projection(weights[j], state["experts"][j][key]["base"]["beta"],
                                                       bank, start, end, norms[j]))
                        shared = bundle_signs(packed[i], packed[j], bank.d, method,
                            _tie_generator(config["bundle_seed"], layer, key, pair_id).initial_seed(),
                            projections=projection, row_chunk=args.row_chunk)
                    methods[method] = {i: shared, j: shared}
                if config["bundle_method"] == "none":
                    methods["none"] = {k: state["sketches"][key][k] for k in (i, j)}
                for method, sketch in methods.items():
                    preservation = {k: (unpack_signs(sketch[k], bank.d) ==
                                        (ba if k == i else bb)).float().mean().item()
                                    for k in (i, j)}
                    disparity = abs(preservation[i] - preservation[j])
                    minimum = min(preservation.values())
                    for k in (i, j):
                        if activations is None:
                            gen = torch.Generator(device="cpu").manual_seed(
                                args.input_seed + layer * 1000003 + list(PROJECTIONS).index(key) * 9176)
                            x = torch.randn(args.synthetic_samples, weights[k].shape[1],
                                            generator=gen).to(weights[k].device, dtype=weights[k].dtype)
                        else:
                            x = activations[layer][key][k].to(weights[k].device, dtype=weights[k].dtype)
                        p = bank.get(weights[k].shape[1], x.device)
                        z = x.reshape(-1, x.shape[-1]).to(p.dtype) @ p
                        qjl = qjl_reference(z, sketch[k].to(z.device),
                                            norms[k].to(z.device), bank.d)
                        exact_r = F.linear(x.float(), residuals[k])
                        exact_w = F.linear(x.float(), originals[k])
                        approx = bases[k](x).float() + qjl
                        rows.append({"layer": layer, "projection": key, "pair": pair_id,
                                     "expert": k, "partner": j if k == i else i,
                                     "method": method,
                                     "residual_norm_ratio": (
                                         torch.linalg.vector_norm(residuals[k]) /
                                         torch.linalg.vector_norm(originals[k]).clamp_min(1e-30)).item(),
                                     "row_cosine": row_cosine, "bit_agreement": agreement,
                                     "bit_preservation": preservation[k],
                                     "pair_preservation_disparity": disparity,
                                     "pair_preservation_min": minimum,
                                     "residual_output_relative_l2": relative(qjl, exact_r),
                                     "weight_output_relative_l2": relative(approx, exact_w),
                                     "output_cosine": cosine(approx, exact_w)})
                print(f"[analysis] layer {layer} {key} pair {pair_id}", flush=True)
    return rows


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT:
        raise ValueError("wrong residual bundle checkpoint")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True)
    model.eval()
    activations = (torch.load(args.activation_path, map_location="cpu", weights_only=False)
                   if args.activation_path else None)
    rows = analyze(model, payload, args, activations)
    _csv(args.output_dir / "experts.csv", rows)
    pair_rows = _aggregate(rows, ("layer", "projection", "pair", "method"))
    _csv(args.output_dir / "pairs.csv", pair_rows)
    _csv(args.output_dir / "projections.csv", _aggregate(rows, ("projection", "method")))
    _csv(args.output_dir / "layers.csv", _aggregate(rows, ("layer", "method")))
    (args.output_dir / "aggregate.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "input": "routed" if activations else "synthetic",
        "expert_rows": len(rows), "overall": _aggregate(rows, ("method",)),
        "layers": _aggregate(rows, ("layer", "method")),
        "projections": _aggregate(rows, ("projection", "method"))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
