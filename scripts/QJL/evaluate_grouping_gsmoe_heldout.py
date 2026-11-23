#!/usr/bin/env python3
"""Held-out routed-output test for saved GS-MoE grouping/SVD checkpoints.

This script never fits a basis, coefficient, or SVD. Both train and held-out
activations are evaluated with the same saved BF16 checkpoint tensors.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.QJL.run_grouping_gsmoe import (
    MODEL_ID, PROJECTIONS, WeightReader, _archive_stale, _metrics, _projected_output, _write_csv,
    collect_calibration, load_calibration_layer, validate_groups,
)
from scripts.QJL.run_shared_lowrank_qjl import selected

SHARD = "c4-train.00001-of-01024.json.gz"
SHARD_REPO_PATH = f"en/{SHARD}"
SHARD_REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
SHARD_SHA256 = "b945059cd1a343cabe311881b7840a6f0363f570e745a0eff0e687e266f6b55d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_hashes(path: Path) -> tuple[set[bytes], int, str]:
    hashes = set()
    file_hash = hashlib.sha256()
    count = 0
    with path.open("rb") as handle:
        for line in handle:
            file_hash.update(line)
            text = json.loads(line)["text"]
            if not isinstance(text, str) or not text:
                raise ValueError(f"invalid C4 text in {path}")
            hashes.add(hashlib.sha256(text.encode("utf-8")).digest())
            count += 1
    return hashes, count, file_hash.hexdigest()


def prepare_disjoint_shard(train_path: Path, heldout_path: Path,
                           download: bool = True) -> dict:
    """Verify distinct source files and exact text-document nonoverlap.

    If two C4 shards share exact documents, write a filtered held-out JSON and
    use only that file for collection. Training files are never modified.
    """
    train_path = train_path.expanduser().resolve()
    heldout_path = heldout_path.expanduser().resolve()
    if train_path == heldout_path or not train_path.is_file():
        raise ValueError("train and held-out C4 paths must be different existing files")
    gz = heldout_path.parent / "en" / SHARD
    if not heldout_path.is_file():
        if not gz.is_file():
            if not download:
                raise FileNotFoundError(heldout_path)
            from huggingface_hub import hf_hub_download
            gz = Path(hf_hub_download(
                repo_id="allenai/c4", repo_type="dataset",
                filename=SHARD_REPO_PATH, revision=SHARD_REVISION,
                local_dir=str(heldout_path.parent)))
        if sha256_file(gz) != SHARD_SHA256:
            raise ValueError("C4 held-out compressed shard SHA-256 mismatch")
        temporary = heldout_path.with_suffix(".json.tmp")
        with gzip.open(gz, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        temporary.replace(heldout_path)
    if gz.is_file() and sha256_file(gz) != SHARD_SHA256:
        raise ValueError("C4 held-out compressed shard SHA-256 mismatch")
    train_hashes, train_docs, train_sha = _document_hashes(train_path)
    held_hashes, held_docs, held_sha = _document_hashes(heldout_path)
    overlaps = train_hashes & held_hashes
    selected_path = heldout_path
    if overlaps:
        selected_path = heldout_path.with_name(heldout_path.stem + ".filtered.json")
        with heldout_path.open("rb") as source, selected_path.open("wb") as target:
            for line in source:
                text = json.loads(line)["text"]
                if hashlib.sha256(text.encode("utf-8")).digest() not in overlaps:
                    target.write(line)
    selected_hashes, selected_docs, selected_sha = _document_hashes(selected_path)
    if train_hashes & selected_hashes:
        raise AssertionError("train and held-out documents overlap after filtering")
    return {"train_file": str(train_path), "train_sha256": train_sha,
            "train_documents": train_docs, "heldout_file": str(heldout_path),
            "heldout_sha256": held_sha, "heldout_documents": held_docs,
            "heldout_compressed_file": str(gz) if gz.is_file() else None,
            "heldout_compressed_sha256": sha256_file(gz) if gz.is_file() else None,
            "heldout_revision": SHARD_REVISION if gz.is_file() else None,
            "exact_text_overlap_documents": len(overlaps),
            "selected_heldout_file": str(selected_path),
            "selected_heldout_sha256": selected_sha,
            "selected_heldout_documents": selected_docs,
            "verified_disjoint": True}


def collect_heldout(args) -> dict:
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    provenance = prepare_disjoint_shard(args.train_data, args.heldout_data)
    layers = selected(args.layers, 24)
    cache = args.heldout_cache.expanduser().resolve()
    train_cache = args.train_cache.expanduser().resolve()
    if cache == train_cache or cache == train_cache.parent:
        raise ValueError("held-out cache must be separate from train calibration cache")
    if any((cache / f"layer_{layer}.pt").exists() for layer in layers):
        raise FileExistsError(f"held-out cache already exists: {cache}")
    settings = SimpleNamespace(
        calibration_data=Path(provenance["selected_heldout_file"]),
        gpu_memory_fraction=args.gpu_memory_fraction,
        calibration_seqlen=args.calibration_seqlen,
        calibration_blocks=args.calibration_blocks,
        calibration_batch_size=args.calibration_batch_size,
        calibration_samples=args.calibration_samples,
        seed=args.seed)
    collect_calibration(settings, args.model_path.expanduser().resolve(), layers, cache)
    config = json.loads((args.model_path / "config.json").read_text())
    experts = list(range(int(config["num_experts"])))
    counts = {}
    for layer in layers:
        samples, frequency, source = load_calibration_layer(cache, layer, experts)
        if source != "full_routing_count":
            raise AssertionError("held-out routing frequency is not a full count")
        minimum = min(samples[i]["gate"].shape[0] for i in experts)
        if minimum < args.min_heldout_samples:
            raise ValueError(f"layer {layer}: minimum held-out samples {minimum} < "
                             f"{args.min_heldout_samples}")
        counts[str(layer)] = {"minimum_samples": minimum,
                              "maximum_samples": max(samples[i]["gate"].shape[0]
                                                     for i in experts),
                              "minimum_routes": min(frequency.values()),
                              "maximum_routes": max(frequency.values())}
    provenance.update({"seed": args.seed, "layers": layers,
                       "expert_count": len(experts),
                       "calibration_blocks": args.calibration_blocks,
                       "calibration_seqlen": args.calibration_seqlen,
                       "calibration_batch_size": args.calibration_batch_size,
                       "sample_cap": args.calibration_samples,
                       "heldout_cache": str(cache), "routed_counts": counts})
    (out / "data_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def ensure_heldout_inputs(args) -> None:
    """Make evaluate a single command after compress; keep incomplete caches."""
    layers = selected(args.layers, 24)
    checkpoints = [args.checkpoint_root / f"shared_{sr}" / f"residual_{rr}" /
                   f"layer_{layer}.pt" for layer in layers
                   for sr in args.shared_ranks for rr in args.residual_ranks]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"compress must finish before held-out evaluation; "
                                f"missing {len(missing)} checkpoint files, first: {missing[0]}")
    out = args.output_dir.expanduser().resolve()
    cache = args.heldout_cache.expanduser().resolve()
    train_cache = args.train_cache.expanduser().resolve()
    if cache == train_cache or cache == train_cache.parent:
        raise ValueError("held-out cache must be separate from train calibration cache")
    provenance_path = (args.provenance.expanduser().resolve() if args.provenance
                       else out / "data_provenance.json")
    if args.provenance and not provenance_path.is_file():
        raise FileNotFoundError(f"explicit provenance file is missing: {provenance_path}")
    existing = [cache / f"layer_{layer}.pt" for layer in layers]
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    compatible = (provenance is not None and provenance.get("verified_disjoint") is True
                  and all(path.is_file() for path in existing))
    if compatible and not args.provenance:
        compatible = (
            set(layers).issubset(provenance.get("layers", []))
            and provenance.get("heldout_cache") == str(cache)
            and provenance.get("train_file") == str(args.train_data.expanduser().resolve())
            and provenance.get("heldout_file") == str(args.heldout_data.expanduser().resolve())
            and provenance.get("seed") == args.seed
            and provenance.get("calibration_blocks") == args.calibration_blocks
            and provenance.get("calibration_seqlen") == args.calibration_seqlen
            and provenance.get("calibration_batch_size") == args.calibration_batch_size
            and provenance.get("sample_cap") == args.calibration_samples)
    if compatible:
        print("[grouping-gsmoe-heldout] reusing routed activation cache", flush=True)
        return
    if args.provenance:
        raise ValueError("explicit provenance and held-out cache do not match requested layers")
    if provenance_path.is_file():
        _archive_stale(provenance_path)
    for path in existing:
        if path.is_file():
            _archive_stale(path)
    print("[grouping-gsmoe-heldout] collecting separate C4 routed activations", flush=True)
    collect_heldout(args)


def _rank_list(value: str, allow_zero: bool = False) -> list[int]:
    ranks = sorted({int(item.strip()) for item in value.split(",")})
    if not ranks or any(rank < (0 if allow_zero else 1) for rank in ranks):
        raise ValueError(f"invalid ranks: {value}")
    return ranks


def _train_reference(path: Path) -> tuple[dict, dict]:
    projections = {}
    with (path / "per_projection.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["expert"]), row["projection"],
                   int(row["shared_rank"]), int(row["residual_rank"]))
            if key in projections:
                raise ValueError(f"duplicate train projection row: {key}")
            projections[key] = row
    whole = {}
    with (path / "per_expert_swiglu.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["layer"]), int(row["expert"]),
                   int(row["shared_rank"]), int(row["residual_rank"]))
            if key in whole:
                raise ValueError(f"duplicate train SwiGLU row: {key}")
            whole[key] = row
    return projections, whole


def _summarize(rows: list[dict], projection: bool) -> list[dict]:
    fields = ("split", "layer", "projection", "shared_rank", "residual_rank") if projection else (
        "split", "layer", "shared_rank", "residual_rank")
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in fields)].append(row)
    result = []
    for key, members in sorted(buckets.items()):
        ref2 = sum(row["reference_squared_norm"] for row in members)
        err2 = sum(row["error_squared_norm"] for row in members)
        out2 = sum(row["output_norm_squared_ratio"] * row["reference_squared_norm"]
                   for row in members)
        result.append({**dict(zip(fields, key)), "expert_count": len(members),
                       "mean_relative_l2": sum(row["relative_l2"] for row in members)
                       / len(members),
                       "pooled_relative_l2": math.sqrt(err2 / ref2),
                       "mean_cosine": sum(row["cosine"] for row in members) / len(members),
                       "pooled_output_norm_squared_ratio": out2 / ref2,
                       "error_squared_norm": err2,
                       "reference_squared_norm": ref2})
    return result


def _paired_summaries(projection_summary: list[dict], swiglu_summary: list[dict]) -> list[dict]:
    paired = []
    for kind, rows in (("projection", projection_summary), ("swiglu", swiglu_summary)):
        fields = ("layer", "projection", "shared_rank", "residual_rank") if kind == "projection" else (
            "layer", "shared_rank", "residual_rank")
        buckets = defaultdict(dict)
        for row in rows:
            buckets[tuple(row[field] for field in fields)][row["split"]] = row
        for key, splits in sorted(buckets.items()):
            if set(splits) != {"train", "heldout"}:
                raise ValueError(f"missing train or held-out summary: {key}")
            train, heldout = splits["train"], splits["heldout"]
            paired.append({"kind": kind, **dict(zip(fields, key)),
                           "projection": key[1] if kind == "projection" else "swiglu",
                           "train_expert_count": train["expert_count"],
                           "heldout_expert_count": heldout["expert_count"],
                           "train_mean_relative_l2": train["mean_relative_l2"],
                           "heldout_mean_relative_l2": heldout["mean_relative_l2"],
                           "train_pooled_relative_l2": train["pooled_relative_l2"],
                           "heldout_pooled_relative_l2": heldout["pooled_relative_l2"],
                           "pooled_l2_delta": heldout["pooled_relative_l2"] -
                           train["pooled_relative_l2"],
                           "pooled_l2_ratio": heldout["pooled_relative_l2"] /
                           train["pooled_relative_l2"],
                           "train_mean_cosine": train["mean_cosine"],
                           "heldout_mean_cosine": heldout["mean_cosine"]})
    return paired


def _pooled(rows: list[dict]) -> tuple[float, float, int]:
    if not rows:
        raise ValueError("no rows for pooled diagnostic")
    err2 = sum(row["error_squared_norm"] for row in rows)
    ref2 = sum(row["reference_squared_norm"] for row in rows)
    return math.sqrt(err2 / ref2), sum(row["cosine"] for row in rows) / len(rows), len(rows)


def _report(args, provenance: dict, projection_rows: list[dict],
            whole_rows: list[dict], max_diff: dict) -> str:
    layers_text = ", ".join(str(layer) for layer in args.layers)
    focus_shared = 256 if 256 in args.shared_ranks else max(args.shared_ranks)
    focus_residuals = ([rank for rank in (0, 64, 256) if rank in args.residual_ranks]
                       if focus_shared == 256 else args.residual_ranks)
    lines = [f"# Held-out activation reconstruction: layers {layers_text}", "",
             "The same saved BF16 checkpoint factors are evaluated on train and "
             "held-out routed inputs. No grouping, projection, coefficient, or SVD "
             "is fitted with held-out data. QJL and lm-eval are not run.", "",
             f"Train source: `{provenance['train_file']}`. Held-out source: "
             f"`{provenance['selected_heldout_file']}`. Exact text-document overlap "
             f"across the complete source shards: {provenance['exact_text_overlap_documents']}. "
             f"Seed: {provenance['seed']}; held-out sample cap: {provenance['sample_cap']}.",
             f"Maximum absolute train-result reproduction differences: "
             f"projection L2 {max_diff['projection_l2']:.3g}, "
             f"SwiGLU L2 {max_diff['swiglu_l2']:.3g}.", "",
             "Pooled relative L2 is sqrt(sum squared output error / sum squared original "
             "output norm). Each table entry uses the original BF16 expert outputs as "
             "reference and the saved BF16 checkpoint factors for reconstruction.", "",
             f"## Layer-wise full SwiGLU output: shared rank {focus_shared}", "",
             "| Layer | Residual rank | Train pooled L2 | Held-out pooled L2 | "
             "Train mean cosine | Held-out mean cosine |",
             "|---:|---:|---:|---:|---:|---:|"]
    for layer in args.layers:
        for rr in focus_residuals:
            train = [r for r in whole_rows if r["layer"] == layer and
                     r["shared_rank"] == focus_shared and r["residual_rank"] == rr and
                     r["split"] == "train"]
            held = [r for r in whole_rows if r["layer"] == layer and
                    r["shared_rank"] == focus_shared and r["residual_rank"] == rr and
                    r["split"] == "heldout"]
            t_l2, t_cos, _ = _pooled(train)
            h_l2, h_cos, _ = _pooled(held)
            lines.append(f"| {layer} | {rr} | {t_l2:.5f} | {h_l2:.5f} | "
                         f"{t_cos:.5f} | {h_cos:.5f} |")
    lines += ["", f"## Gate/Up/Down output: shared rank {focus_shared}, by layer", "",
              "| Layer | Projection | Residual rank | Train pooled L2 | Held-out pooled L2 |",
              "|---:|---|---:|---:|---:|"]
    for layer in args.layers:
        for key in PROJECTIONS:
            for rr in focus_residuals:
                train = [r for r in projection_rows if r["layer"] == layer and
                         r["projection"] == key and r["shared_rank"] == focus_shared and
                         r["residual_rank"] == rr and r["split"] == "train"]
                held = [r for r in projection_rows if r["layer"] == layer and
                        r["projection"] == key and r["shared_rank"] == focus_shared and
                        r["residual_rank"] == rr and r["split"] == "heldout"]
                t_l2, _, _ = _pooled(train)
                h_l2, _, _ = _pooled(held)
                lines.append(f"| {layer} | {key} | {rr} | {t_l2:.5f} | {h_l2:.5f} |")
    if 256 in args.shared_ranks and all(rr in args.residual_ranks for rr in (0, 256)):
        lines += ["", "## Interpretation", ""]
        train_gate = [r for r in projection_rows if r["split"] == "train" and
                      r["projection"] == "gate" and r["shared_rank"] == 256 and
                      r["residual_rank"] == 0]
        held_gate = [r for r in projection_rows if r["split"] == "heldout" and
                     r["projection"] == "gate" and r["shared_rank"] == 256 and
                     r["residual_rank"] == 0]
        train_up = [r for r in projection_rows if r["split"] == "train" and
                    r["projection"] == "up" and r["shared_rank"] == 256 and
                    r["residual_rank"] == 0]
        held_up = [r for r in projection_rows if r["split"] == "heldout" and
                   r["projection"] == "up" and r["shared_rank"] == 256 and
                   r["residual_rank"] == 0]
        tg, _, _ = _pooled(train_gate)
        hg, _, _ = _pooled(held_gate)
        tu, _, _ = _pooled(train_up)
        hu, _, _ = _pooled(held_up)
        lines.append(f"At shared rank 256 and residual rank 0, Gate pooled output L2 "
                     f"changes from {tg:.5f} to {hg:.5f}; Up from {tu:.5f} to {hu:.5f}.")
        train0 = [r for r in whole_rows if r["split"] == "train" and
                  r["shared_rank"] == 256 and r["residual_rank"] == 0]
        held0 = [r for r in whole_rows if r["split"] == "heldout" and
                 r["shared_rank"] == 256 and r["residual_rank"] == 0]
        train256 = [r for r in whole_rows if r["split"] == "train" and
                    r["shared_rank"] == 256 and r["residual_rank"] == 256]
        held256 = [r for r in whole_rows if r["split"] == "heldout" and
                   r["shared_rank"] == 256 and r["residual_rank"] == 256]
        t0, _, _ = _pooled(train0)
        h0, _, _ = _pooled(held0)
        t256, _, _ = _pooled(train256)
        h256, _, _ = _pooled(held256)
        lines.append(f"Full SwiGLU: residual rank 0→256 changes train pooled L2 "
                     f"{t0:.5f}→{t256:.5f} and held-out pooled L2 "
                     f"{h0:.5f}→{h256:.5f}.")
        if h0 > t0 * 1.5 and hg > tg * 1.5 and hu > tu * 1.5:
            lines.append("The small train Gate/Up errors do not persist on this "
                         "held-out C4 shard. The large train/held-out gaps across "
                         "Gate, Up, and full SwiGLU indicate calibration-specific "
                         "reconstruction and poor generalization of these saved "
                         "factors to the new routed inputs.")
        else:
            lines.append("The train/held-out error gap on this C4 shard is "
                         "limited; the saved factors retain routed-output "
                         "reconstruction on these new documents.")
        if h256 < h0:
            lines.append(f"Residual SVD improves held-out full SwiGLU error by "
                         f"{h0 - h256:.5f} absolute ({(h0 - h256) / h0:.1%} relative) "
                         "from residual rank 0 to 256.")
        else:
            lines.append("Residual SVD does not improve pooled held-out full "
                         "SwiGLU error at residual rank 256.")
    source_limit = (
        "The full-covariance train cache records a source SHA-256 matching shard "
        "00000, and the grouping metadata matches that cache. Individual train "
        "document IDs were not recorded. "
        if provenance.get("train_cache_source_verified") else
        "The legacy train cache did not record document IDs or a source checksum; "
        "its association with shard 00000 follows the original run's default. ")
    lines += ["", "## Limits", "",
              source_limit + "We verified exact disjointness of the complete "
              "source-shard text sets. "
              "C4 may still contain similar or overlapping passages across different "
              "documents. The comparisons cover only routed experts and samples from "
              "the selected layers, not full-model language evaluation.", ""]
    return "\n".join(lines)


def _verify_payload(payload: dict, grouping: dict, experts: list[int],
                    shared_rank: int, residual_rank: int,
                    hidden: int, intermediate: int) -> None:
    if payload["groups"] != grouping["groups"]:
        raise AssertionError("checkpoint groups differ from train grouping cache")
    validate_groups(payload["groups"], grouping["settings"]["experts"],
                    len(payload["groups"]))
    if sorted(int(i) for i in payload["experts"]) != grouping["settings"]["experts"]:
        raise AssertionError("checkpoint expert IDs differ from train grouping cache")
    for group_id, bases in payload["bases"].items():
        for key in PROJECTIONS:
            p = bases[key]
            expected = grouping["bases"][group_id][key][:, :shared_rank].to(torch.bfloat16)
            if p.dtype != torch.bfloat16 or p.shape != (hidden, shared_rank) or \
                    not torch.equal(p, expected):
                raise AssertionError(f"checkpoint shared {key} basis differs from train cache")
    for i in experts:
        for key, part in payload["experts"][i].items():
            expected_coeff = ((shared_rank, intermediate) if key == "down" else
                              (intermediate, shared_rank))
            if part["coeff"].dtype != torch.bfloat16 or part["coeff"].shape != expected_coeff:
                raise AssertionError("checkpoint coefficient shape or dtype mismatch")
            if residual_rank == 0 and ("u" in part or "v" in part):
                raise AssertionError("rank 0 checkpoint unexpectedly has residual factors")
            if residual_rank:
                m, n = ((hidden, intermediate) if key == "down" else
                        (intermediate, hidden))
                if (part["u"].dtype != torch.bfloat16 or
                        part["v"].dtype != torch.bfloat16 or
                        part["u"].shape != (m, residual_rank) or
                        part["v"].shape != (residual_rank, n)):
                    raise AssertionError("checkpoint residual factor shape or dtype mismatch")


def _record_metrics(layer: int, expert: int, shared_rank: int, residual_rank: int,
                    split: str, samples: dict, weights: dict, parts: dict,
                    bases: dict, device: torch.device) -> tuple[list[dict], dict]:
    x = samples["gate"].to(device=device, dtype=torch.bfloat16)
    middle = samples["down"].to(device=device, dtype=torch.bfloat16)
    if x.ndim != 2 or middle.ndim != 2 or x.shape[0] != middle.shape[0]:
        raise ValueError(f"unaligned routed activations layer {layer} expert {expert}")
    baseline = {"gate": F.linear(x, weights["gate"]),
                "up": F.linear(x, weights["up"]),
                "down": F.linear(middle, weights["down"])}
    predicted = {}
    rows = []
    for key in PROJECTIONS:
        routed = middle if key == "down" else x
        predicted[key] = _projected_output(routed, parts[key], bases[key], key)
        metrics = _metrics(predicted[key], baseline[key])
        rows.append({"split": split, "layer": layer, "expert": expert,
                     "projection": key, "shared_rank": shared_rank,
                     "residual_rank": residual_rank, "samples": routed.shape[0],
                     **metrics})
    original = F.linear(F.silu(baseline["gate"]) * baseline["up"], weights["down"])
    reconstructed = _projected_output(
        F.silu(predicted["gate"]) * predicted["up"],
        parts["down"], bases["down"], "down")
    whole = {"split": split, "layer": layer, "expert": expert,
             "shared_rank": shared_rank, "residual_rank": residual_rank,
             "samples": x.shape[0], **_metrics(reconstructed, original)}
    return rows, whole


@torch.no_grad()
def evaluate_saved_factors(args) -> dict:
    out = args.output_dir.expanduser().resolve()
    provenance_path = args.provenance or out / "data_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    train_source_verified = True
    if not provenance.get("verified_disjoint") or provenance["exact_text_overlap_documents"] < 0:
        raise ValueError("missing verified train/held-out data separation")
    train_cache = args.train_cache.expanduser().resolve()
    heldout_cache = args.heldout_cache.expanduser().resolve()
    if train_cache == heldout_cache or heldout_cache != Path(provenance["heldout_cache"]):
        raise ValueError("held-out cache identity mismatch")
    if sha256_file(Path(provenance["selected_heldout_file"])) != provenance[
            "selected_heldout_sha256"]:
        raise ValueError("held-out source file changed since collection")
    model_dir = args.model_path.expanduser().resolve()
    config = json.loads((model_dir / "config.json").read_text())
    layers = selected(args.layers, int(config["num_hidden_layers"]))
    args.layers = layers
    if not set(layers).issubset(provenance["layers"]):
        raise ValueError("held-out cache was not collected for every requested layer")
    experts = list(range(int(config["num_experts"])))
    if args.max_experts is not None:
        experts = experts[:args.max_experts]
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    paths = {(layer, sr, rr): args.checkpoint_root / f"shared_{sr}" /
             f"residual_{rr}" / f"layer_{layer}.pt"
             for layer in layers for sr in args.shared_ranks for rr in args.residual_ranks}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} checkpoint layer files missing: {missing[:3]}")
    reference_projection, reference_swiglu = _train_reference(args.train_results)
    device = torch.device(args.device if args.device != "auto" else
                          "cuda" if torch.cuda.is_available() else "cpu")
    projection_rows, whole_rows = [], []
    max_diff = {"projection_l2": 0.0, "projection_cosine": 0.0,
                "projection_norm_ratio": 0.0, "swiglu_l2": 0.0,
                "swiglu_cosine": 0.0}
    integrity = {"checkpoint_sha256": {}, "cache_sha256": {}}
    reader = WeightReader(model_dir)
    try:
        for layer in layers:
            train, _, train_source = load_calibration_layer(train_cache, layer, experts)
            heldout, _, held_source = load_calibration_layer(heldout_cache, layer, experts)
            if train_source != "full_routing_count" or held_source != "full_routing_count":
                raise ValueError("expected uncapped original-router routing counts")
            for i in experts:
                if heldout[i]["gate"].shape[0] < args.min_heldout_samples:
                    raise ValueError(f"held-out expert {layer}/{i} has too few samples")
                if train[i]["gate"].shape[1] != hidden or heldout[i]["gate"].shape[1] != hidden:
                    raise ValueError("Gate/Up routed input shape mismatch")
                if train[i]["down"].shape[1] != intermediate or \
                        heldout[i]["down"].shape[1] != intermediate:
                    raise ValueError("Down routed input shape mismatch")
            grouping_path = args.grouping_cache / f"layer_{layer}.pt"
            cache_paths = [train_cache / f"layer_{layer}.pt",
                           heldout_cache / f"layer_{layer}.pt", grouping_path]
            cache_before = {str(path): sha256_file(path) for path in cache_paths}
            grouping = torch.load(grouping_path, map_location="cpu", weights_only=False)
            train_sidecar = train_cache / f"layer_{layer}.meta.json"
            if train_sidecar.is_file():
                train_metadata = json.loads(train_sidecar.read_text())["metadata"]
                if train_metadata.get("dataset_sha256") != provenance["train_sha256"]:
                    raise ValueError("train covariance cache and source shard hashes disagree")
                if grouping["settings"].get("calibration_metadata") != train_metadata:
                    raise ValueError("grouping was fitted from a different train covariance cache")
            elif "full_cov" in train_cache.parts:
                raise ValueError("full-covariance train cache lacks source provenance")
            else:
                train_source_verified = False
            grouped_experts = grouping["settings"]["experts"]
            validate_groups(grouping["groups"], grouped_experts,
                            len(grouping["groups"]))
            if not set(experts).issubset(grouped_experts):
                raise ValueError("requested experts are absent from checkpoint grouping")
            weights = {i: {key: reader.get(layer, i, key) for key in PROJECTIONS}
                       for i in experts}
            for sr in args.shared_ranks:
                coeff_reference = {}
                for rr in args.residual_ranks:
                    checkpoint = paths[(layer, sr, rr)]
                    before = sha256_file(checkpoint)
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    _verify_payload(payload, grouping, experts, sr, rr, hidden, intermediate)
                    if rr == args.residual_ranks[0]:
                        coeff_reference = {i: {key: payload["experts"][i][key]["coeff"]
                                               for key in PROJECTIONS} for i in experts}
                    else:
                        for i in experts:
                            for key in PROJECTIONS:
                                if not torch.equal(coeff_reference[i][key],
                                                   payload["experts"][i][key]["coeff"]):
                                    raise AssertionError("expert coefficient changed across residual ranks")
                    owner = {i: g for g, members in enumerate(payload["groups"])
                             for i in members}
                    for i in experts:
                        bases = payload["bases"][owner[i]]
                        parts = payload["experts"][i]
                        moved_weights = {key: weights[i][key].to(device)
                                         for key in PROJECTIONS}
                        for split, cache in (("train", train), ("heldout", heldout)):
                            rows, whole = _record_metrics(layer, i, sr, rr, split,
                                                          cache[i], moved_weights,
                                                          parts, bases, device)
                            projection_rows.extend(rows)
                            whole_rows.append(whole)
                            if split == "train":
                                for row in rows:
                                    ref = reference_projection[(layer, i, row["projection"], sr, rr)]
                                    max_diff["projection_l2"] = max(
                                        max_diff["projection_l2"],
                                        abs(row["relative_l2"] - float(ref["output_relative_l2"])))
                                    max_diff["projection_cosine"] = max(
                                        max_diff["projection_cosine"],
                                        abs(row["cosine"] - float(ref["output_cosine"])))
                                    max_diff["projection_norm_ratio"] = max(
                                        max_diff["projection_norm_ratio"],
                                        abs(row["output_norm_squared_ratio"] -
                                            float(ref["output_norm_squared_ratio"])))
                                ref = reference_swiglu[(layer, i, sr, rr)]
                                max_diff["swiglu_l2"] = max(
                                    max_diff["swiglu_l2"],
                                    abs(whole["relative_l2"] - float(ref["relative_l2"])))
                                max_diff["swiglu_cosine"] = max(
                                    max_diff["swiglu_cosine"],
                                    abs(whole["cosine"] - float(ref["cosine"])))
                    after = sha256_file(checkpoint)
                    if before != after:
                        raise AssertionError(f"checkpoint factor file changed: {checkpoint}")
                    integrity["checkpoint_sha256"][str(checkpoint)] = before
                    del payload
            cache_after = {str(path): sha256_file(path) for path in cache_paths}
            if cache_before != cache_after:
                raise AssertionError("train, held-out, or grouping cache changed during evaluation")
            integrity["cache_sha256"].update(cache_before)
            print(f"[grouping-gsmoe-heldout] layer {layer}: "
                  f"{len(experts)} experts, {len(args.shared_ranks) * len(args.residual_ranks)} "
                  "saved checkpoint combinations", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        reader.close()
    provenance["train_cache_source_verified"] = train_source_verified
    for name, difference in max_diff.items():
        if difference > args.train_reproduction_atol:
            raise AssertionError(f"train result not reproduced: {name} max diff {difference}")
    projection_summary = _summarize(projection_rows, projection=True)
    swiglu_summary = _summarize(whole_rows, projection=False)
    comparisons = _paired_summaries(projection_summary, swiglu_summary)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "per_projection.csv", projection_rows)
    _write_csv(out / "per_expert_swiglu.csv", whole_rows)
    _write_csv(out / "projection_summary.csv", projection_summary)
    _write_csv(out / "swiglu_summary.csv", swiglu_summary)
    _write_csv(out / "train_vs_heldout.csv", comparisons)
    (out / "factor_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (out / "REPORT.md").write_text(_report(args, provenance, projection_rows,
                                            whole_rows, max_diff))
    result = {"layers": layers, "experts": experts,
              "shared_ranks": args.shared_ranks, "residual_ranks": args.residual_ranks,
              "data_provenance": provenance, "train_reproduction_max_abs_diff": max_diff,
              "projection_summary": projection_summary,
              "swiglu_summary": swiglu_summary}
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT / "results/grouping_gsmoe/full_cov"
    parser.add_argument("--mode", choices=("collect", "evaluate"), required=True)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat"))
    parser.add_argument("--layers", default="0,12,23")
    parser.add_argument("--max-experts", type=int,
                        help="evaluation smoke only; collection always routes all experts")
    parser.add_argument("--shared-ranks", default="64,128,256")
    parser.add_argument("--residual-ranks", default="0,32,64,128,256")
    parser.add_argument("--train-data", type=Path,
                        default=ROOT / "rescode/data/c4-train.00000-of-01024.json")
    parser.add_argument("--heldout-data", type=Path,
                        default=root / "heldout_data/c4-train.00001-of-01024.json")
    parser.add_argument("--train-cache", type=Path, default=root / "calibration")
    parser.add_argument("--heldout-cache", type=Path,
                        default=root / "heldout_activation_cache_shard00001")
    parser.add_argument("--grouping-cache", type=Path, default=root / "grouping")
    parser.add_argument("--checkpoint-root", type=Path, default=root / "checkpoints")
    parser.add_argument("--train-results", type=Path,
                        default=root / "analysis")
    parser.add_argument("--output-dir", type=Path,
                        default=root / "heldout_layers_0_12_23")
    parser.add_argument("--provenance", type=Path,
                        help="collection provenance JSON; defaults to output-dir/data_provenance.json")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--calibration-blocks", type=int, default=32)
    parser.add_argument("--calibration-seqlen", type=int, default=2048)
    parser.add_argument("--calibration-batch-size", type=int, default=2)
    parser.add_argument("--calibration-samples", type=int, default=256)
    parser.add_argument("--min-heldout-samples", type=int, default=256)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.9)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-reproduction-atol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.shared_ranks = _rank_list(args.shared_ranks)
    args.residual_ranks = _rank_list(args.residual_ranks, allow_zero=True)
    if args.calibration_samples < args.min_heldout_samples or args.min_heldout_samples < 2:
        parser.error("sample cap must be at least min-heldout-samples >= 2")
    if args.calibration_blocks < 1 or args.calibration_seqlen < 1:
        parser.error("calibration blocks and sequence length must be positive")
    return args


def main(argv=None):
    args = arguments(argv)
    if args.mode == "collect":
        return collect_heldout(args)
    ensure_heldout_inputs(args)
    return evaluate_saved_factors(args)


if __name__ == "__main__":
    main()
