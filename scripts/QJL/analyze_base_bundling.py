#!/usr/bin/env python3
"""Activation-aware pair similarity, shared QJL Base, and residual analysis.

The script never evaluates a compressed model. Only `calibrate` runs original
model forward passes; later numeric stages read cached routed moments.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from rescode.qjl.base_bundle_analysis import (CASES, PROJECTIONS, PrefixProjectionBank, StreamingRoutedMoments,
    base_cases, base_metrics, covariance_factors, covariance_terms, make_expert_cache,
    pair_similarity, perfect_matching,
    psd_sqrt, spectrum_rows)
from rescode.qjl.core import ProjectionBank


def parse_ints(value):
    result = sorted(set(int(x) for x in value.split(",")))
    if not result or any(x < 0 for x in result):
        raise argparse.ArgumentTypeError("expected nonnegative comma-separated integers")
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True,
                   choices=("calibrate", "similarity", "base", "spectrum", "report", "all"))
    p.add_argument("--model-path", type=Path,
                   default=Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat"))
    p.add_argument("--calibration-jsonl", type=Path,
                   help="JSONL with a text field; disjoint from held-out JSONL")
    p.add_argument("--heldout-jsonl", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("results/base_bundling_analysis"))
    p.add_argument("--layers", type=parse_ints, default=[12])
    p.add_argument("--projections", default="gate,up,down")
    p.add_argument("--calibration-sizes", type=parse_ints, default=[128, 256, 512])
    p.add_argument("--heldout-size", type=int, default=128)
    p.add_argument("--input-ranks", type=parse_ints, default=[32, 64, 128, 256, 512])
    p.add_argument("--residual-ranks", type=parse_ints, default=[0, 32, 64, 128, 256, 352])
    p.add_argument("--qjl-d-factor", type=float, default=2.0)
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--bundle-seed", type=int, default=0)
    p.add_argument("--pairing-seed", type=int, default=0)
    p.add_argument("--svd-seed", type=int, default=0)
    p.add_argument("--svd-oversample", type=int, default=8)
    p.add_argument("--svd-niter", type=int, default=2)
    p.add_argument("--regularization", type=float, default=1e-5)
    p.add_argument("--probe-limit", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--spectrum-scope", choices=("representative", "all"), default="representative")
    p.add_argument("--representative-count", type=int, default=5)
    p.add_argument("--spectrum-cases", default=",".join(CASES))
    p.add_argument("--exact-representative", action="store_true")
    p.add_argument("--matching-rank", type=int, default=128)
    p.add_argument("--joint-lambda", type=float, default=1.0)
    p.add_argument("--gpu-memory-fraction", type=float, default=.9)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(argv)
    a.projections = a.projections.split(",")
    a.spectrum_cases = a.spectrum_cases.split(",")
    if any(x not in PROJECTIONS for x in a.projections) or any(x not in CASES for x in a.spectrum_cases):
        p.error("invalid projection or spectrum case")
    if min(a.calibration_sizes) < 1 or a.heldout_size < 1 or a.batch_size < 1 or a.max_length < 1:
        p.error("sequence sizes, batch size, and max length must be positive")
    if a.qjl_d_factor < 1 or a.regularization <= 0 or not 0 < a.gpu_memory_fraction <= 1:
        p.error("invalid projection factor, regularization, or GPU memory fraction")
    if a.stage in ("calibrate", "all") and (a.calibration_jsonl is None or a.heldout_jsonl is None):
        p.error("calibrate/all requires --calibration-jsonl and --heldout-jsonl")
    if a.exact_representative and a.spectrum_scope == "all":
        p.error("--exact-representative requires --spectrum-scope representative")
    if a.matching_rank not in a.residual_ranks:
        p.error("--matching-rank must appear in --residual-ranks")
    return a


def atomic_json(path, obj):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")
    os.replace(tmp, path)


def atomic_torch(path, obj):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def pair_key(row):
    return int(row["expert_i"]), int(row["expert_j"])


def completed_pairs(path, rows_per_pair, resume):
    if not resume:
        path.unlink(missing_ok=True)
        return set()
    rows = read_csv(path)
    if not rows:
        return set()
    counts = {}
    for row in rows:
        key = pair_key(row)
        counts[key] = counts.get(key, 0) + 1
    complete = {key for key, count in counts.items() if count == rows_per_pair}
    if len(complete) != len(counts):
        kept = [row for row in rows if pair_key(row) in complete]
        if kept:
            write_csv(path, kept)
        else:
            path.unlink(missing_ok=True)
    return complete


def append_csv(path, rows):
    if not rows:
        return
    if not path.exists():
        write_csv(path, rows)
        return
    with path.open(newline="") as handle:
        fields = next(csv.reader(handle))
    with path.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerows(rows)


def file_fingerprint(path):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path.resolve()), "sha256": h.hexdigest()}


def config_for(a, existing=None, reuse_inputs=False):
    cal_fingerprint = (existing["calibration_jsonl"] if existing and reuse_inputs and
                       a.calibration_jsonl is None else file_fingerprint(a.calibration_jsonl))
    held_fingerprint = (existing["heldout_jsonl"] if existing and reuse_inputs and
                        a.heldout_jsonl is None else file_fingerprint(a.heldout_jsonl))
    return {"model_path": str(a.model_path.resolve()),
            "calibration_jsonl": cal_fingerprint,
            "heldout_jsonl": held_fingerprint,
            "layers": a.layers, "projections": a.projections,
            "calibration_sizes": a.calibration_sizes, "heldout_size": a.heldout_size,
            "input_ranks": a.input_ranks, "residual_ranks": a.residual_ranks,
            "qjl_d_factor": a.qjl_d_factor, "projection_seed": a.projection_seed,
            "bundle_seed": a.bundle_seed, "pairing_seed": a.pairing_seed,
            "svd_seed": a.svd_seed, "svd_oversample": a.svd_oversample,
            "svd_niter": a.svd_niter, "regularization": a.regularization,
            "probe_limit": a.probe_limit, "max_length": a.max_length,
            "spectrum_scope": a.spectrum_scope, "representative_count": a.representative_count,
            "spectrum_cases": a.spectrum_cases, "exact_representative": a.exact_representative,
            "matching_rank": a.matching_rank, "joint_lambda": a.joint_lambda}


def prepare(a):
    a.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = a.output_dir / "config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        reuse_cal = a.calibration_jsonl is None
        reuse_held = a.heldout_jsonl is None
        config = config_for(a, old, reuse_inputs=True)
        if old != config:
            raise ValueError("output configuration differs; choose another --output-dir")
        if reuse_cal and old["calibration_jsonl"]:
            a.calibration_jsonl = Path(old["calibration_jsonl"]["path"])
        if reuse_held and old["heldout_jsonl"]:
            a.heldout_jsonl = Path(old["heldout_jsonl"]["path"])
    else:
        config = config_for(a)
        atomic_json(config_path, config)
    return config


def load_model(a, tokenizer=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory *
                             a.gpu_memory_fraction) for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    model = AutoModelForCausalLM.from_pretrained(str(a.model_path), torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True, max_memory=max_memory)
    model.eval()
    if tokenizer:
        tok = AutoTokenizer.from_pretrained(str(a.model_path), trust_remote_code=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        return model, tok
    return model


def text_batches(path, limit, batch_size, tokenizer, max_length, boundaries=(), offset=0):
    records = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item["text"]
            if not isinstance(text, str) or not text:
                raise ValueError("each JSONL record needs nonempty text")
            records.append(text)
            if len(records) == limit:
                break
    if len(records) < limit:
        raise ValueError(f"{path}: expected {limit} records, found {len(records)}")
    # Tokenize batches lazily. Forward boundaries follow the requested snapshots.
    start = offset
    while start < len(records):
        next_boundary = min((b for b in boundaries if b > start), default=limit)
        end = min(start + batch_size, next_boundary)
        yield tokenizer(records[start:end], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        start = end


def calibrate(a, model, tokenizer):
    if a.calibration_jsonl.resolve() == a.heldout_jsonl.resolve():
        raise ValueError("calibration and held-out files must differ")
    # File hash overlap detects copied datasets as well as identical paths.
    if file_fingerprint(a.calibration_jsonl)["sha256"] == file_fingerprint(a.heldout_jsonl)["sha256"]:
        raise ValueError("calibration and held-out file contents are identical")
    def text_keys(path, limit):
        keys = set()
        seen = 0
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    keys.add(hashlib.sha256(json.loads(line)["text"].encode()).digest())
                    seen += 1
                    if seen >= limit:
                        break
        return keys
    if text_keys(a.calibration_jsonl, max(a.calibration_sizes)) & text_keys(a.heldout_jsonl, a.heldout_size):
        raise ValueError("calibration and held-out text records overlap")
    cache = a.output_dir / "cache"
    cache.mkdir(exist_ok=True)
    for split, path, targets in (("calibration", a.calibration_jsonl, a.calibration_sizes),
                                 ("heldout", a.heldout_jsonl, [a.heldout_size])):
        outputs = [cache / f"{split}_{n}.pt" for n in targets]
        if a.resume and all(x.exists() for x in outputs):
            print(f"[calibrate] {split}: resumed", flush=True)
            continue
        collector = StreamingRoutedMoments(model, a.layers, a.probe_limit)
        completed = [n for n in targets if (cache / f"{split}_{n}.pt").exists()]
        count = max(completed) if a.resume and completed else 0
        if count:
            collector.restore(torch.load(cache / f"{split}_{count}.pt",
                                         map_location="cpu", weights_only=False))
        target = max(targets)
        try:
            with torch.no_grad():
                for batch in text_batches(path, target, a.batch_size, tokenizer,
                                          a.max_length, targets, offset=count):
                    device = next(model.parameters()).device
                    collector.begin_batch(batch["attention_mask"])
                    model(**{k: v.to(device) for k, v in batch.items()}, use_cache=False)
                    count += batch["input_ids"].shape[0]
                    if count in targets:
                        atomic_torch(cache / f"{split}_{count}.pt", collector.snapshot())
                    print(f"[calibrate] {split}: {count}/{target}", flush=True)
        finally:
            collector.close()
        if count != target:
            raise RuntimeError("sequence count does not equal target; adjust --batch-size")


def stage_file(a, stage, n, layer, projection):
    folder = a.output_dir / "cache" / stage
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"s{n}_l{layer}_{projection}.csv"


def merged_stage(a, stage, output):
    rows = []
    for n, layer, proj in itertools.product(a.calibration_sizes, a.layers, a.projections):
        path = stage_file(a, stage, n, layer, proj)
        if path.exists():
            rows.extend(read_csv(path))
    write_csv(a.output_dir / output, rows)
    return rows


def snapshot(a, n, split="calibration"):
    path = a.output_dir / "cache" / f"{split}_{n}.pt"
    if not path.exists():
        raise FileNotFoundError(f"run --stage calibrate first: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def expert_data(model, a, layer, projection, cal, bank, half):
    experts = model.model.layers[layer].mlp.experts
    analysis_device = getattr(experts[0], projection + "_proj").weight.device
    output = {}
    for i, expert in enumerate(experts):
        stat = cal["experts"].get(i, {}).get(projection)
        if stat is None or stat["count"] == 0:
            raise ValueError(f"layer {layer} expert {i} {projection} has no routed tokens")
        w = getattr(expert, projection + "_proj").weight.detach().to(analysis_device).float()
        c = stat["covariance"].to(analysis_device)
        output[i] = make_expert_cache(w, c, bank, half, a.input_ranks)
    return output


def banks_for(a, model, layer, projection):
    w = getattr(model.model.layers[layer].mlp.experts[0], projection + "_proj").weight
    n = w.shape[1]
    full = ProjectionBank(round(n * a.qjl_d_factor), a.projection_seed)
    return full, PrefixProjectionBank(full, n)


def iter_units(a, model):
    for n in a.calibration_sizes:
        cal = snapshot(a, n)
        held = snapshot(a, a.heldout_size, "heldout")
        for layer in a.layers:
            for projection in a.projections:
                bank, half = banks_for(a, model, layer, projection)
                # The caller builds and releases one expert cache before the
                # generator advances. Keeping it here overlaps two projections.
                yield n, layer, projection, cal[layer], held[layer], bank, half


def release_unit():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def similarity(a, model):
    for n, layer, projection, cal, held, bank, half in iter_units(a, model):
        path = stage_file(a, "similarity", n, layer, projection)
        completed = completed_pairs(path, 1, a.resume)
        expert_count = len(model.model.layers[layer].mlp.experts)
        if len(completed) == expert_count * (expert_count - 1) // 2:
            print(f"[similarity] {n} layer {layer} {projection}: resumed", flush=True)
            continue
        data = expert_data(model, a, layer, projection, cal, bank, half)
        counts = {i: cal["experts"][i][projection]["count"] for i in data}
        g = bank.get(next(iter(data.values())).w.shape[1], next(iter(data.values())).w.device).float()
        probes = {i: cal["experts"][i][projection]["probe"].to(data[i].w.device).float()
                  for i in data}
        hidden_probes = {i: cal["experts"][i]["gate"]["probe"].to(data[i].w.device).float()
                         for i in data} if projection == "down" else {}
        original_cache = {}
        experts = model.model.layers[layer].mlp.experts
        def ffn(target, x):
            key = (target, x.data_ptr())
            if key not in original_cache:
                # Probe-only functional comparison. Calibration forward is never replayed.
                weight = experts[target].gate_proj.weight
                original_cache[key] = experts[target](x.to(device=weight.device,
                                                             dtype=weight.dtype)).float().to(x.device)
            return original_cache[key]
        for i, j in itertools.combinations(data, 2):
            if (i, j) in completed:
                continue
            probe_i = probes[i]
            probe_j = probes[j]
            row = pair_similarity(i, j, data[i], data[j], int(cal["coroute"][i, j]),
                                  counts, a.input_ranks, probe_i, probe_j,
                                  ffn if projection != "down" else None, g)
            if projection == "down":
                for direction, src in (("i_to_j", i), ("j_to_i", j)):
                    x = hidden_probes[src]
                    row[f"ffn_cosine_{direction}"] = float(F.cosine_similarity(
                        ffn(i, x).flatten().unsqueeze(0), ffn(j, x).flatten().unsqueeze(0)).item())
            row["functional_projection_cosine"] = float(np.mean([
                row["projection_cosine_i_to_j"], row["projection_cosine_j_to_i"]]))
            row["functional_ffn_cosine"] = float(np.mean([
                row["ffn_cosine_i_to_j"], row["ffn_cosine_j_to_i"]]))
            row["functional_cosine"] = float(np.mean([
                row["functional_projection_cosine"], row["functional_ffn_cosine"]]))
            row.update({"layer": layer, "projection": projection, "calibration_size": n})
            append_csv(path, [row])
        print(f"[similarity] {n} layer {layer} {projection}: {len(read_csv(path))} pairs", flush=True)
        del data, probes, hidden_probes, original_cache, ffn, g, counts
        release_unit()
    merged_stage(a, "similarity", "pairwise_metrics.csv")


def storage_estimate(case, output_dim, input_dim, d):
    matrices = 2 * output_dim * input_dim
    if case in ("A", "B"):
        sketch_bytes, scale_bytes = 0, 0
        dense_bytes = 2 * output_dim * input_dim  # one BF16 shared matrix
    else:
        sketches = 2 if case.startswith("C") else 1
        bits_per_row = input_dim if case == "C1" else d
        sketch_bytes = sketches * output_dim * math.ceil(bits_per_row / 8)
        scale_bytes = 2 * output_dim * 4  # FP32 expert-specific row scales
        dense_bytes = 0
    return {"effective_bpw": 8 * (sketch_bytes + scale_bytes + dense_bytes) / matrices,
            "shared_sketch_bytes": sketch_bytes if not case.startswith("C") else 0,
            "individual_sketch_bytes": sketch_bytes if case.startswith("C") else 0,
            "row_scale_bytes": scale_bytes, "dense_shared_bf16_bytes": dense_bytes,
            "runtime_projection_bytes": 0 if case in ("A", "B") else
                                        2 * input_dim * (input_dim if case == "C1" else d),
            "storage_assumption": "BF16 dense shared" if case in ("A", "B") else
                                  "packed signs plus FP32 row scales; projection amortized globally"}


@torch.no_grad()
def base(a, model):
    for n, layer, projection, cal, held, bank, half in iter_units(a, model):
        path = stage_file(a, "base", n, layer, projection)
        completed = completed_pairs(path, 2 * len(CASES), a.resume)
        expert_count = len(model.model.layers[layer].mlp.experts)
        if len(completed) == expert_count * (expert_count - 1) // 2:
            print(f"[base] {n} layer {layer} {projection}: resumed", flush=True)
            continue
        data = expert_data(model, a, layer, projection, cal, bank, half)
        cal_terms = {expert_id: (e.wc, (e.wc * e.w).sum().clamp_min(1e-30))
                     for expert_id, e in data.items()}
        held_terms = {}
        for expert_id, e in data.items():
            h = held["experts"][expert_id][projection]["covariance"].to(e.w.device)
            held_terms[expert_id] = covariance_terms(e.w, h)
        started = time.perf_counter()
        processed = 0
        for i, j in itertools.combinations(data, 2):
            if (i, j) in completed:
                continue
            pair_rows = []
            cases = base_cases(data[i], data[j], bank, half,
                               a.bundle_seed + layer * 1000003 + i * 65537 + j,
                               a.regularization)
            for case, (bi, bj, stats) in cases.items():
                for expert_id, b, e in ((i, bi, data[i]), (j, bj, data[j])):
                    h = held["experts"][expert_id][projection]["covariance"].to(e.w.device)
                    probe = cal["experts"][expert_id][projection]["probe"]
                    metric = base_metrics(e.w, b, e.c, h,
                                          probe.to(e.w.device).float() if probe is not None else None,
                                          cal_terms=cal_terms[expert_id],
                                          held_terms=held_terms[expert_id])
                    metric.update({"layer": layer, "projection": projection, "calibration_size": n,
                                   "expert_i": i, "expert_j": j, "expert_id": expert_id,
                                   "case": case,
                                   "sharing": "strict_dense" if case in ("A", "B") else
                                              "individual_binary" if case.startswith("C") else "shared_binary_row_scale",
                                   "regularization": a.regularization if case == "B" else 0,
                                   "bit_preservation": stats.get("bit_preservation_i" if expert_id == i else "bit_preservation_j", math.nan)})
                    metric.update(storage_estimate(case, *e.w.shape, bank.d))
                    pair_rows.append(metric)
            append_csv(path, pair_rows)
            processed += 1
            if processed % 100 == 0:
                rate = processed / max(time.perf_counter() - started, 1e-6)
                remaining = (expert_count * (expert_count - 1) // 2 -
                             len(completed) - processed) / max(rate, 1e-6)
                print(f"[base] {n} layer {layer} {projection}: "
                      f"{len(completed) + processed}/{expert_count * (expert_count - 1) // 2} "
                      f"pairs, ETA {remaining / 60:.1f} min", flush=True)
        print(f"[base] {n} layer {layer} {projection}: {len(read_csv(path))} expert cases", flush=True)
        del data, cal_terms, held_terms
        if "cases" in locals():
            del cases, pair_rows, bi, bj, b, e, h
        release_unit()
    merged_stage(a, "base", "base_metrics.csv")


def representative_pairs(a, sim_rows, base_rows, n_experts):
    if a.spectrum_scope == "all":
        return {tuple(pair): "all" for pair in itertools.combinations(range(n_experts), 2)}
    chosen = {}
    def select(rows, key, label):
        sorted_rows = sorted(rows, key=lambda r: float(r[key]))
        for r in sorted_rows[:a.representative_count]:
            chosen[(int(r["expert_i"]), int(r["expert_j"]))] = f"{label}_bottom"
        for r in sorted_rows[-a.representative_count:]:
            chosen[(int(r["expert_i"]), int(r["expert_j"]))] = f"{label}_top"
    for key in ("raw_cosine", "subspace_overlap_r" + str(a.input_ranks[-1]),
                "activation_cosine", "bit_agreement", "functional_cosine"):
        select(sim_rows, key, key)
    pair_base = {}
    for r in base_rows:
        if r["case"] == "F":
            key = (int(r["expert_i"]), int(r["expert_j"]))
            pair_base.setdefault(key, []).append(float(r["calibration_error"]))
    select([{"expert_i": i, "expert_j": j, "base_error": np.mean(vals)}
            for (i, j), vals in pair_base.items()], "base_error", "base_error")
    random_pairs = perfect_matching(sim_rows, "random", n_experts=n_experts, seed=a.pairing_seed)
    for pair in random_pairs:
        chosen.setdefault(pair, "random_feasibility_control")
    return chosen


@torch.no_grad()
def spectrum(a, model):
    spectra_dir = a.output_dir / "spectra"
    spectra_dir.mkdir(exist_ok=True)
    for n, layer, projection, cal, held, bank, half in iter_units(a, model):
        path = stage_file(a, "spectrum", n, layer, projection)
        methods = ["randomized", "exact"] if a.exact_representative else ["randomized"]
        completed = completed_pairs(path, 2 * len(a.spectrum_cases) * 2 *
                                    len(a.residual_ranks) * len(methods), a.resume)
        sim = read_csv(stage_file(a, "similarity", n, layer, projection))
        bas = read_csv(stage_file(a, "base", n, layer, projection))
        if not sim or not bas:
            raise FileNotFoundError("run similarity and base before spectrum")
        expert_count = len(model.model.layers[layer].mlp.experts)
        pairs = representative_pairs(a, sim, bas, expert_count)
        if set(pairs).issubset(completed):
            print(f"[spectrum] {n} layer {layer} {projection}: resumed", flush=True)
            continue
        data = expert_data(model, a, layer, projection, cal, bank, half)
        factors = {expert_id: covariance_factors(e.c, a.regularization,
                                                 e.eigenvalues, e.eigenvectors)
                   for expert_id, e in data.items()}
        weight_energies = {}
        for expert_id, e in data.items():
            h = held["experts"][expert_id][projection]["covariance"].to(e.w.device)
            weight_energies[expert_id] = ((e.wc * e.w).sum().clamp_min(1e-30),
                                          covariance_terms(e.w, h)[1])
        individual_spectra = {}
        started = time.perf_counter()
        processed = 0
        for (i, j), rule in pairs.items():
            if (i, j) in completed:
                continue
            pair_rows = []
            cases = base_cases(data[i], data[j], bank, half,
                               a.bundle_seed + layer * 1000003 + i * 65537 + j,
                               a.regularization)
            for case in a.spectrum_cases:
                bi, bj, _ = cases[case]
                for expert_id, b, e in ((i, bi, data[i]), (j, bj, data[j])):
                    h = held["experts"][expert_id][projection]["covariance"].to(e.w.device)
                    for method in methods:
                        cache_key = (case, expert_id, method)
                        if case.startswith("C") and cache_key in individual_spectra:
                            outputs = individual_spectra[cache_key]
                        else:
                            outputs = spectrum_rows(e.w, b, e.c, h, a.residual_ranks,
                                                    a.svd_seed + expert_id + (0 if case.startswith("C")
                                                                               else i * 65537 + j),
                                                    method=method, oversample=a.svd_oversample,
                                                    ridge=a.regularization,
                                                    covariance_root=factors[expert_id][0],
                                                    covariance_inverse_root=factors[expert_id][1],
                                                    calibration_weight_energy=weight_energies[expert_id][0],
                                                    heldout_weight_energy=weight_energies[expert_id][1])
                            for out in outputs:
                                singular = out.pop("singular_values")
                                if singular is not None:
                                    stem = (f"individual_e{expert_id}" if case.startswith("C")
                                            else f"p{i}-{j}_e{expert_id}")
                                    filename = f"s{n}_l{layer}_{projection}_{stem}_{case}_{out['space']}_{method}.npz"
                                    values = np.asarray(singular, dtype=np.float32)
                                    energy = np.cumsum(values.astype(np.float64) ** 2)
                                    energy /= max(float((e.w - b).square().sum()), 1e-30) if out["space"] == "weight" else max(float(((e.w - b) @ factors[expert_id][0]).square().sum()), 1e-30)
                                    np.savez_compressed(spectra_dir / filename, singular_values=values,
                                                        cumulative_energy=energy)
                                    out["spectrum_file"] = filename
                            if case.startswith("C"):
                                individual_spectra[cache_key] = outputs
                        for out in outputs:
                            record = dict(out)
                            record.update({"layer": layer, "projection": projection,
                                           "calibration_size": n, "expert_i": i, "expert_j": j,
                                           "expert_id": expert_id, "case": case, "selection_rule": rule})
                            pair_rows.append(record)
            append_csv(path, pair_rows)
            processed += 1
            if processed % 5 == 0:
                remaining = (len(pairs) - len(completed.intersection(pairs)) - processed) * \
                            (time.perf_counter() - started) / processed
                print(f"[spectrum] {n} layer {layer} {projection}: "
                      f"{len(completed.intersection(pairs)) + processed}/{len(pairs)} "
                      f"pairs, ETA {remaining / 60:.1f} min", flush=True)
        print(f"[spectrum] {n} layer {layer} {projection}: {len(read_csv(path))} rank records", flush=True)
        del data, factors, weight_energies, individual_spectra
        if "cases" in locals():
            del cases, pair_rows, outputs, bi, bj, b, e, h
        release_unit()
    merged_stage(a, "spectrum", "residual_metrics.csv")


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    out = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2
        start = end
    return out


def correlations(a, similarity_rows, base_rows, residual_rows):
    lookup = {}
    for r in base_rows:
        key = (r["calibration_size"], r["layer"], r["projection"], r["expert_i"], r["expert_j"], r["case"])
        lookup.setdefault(key, []).append(r)
    residual_lookup = {}
    for r in residual_rows:
        if r["method"] == "randomized" and r["space"] == "weight":
            key = (r["calibration_size"], r["layer"], r["projection"], r["expert_i"], r["expert_j"], r["case"], r["rank"])
            residual_lookup.setdefault(key, []).append(r)
    rows = []
    for n, layer, proj in itertools.product(a.calibration_sizes, a.layers, a.projections):
        sims = [r for r in similarity_rows if (int(r["calibration_size"]), int(r["layer"]), r["projection"]) == (n, layer, proj)]
        if not sims:
            continue
        sim_keys = [k for k in sims[0] if "cosine" in k or "overlap" in k or
                    k in ("bit_agreement", "conditioned_bit_agreement",
                          "theory_bit_agreement", "corouting_jaccard")]
        for case in CASES:
            for outcome in ("calibration_error", "heldout_error", "weight_error"):
                for metric in sim_keys:
                    x, y = [], []
                    for s in sims:
                        key = (str(n), str(layer), proj, s["expert_i"], s["expert_j"], case)
                        b = lookup.get(key)
                        if b:
                            sx, sy = float(s[metric]), float(np.mean([float(z[outcome]) for z in b]))
                            if math.isfinite(sx) and math.isfinite(sy):
                                x.append(sx); y.append(sy)
                    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
                        continue
                    rows.append({"layer": layer, "projection": proj, "calibration_size": n,
                                 "case": case, "similarity": metric, "outcome": outcome,
                                 "pearson": float(np.corrcoef(x, y)[0, 1]),
                                 "spearman": float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
                                 "pair_count": len(x)})
            for rank in a.residual_ranks:
                for outcome in ("residual_norm_ratio", "estimated_energy_retention", "estimated_normalized_tail"):
                    for metric in sim_keys:
                        x, y = [], []
                        for s in sims:
                            key = (str(n), str(layer), proj, s["expert_i"], s["expert_j"], case, str(rank))
                            r = residual_lookup.get(key)
                            if r:
                                sx, sy = float(s[metric]), float(np.mean([float(z[outcome]) for z in r]))
                                if math.isfinite(sx) and math.isfinite(sy):
                                    x.append(sx); y.append(sy)
                        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
                            continue
                        rows.append({"layer": layer, "projection": proj, "calibration_size": n,
                                     "case": case, "similarity": metric, "outcome": f"{outcome}_r{rank}",
                                     "pearson": float(np.corrcoef(x, y)[0, 1]),
                                     "spearman": float(np.corrcoef(ranks(x), ranks(y))[0, 1]),
                                     "pair_count": len(x)})
    return rows


def grouping(a, similarity_rows, base_rows, residual_rows):
    output = []
    for n, layer, proj in itertools.product(a.calibration_sizes, a.layers, a.projections):
        sim = [r for r in similarity_rows if (int(r["calibration_size"]), int(r["layer"]), r["projection"]) == (n, layer, proj)]
        if not sim:
            continue
        n_experts = max(max(int(r["expert_i"]), int(r["expert_j"])) for r in sim) + 1
        for case in CASES:
            base_lookup = {}
            for r in base_rows:
                if (int(r["calibration_size"]), int(r["layer"]), r["projection"], r["case"]) == (n, layer, proj, case):
                    base_lookup.setdefault((int(r["expert_i"]), int(r["expert_j"])), []).append(r)
            if not base_lookup:
                continue
            residual_lookup = {}
            for r in residual_rows:
                if (int(r["calibration_size"]), int(r["layer"]), r["projection"], r["case"], r["space"], r["rank"], r["method"]) == (n, layer, proj, case, "weight", str(a.matching_rank), "randomized"):
                    residual_lookup.setdefault((int(r["expert_i"]), int(r["expert_j"])), []).append(r)
            scoring = []
            for row in sim:
                key = (int(row["expert_i"]), int(row["expert_j"]))
                b = base_lookup.get(key, [])
                r = residual_lookup.get(key, [])
                row = dict(row)
                row["base_error_score"] = np.mean([float(x["calibration_error"])**2 for x in b]) if b else math.nan
                row["joint_cost"] = (row["base_error_score"] + a.joint_lambda *
                                     np.mean([float(x["estimated_normalized_tail"])**2 for x in r])) if r else math.nan
                scoring.append(row)
            criteria = [("random", "random", True), ("raw_weight", "raw_cosine", True),
                        ("subspace", f"subspace_overlap_r{a.input_ranks[-1]}", True),
                        ("functional", "functional_cosine", True),
                        ("activation_cosine", "activation_cosine", True),
                        ("hamming", "bit_agreement", True),
                        ("base_error", "base_error_score", False)]
            if residual_lookup:
                criteria.append(("joint_cost", "joint_cost", False))
            for name, key, maximize in criteria:
                try:
                    pairs = perfect_matching(scoring, key, maximize, n_experts, a.pairing_seed)
                except ValueError:
                    if name == "joint_cost":
                        raise ValueError("joint-cost candidate graph lacks a perfect matching; include random controls")
                    raise
                selected = [r for pair in pairs for r in base_lookup[pair]]
                output.append({"layer": layer, "projection": proj, "calibration_size": n,
                               "case": case, "criterion": name,
                               "matching_scope": "candidate_restricted" if name == "joint_cost" and a.spectrum_scope != "all" else "all_pairs",
                               "pair_count": len(pairs), "pairs": json.dumps(pairs),
                               "calibration_error": float(np.mean([float(x["calibration_error"]) for x in selected])),
                               "heldout_error": float(np.mean([float(x["heldout_error"]) for x in selected])),
                               "weight_error": float(np.mean([float(x["weight_error"]) for x in selected]))})
    return output


def plots(a, sim, base, residual, correlations_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = a.output_dir / "plots"
    folder.mkdir(exist_ok=True)
    for n, layer, proj in itertools.product(a.calibration_sizes, a.layers, a.projections):
        rows = [r for r in sim if (int(r["calibration_size"]), int(r["layer"]), r["projection"]) == (n, layer, proj)]
        if not rows:
            continue
        prefix = f"s{n}_l{layer}_{proj}"
        m = max(max(int(r["expert_i"]), int(r["expert_j"])) for r in rows) + 1
        heat = np.eye(m)
        for r in rows:
            i, j = int(r["expert_i"]), int(r["expert_j"])
            heat[i, j] = heat[j, i] = float(r["activation_cosine"])
        fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(heat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title("Activation-conditioned weight cosine"); fig.tight_layout()
        fig.savefig(folder / f"{prefix}_similarity_heatmap.png", dpi=140); plt.close(fig)
        base_map = {}
        for r in base:
            if (int(r["calibration_size"]), int(r["layer"]), r["projection"], r["case"]) == (n, layer, proj, "F"):
                base_map.setdefault((int(r["expert_i"]), int(r["expert_j"])), []).append(float(r["heldout_error"]))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter([float(r["activation_cosine"]) for r in rows],
                   [np.mean(base_map[(int(r["expert_i"]), int(r["expert_j"]))]) for r in rows], s=5)
        ax.set(xlabel="Activation-conditioned cosine", ylabel="Held-out Base error")
        fig.tight_layout(); fig.savefig(folder / f"{prefix}_similarity_vs_error.png", dpi=140); plt.close(fig)
        cor = [r for r in correlations_rows if (int(r["calibration_size"]), int(r["layer"]), r["projection"], r["case"], r["outcome"]) == (n, layer, proj, "F", "heldout_error")]
        if cor:
            fig, ax = plt.subplots(figsize=(max(6, len(cor)*.5), 2))
            ax.imshow(np.array([[float(r["pearson"]) for r in cor]]), vmin=-1, vmax=1, cmap="coolwarm")
            ax.set_xticks(range(len(cor)), [r["similarity"] for r in cor], rotation=90)
            ax.set_yticks([0], ["Pearson"]); fig.tight_layout()
            fig.savefig(folder / f"{prefix}_correlation_heatmap.png", dpi=140); plt.close(fig)
    if residual:
        paths = sorted((a.output_dir / "spectra").glob("*.npz"))[:12]
        for kind in ("singular_values", "cumulative_energy"):
            fig, ax = plt.subplots(figsize=(7, 5))
            for path in paths:
                with np.load(path) as arr:
                    ax.plot(arr[kind], alpha=.5)
            ax.set(xlabel="Rank", ylabel=kind)
            if kind == "singular_values": ax.set_yscale("log")
            fig.tight_layout(); fig.savefig(folder / f"{kind}.png", dpi=140); plt.close(fig)
        curves = {}
        for r in residual:
            if r["space"] == "weight" and r["method"] == "randomized":
                curves.setdefault((r["case"], int(r["rank"])), []).append(float(r["actual_heldout_error"]))
        fig, ax = plt.subplots(figsize=(7, 5))
        for case in sorted(set(k[0] for k in curves)):
            xs = sorted(k[1] for k in curves if k[0] == case)
            ax.plot(xs, [np.mean(curves[case, x]) for x in xs], label=case)
        ax.set(xlabel="Residual rank", ylabel="Held-out reconstruction error"); ax.legend()
        fig.tight_layout(); fig.savefig(folder / "rank_vs_error.png", dpi=140); plt.close(fig)


def report(a):
    sim = read_csv(a.output_dir / "pairwise_metrics.csv")
    bas = read_csv(a.output_dir / "base_metrics.csv")
    res = read_csv(a.output_dir / "residual_metrics.csv")
    if not sim or not bas:
        raise FileNotFoundError("similarity and base CSV are required for report")
    cor = correlations(a, sim, bas, res)
    groups = grouping(a, sim, bas, res)
    write_csv(a.output_dir / "correlation.csv", cor)
    write_csv(a.output_dir / "grouping_summary.csv", groups)
    plots(a, sim, bas, res, cor)
    best_grouping = []
    for n, layer, proj, case in itertools.product(a.calibration_sizes, a.layers, a.projections, CASES):
        candidates = [r for r in groups if (r["calibration_size"], r["layer"], r["projection"], r["case"]) ==
                      (n, layer, proj, case)]
        if candidates:
            best = min(candidates, key=lambda r: r["heldout_error"])
            best_grouping.append({"calibration_size": n, "layer": layer,
                                  "projection": proj, "case": case, "criterion": best["criterion"],
                                  "heldout_error": best["heldout_error"],
                                  "matching_scope": best["matching_scope"]})
    summary = {"pair_rows": len(sim), "base_rows": len(bas), "residual_rows": len(res),
               "correlation_rows": len(cor), "grouping_rows": len(groups),
               "best_grouping_by_heldout_error": best_grouping,
               "stages": json.loads((a.output_dir / "stage_runtime.json").read_text()) if
                         (a.output_dir / "stage_runtime.json").exists() else {},
               "matching_note": "joint-cost matching is candidate-restricted when spectrum-scope=representative",
               "heldout_usage": "heldout covariance used only for final metrics; no fitting or matching"}
    atomic_json(a.output_dir / "summary.json", summary)
    text = ["# Activation-aware Base Bundling Analysis", "",
            f"Pair rows: {len(sim)}; Base rows: {len(bas)}; residual rows: {len(res)}.", "",
            "Grouping uses calibration scores. Held-out errors are reported only after matching.", "",
            "Case A/B share a dense weight exactly; C1/C2 are individual binary; D-H share one binary code with expert row scales.", "",
            "Randomized SVD energy is an estimate; actual BF16 factor error is measured separately.", "",
            "## Lowest held-out error by unit", ""]
    for r in best_grouping:
        text.append(f"- {r['calibration_size']} / layer {r['layer']} / {r['projection']} / {r['case']}: "
                    f"{r['criterion']} ({r['heldout_error']:.5f}, {r['matching_scope']})")
    text += ["", "## Grouping summary", ""]
    for r in groups:
        text.append(f"- {r['calibration_size']} / layer {r['layer']} / {r['projection']} / {r['case']} / {r['criterion']}: "
                    f"held-out error {r['heldout_error']:.5f} ({r['matching_scope']})")
    (a.output_dir / "REPORT.md").write_text("\n".join(text) + "\n")


def main(argv=None):
    a = arguments(argv)
    prepare(a)
    stages = ["calibrate", "similarity", "base", "spectrum", "report"] if a.stage == "all" else [a.stage]
    model = tokenizer = None
    runtimes_path = a.output_dir / "stage_runtime.json"
    runtimes = json.loads(runtimes_path.read_text()) if runtimes_path.exists() else {}
    for stage in stages:
        if stage != "report" and model is None:
            if stage == "calibrate":
                model, tokenizer = load_model(a, True)
            else:
                model = load_model(a)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        {"calibrate": lambda: calibrate(a, model, tokenizer),
         "similarity": lambda: similarity(a, model),
         "base": lambda: base(a, model),
         "spectrum": lambda: spectrum(a, model),
         "report": lambda: report(a)}[stage]()
        runtimes[stage] = {"seconds": time.perf_counter() - start,
                           "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
        atomic_json(runtimes_path, runtimes)
        release_unit()
        if stage == "report":
            summary_path = a.output_dir / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["stages"] = runtimes
            atomic_json(summary_path, summary)
        print(f"[{stage}] complete: {runtimes[stage]}", flush=True)


if __name__ == "__main__":
    main()
