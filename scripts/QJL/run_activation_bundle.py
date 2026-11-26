#!/usr/bin/env python3
"""Calibrate, compress, evaluate, and report activation-aware QJL bundles."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from diagnose_codebook_base import atomic_json, select
from rescode.evaluation import evaluate_fewshot
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.activation_bundle_model import (CASES, FORMAT, checkpoint_sizes,
                                                 compress_layer, load_checkpoint)
from rescode.qjl.base_bundle_analysis import StreamingRoutedMoments
from rescode.qjl.activation_calibration import (FORMAT as CALIBRATION_FORMAT,
                                                collect_layerwise, token_batches)
from run_base_bundling import load_model, original_storage, save_csv, save_torch

DEFAULT_TASKS = "arc_easy,arc_challenge,hellaswag,winogrande"
ALL_TASKS = "winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("calibrate", "compress", "eval", "sweep", "report"), required=True)
    p.add_argument("--model-path", type=Path, default=Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat"))
    p.add_argument("--output-dir", type=Path, default=Path("results/activation_bundle/d2n_k2"))
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--cache-dir", type=Path)
    p.add_argument("--calibration-jsonl", type=Path)
    p.add_argument("--heldout-jsonl", type=Path)
    p.add_argument("--calibration-size", type=int, default=512)
    p.add_argument("--heldout-size", type=int, default=128)
    p.add_argument("--calibration-batch-size", type=int, default=8)
    p.add_argument("--calibration-backend", choices=("reference", "layerwise"), default="layerwise")
    p.add_argument("--hidden-cache", choices=("auto", "gpu", "cpu", "disk"), default="auto")
    p.add_argument("--length-bucketing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--layers", default="all")
    p.add_argument("--case", choices=("original",) + CASES)
    p.add_argument("--cases", default="A,B,C,D,E")
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--pairing", choices=("random", "similarity"), default="similarity")
    p.add_argument("--qjl-d-factor", type=float, default=2.)
    p.add_argument("--rank", type=int, help="single compression rank or evaluation rank")
    p.add_argument("--ranks", help="comma-separated compression ranks; default 32,64,128,256,352")
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--pairing-seed", type=int, default=0)
    p.add_argument("--bundle-seed", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--svd-method", choices=("exact", "randomized"), default="exact")
    p.add_argument("--svd-oversample", type=int, default=16)
    p.add_argument("--svd-niter", type=int, default=2)
    p.add_argument("--svd-seed", type=int, default=0)
    p.add_argument("--refine-sweeps", type=int, default=1)
    p.add_argument("--refine-block", type=int, default=32)
    p.add_argument("--refine-all-bits", action="store_true")
    p.add_argument("--row-chunk", type=int, default=64)
    p.add_argument("--refine-row-chunk", type=int, default=256,
                   help="0 processes all rows together")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--backend", choices=("triton", "reference"), default="triton")
    p.add_argument("--task", default=DEFAULT_TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-fraction", type=float, default=.94)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args(argv)
    args.cases = [x.strip() for x in args.cases.split(",") if x.strip()]
    if len(set(args.cases)) != len(args.cases) or any(x not in ("original",) + CASES for x in args.cases):
        p.error("--cases must list distinct original,A,B,C,D,E values")
    if args.mode == "eval" and args.case is None:
        p.error("--case is required for eval")
    if (args.calibration_jsonl is None) != (args.heldout_jsonl is None):
        p.error("provide both --calibration-jsonl and --heldout-jsonl, or neither")
    args.explicit_rank = args.rank is not None
    try:
        args.ranks = ([int(x) for x in args.ranks.split(",")] if args.ranks else
                      [args.rank] if args.rank is not None else [32, 64, 128, 256, 352])
    except ValueError:
        p.error("--ranks must be comma-separated integers")
    args.ranks = sorted(set(args.ranks))
    if args.rank is None:
        args.rank = max(args.ranks)
    if (args.group_size < 2 or 60 % args.group_size or args.qjl_d_factor <= 0 or
            any(rank < 0 for rank in args.ranks) or args.rank < 0 or args.ridge <= 0 or
            args.row_chunk < 1 or args.refine_row_chunk < 0 or args.refine_block < 1 or
            args.svd_oversample < 0 or args.svd_niter < 0 or args.calibration_batch_size < 1 or
            args.refine_sweeps < 0 or args.calibration_size < 1 or args.heldout_size < 1 or
            args.eval_batch_size < 1 or args.eval_limit < 0):
        p.error("invalid dimension, group, rank, calibration, refinement, or evaluation option")
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = (args.checkpoint_dir or args.output_dir / "checkpoint").resolve()
    args.cache_dir = (args.cache_dir or Path("results/activation_bundle/shared_cache")).resolve()
    return args


def _texts(path, limit):
    texts = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                text = item["text"]
                if not isinstance(text, str) or not text:
                    raise ValueError("JSONL records need a nonempty text field")
                texts.append(text)
            if len(texts) == limit:
                break
    if len(texts) != limit:
        raise ValueError(f"{path}: expected {limit} records, found {len(texts)}")
    return texts


def prepare_default_texts(args):
    """Create disjoint, deterministic WikiText-103 JSONL inputs on demand."""
    if args.calibration_jsonl is not None:
        return
    from datasets import load_dataset
    target = args.output_dir / "data"
    target.mkdir(parents=True, exist_ok=True)
    paths = {"calibration": target / "wikitext103_calibration.jsonl",
             "heldout": target / "wikitext103_heldout.jsonl"}
    for label, split, count in (("calibration", "train", args.calibration_size),
                                ("heldout", "validation", args.heldout_size)):
        path = paths[label]
        if path.exists():
            try:
                if len(_texts(path, count)) == count:
                    continue
            except ValueError:
                pass
        temporary = path.with_suffix(".jsonl.tmp")
        selected = 0
        with temporary.open("w") as handle:
            for record in load_dataset("wikitext", "wikitext-103-raw-v1",
                                       split=split, streaming=True):
                text = record["text"].strip()
                if len(text.split()) < 8 or text.startswith("= "):
                    continue
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                selected += 1
                if selected == count:
                    break
        if selected != count:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"WikiText-103 {split} has only {selected} usable records")
        os.replace(temporary, path)
    args.calibration_jsonl = paths["calibration"]
    args.heldout_jsonl = paths["heldout"]
    atomic_json(target / "source.json", {"dataset": "wikitext-103-raw-v1",
                                         "calibration_split": "train",
                                         "heldout_split": "validation",
                                         "calibration_size": args.calibration_size,
                                         "heldout_size": args.heldout_size})


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_identity(path: Path):
    """Identify the local model revision without hashing tens of GB of weights."""
    identity = {name: (_sha256(path / name) if (path / name).is_file() else None)
                for name in ("config.json", "model.safetensors.index.json")}
    metadata = path / ".cache/huggingface/download"
    config_metadata = metadata / "config.json.metadata"
    identity["hub_revision"] = (config_metadata.read_text().splitlines()[0]
                                if config_metadata.is_file() else None)
    identity["weight_shards"] = [
        {"name": shard.name, "bytes": shard.stat().st_size,
         "etag": ((metadata / (shard.name + ".metadata")).read_text().splitlines()[1]
                  if (metadata / (shard.name + ".metadata")).is_file() else None)}
        for shard in sorted(path.glob("*.safetensors"))]
    return identity


def _cache_source(args, layers):
    return {"format": CALIBRATION_FORMAT,
            "model_path": str(args.model_path.resolve()),
            "model_identity": _model_identity(args.model_path),
            "layers": layers, "calibration_size": args.calibration_size,
            "heldout_size": args.heldout_size, "max_length": args.max_length,
            "calibration_batch_size": args.calibration_batch_size,
            "length_bucketing": args.length_bucketing,
            "calibration_jsonl": (str(args.calibration_jsonl.resolve())
                                  if args.calibration_jsonl else None),
            "heldout_jsonl": (str(args.heldout_jsonl.resolve())
                              if args.heldout_jsonl else None),
            "calibration_sha256": (_sha256(args.calibration_jsonl)
                                   if args.calibration_jsonl else None),
            "heldout_sha256": (_sha256(args.heldout_jsonl)
                               if args.heldout_jsonl else None)}


def cache_complete(args, layers):
    """Validate cache identity before trusting any covariance shard."""
    manifest = args.cache_dir / "calibration_manifest.json"
    shards = [args.cache_dir / f"{split}_layer_{layer}.pt"
              for split in ("calibration", "heldout") for layer in layers]
    if not manifest.is_file():
        if any(path.exists() for path in shards):
            raise ValueError("unmanifested covariance cache; use another CACHE_DIR")
        return False
    old = json.loads(manifest.read_text())
    wanted = _cache_source(args, layers)
    keys = ["model_path", "calibration_size", "heldout_size", "max_length"]
    if old.get("format") == CALIBRATION_FORMAT:
        keys += ["calibration_batch_size", "length_bucketing"]
    for key in keys:
        if old.get(key) != wanted[key]:
            raise ValueError(f"covariance cache {key} differs; use another CACHE_DIR")
    if not set(layers).issubset(old.get("layers", [])):
        raise ValueError("covariance cache does not cover requested layers")
    if old.get("format") == CALIBRATION_FORMAT:
        if old.get("model_identity") != wanted["model_identity"]:
            raise ValueError("model revision differs from covariance cache")
    elif old.get("format") is not None:
        raise ValueError(f"unknown covariance cache format: {old['format']}")
    # Version-1 manifests lack a revision hash, but their source path and
    # dataset fingerprints remain valid and their layer payloads are readable.
    for split in ("calibration", "heldout"):
        source = getattr(args, f"{split}_jsonl", None)
        if source is None and old.get(f"{split}_jsonl"):
            source = Path(old[f"{split}_jsonl"])
        if source is None or not source.is_file():
            raise ValueError(f"{split} dataset cannot be verified from covariance cache")
        fingerprint = _sha256(source)
        if old.get(f"{split}_sha256") != fingerprint:
            raise ValueError(f"{split} dataset differs from covariance cache")
    return all(path.is_file() for path in shards)


def calibrate(args, model, tokenizer, layers):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    calibration = _texts(args.calibration_jsonl, args.calibration_size)
    heldout = _texts(args.heldout_jsonl, args.heldout_size)
    fingerprints = [{hashlib.sha256(s.encode()).digest() for s in group}
                    for group in (calibration, heldout)]
    if fingerprints[0] & fingerprints[1]:
        raise ValueError("calibration and held-out text overlap")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    source = _cache_source(args, layers)
    source_path = args.cache_dir / "calibration_manifest.json"
    already_complete = cache_complete(args, layers)
    if already_complete and args.resume:
        print("[calibrate] compatible covariance cache reused", flush=True)
        return
    if not source_path.exists():
        atomic_json(source_path, source)
    elif json.loads(source_path.read_text()).get("format") is None:
        # Adopt a compatible version-1 cache. Its covariance payload remains
        # unchanged; the new manifest adds the model revision hash.
        atomic_json(source_path, source)
    if args.calibration_backend == "layerwise":
        batches = {"calibration": token_batches(tokenizer, calibration,
                       args.calibration_batch_size, args.max_length, args.length_bucketing),
                   "heldout": token_batches(tokenizer, heldout,
                       args.calibration_batch_size, args.max_length, args.length_bucketing)}
        profiles, storage = collect_layerwise(model.model, batches, layers, args.cache_dir,
             hidden_cache=args.hidden_cache, resume=args.resume, profile=args.profile)
        atomic_json(args.cache_dir / "calibration_profile.json",
                    {"backend": "layerwise", "hidden_cache": storage,
                     "layers": profiles,
                     "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                     "cache_bytes": sum(
                         path.stat().st_size for path in args.cache_dir.rglob("*") if path.is_file())})
        return
    reference_profile = []
    for split, texts in (("calibration", calibration), ("heldout", heldout)):
        paths = [args.cache_dir / f"{split}_layer_{layer}.pt" for layer in layers]
        if args.resume and all(path.exists() for path in paths):
            print(f"[calibrate] {split}: resumed", flush=True)
            continue
        collector = StreamingRoutedMoments(model, layers, probe_limit=0, profile=args.profile)
        forward_seconds = 0.
        batches = token_batches(tokenizer, texts, args.calibration_batch_size,
                                args.max_length, args.length_bucketing)
        try:
            with torch.no_grad():
                for batch_number, batch in enumerate(batches, 1):
                    collector.begin_batch(batch["attention_mask"])
                    device = next(model.parameters()).device
                    if args.profile and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    # Routed hooks live in the backbone. Avoid constructing
                    # the unused full-vocabulary LM logits during calibration.
                    model.model(input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device), use_cache=False)
                    if args.profile and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    forward_seconds += time.perf_counter() - started
                    print(f"[calibrate] {split}: {batch_number}/{len(batches)} batches", flush=True)
            snapshot = collector.snapshot()
            checkpoint_started = time.perf_counter()
            for layer in layers:
                save_torch(args.cache_dir / f"{split}_layer_{layer}.pt", snapshot[layer])
            reference_profile.append({"split": split, "forward_seconds": forward_seconds,
                **collector.profile, "checkpoint_io_seconds": time.perf_counter() - checkpoint_started})
        finally:
            collector.close()
            del collector
            gc.collect()
    atomic_json(source_path, source)
    atomic_json(args.cache_dir / "calibration_profile.json",
                {"backend": "reference", "splits": reference_profile,
                 "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                 "cache_bytes": sum(path.stat().st_size for path in args.cache_dir.rglob("*")
                                    if path.is_file())})


def _config(args, layers):
    source_path = args.cache_dir / "calibration_manifest.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else None
    return {"format": FORMAT, "model_path": str(args.model_path.resolve()),
            "calibration_source": source,
            "layers": layers, "qjl_d_factor": args.qjl_d_factor,
            "rank": max(args.ranks), "ranks": args.ranks,
            "group_size": args.group_size, "pairing": args.pairing,
            "projection_seed": args.projection_seed, "pairing_seed": args.pairing_seed,
            "bundle_seed": args.bundle_seed, "ridge": args.ridge,
            "svd_method": args.svd_method, "svd_oversample": args.svd_oversample,
            "svd_niter": args.svd_niter, "svd_seed": args.svd_seed,
            "refine_sweeps": args.refine_sweeps, "refine_block": args.refine_block,
            "refine_row_chunk": args.refine_row_chunk,
            "conflict_only": not args.refine_all_bits,
            "row_chunk": args.row_chunk, "cases": [c for c in args.cases if c != "original"]}


def _layer_paths(root, layer, config):
    names = [f"base_{case}_layer_{layer}.pt" for case in ("A", "B")]
    if "E" in config["cases"]:
        names.extend(f"base_E_r{rank}_layer_{layer}.pt" for rank in config["ranks"])
    names += [f"factor_{case}_r{rank}_layer_{layer}.pt"
              for case in ("C", "D", "E") if case in config["cases"]
              for rank in config["ranks"]]
    names.append(f"per_expert_layer_{layer}.csv")
    names.append(f"spectrum_layer_{layer}.npz")
    names.append(f"compression_layer_{layer}.json")
    return [root / name for name in names]


def compress(args, model, layers):
    if not cache_complete(args, layers):
        raise FileNotFoundError("compatible calibration and held-out covariance shards are required")
    root = args.checkpoint_dir
    root.mkdir(parents=True, exist_ok=True)
    config = _config(args, layers)
    manifest = root / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old != config:
            raise ValueError("checkpoint manifest differs; choose another directory")
        if not args.resume:
            raise FileExistsError("checkpoint exists; pass --resume")
    else:
        atomic_json(manifest, config)
    banks = {}
    for layer in layers:
        paths = _layer_paths(root, layer, config)
        if args.resume and all(path.exists() for path in paths):
            print(f"[compress] layer {layer}: resumed", flush=True)
            block = model.model.layers[layer].mlp
            block.experts = nn.ModuleList(nn.Identity() for _ in block.experts)
            continue
        cal_path = args.cache_dir / f"calibration_layer_{layer}.pt"
        if not cal_path.exists():
            raise FileNotFoundError(f"run --mode calibrate first: {cal_path}")
        calibration = torch.load(cal_path, map_location="cpu", weights_only=False)
        held_path = args.cache_dir / f"heldout_layer_{layer}.pt"
        if not held_path.exists():
            raise FileNotFoundError(f"held-out covariance cache missing: {held_path}")
        heldout = torch.load(held_path, map_location="cpu", weights_only=False)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        stage_profile = {} if args.profile else None
        base, factors, refined, records, spectra = compress_layer(model.model.layers[layer].mlp,
                                            layer, calibration, config, banks, heldout,
                                            profile=stage_profile)
        checkpoint_started = time.perf_counter()
        for case, state in base.items():
            save_torch(root / f"base_{case}_layer_{layer}.pt", state)
        if "E" in config["cases"]:
            for rank in config["ranks"]:
                save_torch(root / f"base_E_r{rank}_layer_{layer}.pt", refined[rank])
        for case in ("C", "D"):
            if case in config["cases"]:
                for rank in config["ranks"]:
                    sliced = {"format": FORMAT, "layer": layer, "case": case,
                              "rank": rank, "projections": {}}
                    for key, part in factors[case]["projections"].items():
                        sliced["projections"][key] = {
                            i: {"u": pair["u"][:, :rank].contiguous(),
                                "v": pair["v"][:rank].contiguous()}
                            for i, pair in part.items()}
                    save_torch(root / f"factor_{case}_r{rank}_layer_{layer}.pt", sliced)
        if "E" in config["cases"]:
            for rank in config["ranks"]:
                save_torch(root / f"factor_E_r{rank}_layer_{layer}.pt", factors["E"][rank])
        fields = list(dict.fromkeys(key for row in records for key in row))
        save_csv(root / f"per_expert_layer_{layer}.csv",
                 [{key: row.get(key) for key in fields} for row in records])
        np.savez_compressed(root / f"spectrum_layer_{layer}.npz", **spectra)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        atomic_json(root / f"compression_layer_{layer}.json", {
            "layer": layer, "wall_seconds": time.perf_counter() - started,
            "profile_enabled": args.profile, "stages": stage_profile or {},
            "checkpoint_io_seconds": checkpoint_seconds,
            "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "covariance_cache_bytes": cal_path.stat().st_size + held_path.stat().st_size,
            "gpu_peak_allocated_bytes": [torch.cuda.max_memory_allocated(i)
                                         for i in range(torch.cuda.device_count())]})
        block = model.model.layers[layer].mlp
        block.experts = nn.ModuleList(nn.Identity() for _ in block.experts)
        del calibration, heldout, base, factors, refined, records, spectra
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[compress] layer {layer}: {time.perf_counter() - started:.1f}s", flush=True)
    report(args)


def _resident(model):
    sketch, scales, factors = {}, {}, {}
    for layer in model.model.layers:
        for expert in getattr(layer.mlp, "_compact_experts", ()):
            for base in expert.bases.values():
                for tensor, bucket in ((base.packed, sketch), (base.norms, scales)):
                    if tensor.is_cuda:
                        storage = tensor.untyped_storage()
                        bucket[storage.data_ptr()] = storage.nbytes()
            for correction in expert.corrections.values():
                for tensor in (correction.u, correction.v):
                    if tensor.is_cuda:
                        storage = tensor.untyped_storage()
                        factors[storage.data_ptr()] = storage.nbytes()
    projection = sum(bank.resident_bytes("cuda") for bank in
                     getattr(model, "_activation_bundle_banks", {}).values())
    return {"gpu_shared_sketch_bytes": sum(sketch.values()),
            "gpu_row_scale_bytes": sum(scales.values()),
            "gpu_lowrank_bytes": sum(factors.values()),
            "gpu_projection_bytes": projection,
            "gpu_compact_total_bytes": sum(sketch.values()) + sum(scales.values()) +
                                       sum(factors.values()) + projection}


def evaluation_signature(args, case, rank):
    manifest = args.checkpoint_dir / "manifest.json"
    return {"case": case, "rank": rank if case in ("C", "D", "E") else 0,
            "task": args.task, "eval_limit": args.eval_limit,
            "eval_batch_size": args.eval_batch_size, "backend": args.backend,
            "model_path": str(args.model_path.resolve()),
            "model_identity": _model_identity(args.model_path),
            "checkpoint_dir": (str(args.checkpoint_dir) if case != "original" else None),
            "checkpoint_manifest_sha256": (_sha256(manifest) if case != "original" and
                                            manifest.exists() else None)}


def eval_result_path(args, case, rank):
    return (args.output_dir / case /
            (f"rank_{rank}" if case in ("C", "D", "E") else "") /
            "eval_metrics.json")


def completed_eval_matches(args, case, rank):
    path = eval_result_path(args, case, rank)
    if not path.exists():
        return False
    existing = json.loads(path.read_text())
    wanted = evaluation_signature(args, case, rank)
    if existing.get("request_signature") != wanted:
        raise ValueError(f"saved evaluation settings differ: {path}; use another OUTPUT_DIR")
    return True


def evaluate(args, model, tokenizer, layers):
    case = args.case
    rank = args.rank
    if case in ("C", "D", "E") and not args.explicit_rank:
        manifest = json.loads((args.checkpoint_dir / "manifest.json").read_text())
        rank = max(manifest["ranks"])
    if case == "original":
        storage = original_storage(model, args.model_path, layers)
        config = None
    else:
        config = load_checkpoint(model, args.checkpoint_dir, case, args.backend, rank)
        storage = checkpoint_sizes(args.checkpoint_dir, config, case, rank)
        original_full = sum(path.stat().st_size for path in args.model_path.glob("*.safetensors"))
        routed_original = 2 * storage["routed_weight_count"]
        if original_full:
            estimated_full = original_full - routed_original + storage["actual_checkpoint_bytes"]
            storage.update({"original_model_checkpoint_bytes": original_full,
                            "estimated_full_model_checkpoint_bytes": estimated_full,
                            "whole_model_storage_reduction": 1 - estimated_full / original_full,
                            "expert_compression_ratio": routed_original / storage["actual_checkpoint_bytes"]})
    case_dir = args.output_dir / case / (f"rank_{rank}" if case in ("C", "D", "E") else "")
    case_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = evaluate_fewshot(model, tokenizer, args.task, 0,
        eval_batch_size=args.eval_batch_size, output_path=str(case_dir / "lm_eval.txt"),
        limit=args.eval_limit or None)
    elapsed = time.perf_counter() - started
    summary = report_eval_summary(result, args.task, str(case_dir / "lm_eval.txt"),
                                  header=f"Activation bundle {case}")
    runtime = _resident(model) if config else {}
    metrics = {"case": case, "rank": rank if case in ("C", "D", "E") else 0,
               "request_signature": evaluation_signature(args, case, rank),
               "task": args.task, "eval_limit": args.eval_limit,
               "eval_batch_size": args.eval_batch_size, "backend": args.backend,
               "evaluation_wall_seconds": elapsed, "task_summary": summary,
               "task_results": result.get("results", {}), "storage": storage,
               "runtime": runtime,
               "gpu_process_allocated_bytes": [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())],
               "gpu_process_peak_allocated_bytes": [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]}
    atomic_json(case_dir / "eval_metrics.json", metrics)
    report(args)
    print(f"[eval] saved {case_dir / 'eval_metrics.json'}", flush=True)


def report(args):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = args.checkpoint_dir / "manifest.json"
    config = json.loads(manifest.read_text()) if manifest.exists() else None
    all_rows = []
    if config:
        for layer in config["layers"]:
            path = args.checkpoint_dir / f"per_expert_layer_{layer}.csv"
            if path.exists():
                with path.open(newline="") as handle:
                    all_rows.extend(csv.DictReader(handle))
        if all_rows:
            save_csv(out / "per_expert.csv", all_rows)
            grouped = {}
            fields = ("pre_bundle_bit_agreement", "bit_preservation", "base_weight_relative_error",
                      "base_activation_loss", "residual_weight_ratio", "residual_activation_ratio",
                      "activation_loss", "heldout_activation_loss",
                      "activation_relative_l2_error", "heldout_relative_l2_error",
                      "weight_relative_error", "svd_energy", "theoretical_tail_error",
                      "accepted_flip_ratio")
            for row in all_rows:
                key = (row["case"], int(row["rank"]))
                grouped.setdefault(key, []).append(row)
            aggregate = []
            for (case, rank), subset in sorted(grouped.items()):
                item = {"case": case, "rank": rank, "matrix_count": len(subset)}
                for field in fields:
                    values = [float(row[field]) for row in subset if row.get(field) not in (None, "", "None")]
                    if values:
                        item[f"mean_{field}"] = sum(values) / len(values)
                aggregate.append(item)
            atomic_json(out / "analysis_summary.json", aggregate)
            spectrum_paths = [args.checkpoint_dir / f"spectrum_layer_{layer}.npz"
                              for layer in config["layers"]]
            plot_dir = out / "plots"
            plot_paths = [plot_dir / "spectrum.png", plot_dir / "cumulative_energy.png"]
            refresh_plots = (any(not path.exists() for path in plot_paths) or
                             any(path.exists() and path.stat().st_mtime >
                                 min(plot.stat().st_mtime for plot in plot_paths if plot.exists())
                                 for path in spectrum_paths))
            spectra = {}
            if refresh_plots:
                source_energies = {row["spectrum_file"].split(":", 1)[-1]:
                                   float(row["spectrum_source_energy"])
                                   for row in all_rows if row.get("spectrum_file") and
                                   row.get("spectrum_source_energy") not in (None, "")}
                for path in spectrum_paths:
                    if not path.exists():
                        continue
                    with np.load(path) as data:
                        for name in data.files:
                            case = name.split("_")[-1]
                            if name.endswith(f"_E_r{max(config['ranks'])}"):
                                case = "E"
                            if case in ("C", "D", "E"):
                                values = data[name].astype(np.float64)
                                energy = values ** 2
                                denominator = source_energies.get(name, energy.sum())
                                spectra.setdefault(case, []).append(energy / max(denominator, 1e-30))
            if spectra:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                plot_dir.mkdir(exist_ok=True)
                for filename, cumulative in (("spectrum.png", False),
                                             ("cumulative_energy.png", True)):
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for case, arrays in spectra.items():
                        size = min(map(len, arrays))
                        mean = np.mean(np.stack([a[:size] for a in arrays]), axis=0)
                        ax.plot(np.arange(1, size + 1), mean.cumsum() if cumulative else mean,
                                label=case)
                    if not cumulative:
                        ax.set_yscale("log")
                    ax.set_xlabel("Rank")
                    ax.set_ylabel("Fraction of full residual energy" if cumulative
                                  else "Singular energy / full residual energy")
                    if config["svd_method"] == "randomized":
                        ax.set_title("Randomized truncated spectrum estimate")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(plot_dir / filename, dpi=160)
                    plt.close(fig)
    summary_rows = []
    paths = ([out / case / "eval_metrics.json" for case in ("original", "A", "B")] +
             [out / case / f"rank_{rank}" / "eval_metrics.json"
              for case in ("C", "D", "E") for rank in
              (config["ranks"] if config else args.ranks)])
    for path in paths:
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        case = metrics["case"]
        task_summary = metrics["task_summary"]
        row = {"case": case, "rank": metrics.get("rank", 0),
               "average_accuracy": task_summary["avg"],
               "average_acc_norm": task_summary["avg_norm"],
               "evaluation_wall_seconds": metrics["evaluation_wall_seconds"]}
        for task, acc, norm in task_summary["tasks"]:
            row[f"{task}_acc"] = acc
            row[f"{task}_acc_norm"] = norm
        row.update(metrics["storage"])
        summary_rows.append(row)
    if summary_rows:
        fields = list(dict.fromkeys(key for row in summary_rows for key in row))
        save_csv(out / "summary.csv", [{key: row.get(key) for key in fields}
                                        for row in summary_rows])
        atomic_json(out / "summary.json", summary_rows)
    lines = ["# Activation-aware QJL binary bundling", "",
             f"Checkpoint: `{args.checkpoint_dir}`", "",
             f"Calibrated layer count: {len(config['layers']) if config else 0}", "",
             f"Per-expert records: {len(all_rows)}", ""]
    if config:
        total_seconds = 0.
        peaks = []
        cpu_peaks = []
        cache_bytes = 0
        stage_totals = {}
        for layer in config["layers"]:
            path = args.checkpoint_dir / f"compression_layer_{layer}.json"
            if path.exists():
                item = json.loads(path.read_text())
                total_seconds += item["wall_seconds"]
                peaks.extend(item["gpu_peak_allocated_bytes"])
                cpu_peaks.append(item.get("cpu_peak_rss_bytes", 0))
                cache_bytes += item.get("covariance_cache_bytes", 0)
                stage_totals["checkpoint_io_seconds"] = (
                    stage_totals.get("checkpoint_io_seconds", 0.) +
                    item.get("checkpoint_io_seconds", 0.))
                for name, seconds in item.get("stages", {}).items():
                    stage_totals[name] = stage_totals.get(name, 0.) + seconds
        calibration_profile_path = args.cache_dir / "calibration_profile.json"
        calibration_profile = (json.loads(calibration_profile_path.read_text())
                               if calibration_profile_path.exists() else None)
        atomic_json(out / "stage_profile.json", {"calibration": calibration_profile,
                    "compression_stage_seconds": stage_totals,
                    "compression_wall_seconds": total_seconds,
                    "compression_peak_gpu_bytes": max(peaks, default=0),
                    "compression_peak_cpu_rss_bytes": max(cpu_peaks, default=0),
                    "selected_covariance_cache_bytes": cache_bytes})
        if total_seconds:
            lines += [f"Compression time across completed layers: {total_seconds:.1f} s.",
                      f"Peak GPU allocation during compression: {max(peaks, default=0)} bytes.",
                      f"Peak CPU RSS during compression: {max(cpu_peaks, default=0)} bytes.",
                      f"Selected covariance cache bytes: {cache_bytes}.", ""]
        if calibration_profile:
            lines += [f"Calibration backend: {calibration_profile['backend']}.",
                      f"Calibration cache bytes: {calibration_profile.get('cache_bytes', 0)}.", ""]
        if stage_totals:
            lines += ["| Compression stage | Seconds |", "|---|---:|"]
            lines += [f"| {name} | {seconds:.3f} |"
                      for name, seconds in sorted(stage_totals.items())]
            lines += [""]
    if summary_rows:
        lines += ["| Case | Mean accuracy | Mean acc_norm | Checkpoint bytes | BPW |",
                  "|---|---:|---:|---:|---:|"]
        for row in summary_rows:
            lines.append(f"| {row['case']} | {row['average_accuracy']:.4f} | "
                         f"{row['average_acc_norm']:.4f} | {row['actual_checkpoint_bytes']} | "
                         f"{row['effective_bpw']:.3f} |")
    else:
        lines += ["Evaluation has not been run."]
    lines += ["", "A and B contain packed group sketches and expert scales only. "
              "C and D use B's base; E uses a refined code. Calibration cache bytes "
              "are excluded from inference checkpoint bytes. Original checkpoint "
              "bytes count the whole BF16 model, while compressed checkpoint bytes "
              "count routed weights of selected layers.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main(argv=None):
    args = arguments(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "report":
        report(args)
        return
    if args.mode == "sweep":
        cache_manifest = args.cache_dir / "calibration_manifest.json"
        if args.calibration_jsonl is None and cache_manifest.exists():
            cached = json.loads(cache_manifest.read_text())
            paths = [Path(cached[f"{split}_jsonl"]) for split in ("calibration", "heldout")]
            if all(path.is_file() for path in paths):
                args.calibration_jsonl, args.heldout_jsonl = paths
        if args.calibration_jsonl is None and not cache_manifest.exists():
            prepare_default_texts(args)
        cache_ready = cache_complete(args, select(args.layers, 24))
        # Separate processes release all GPU allocations between cases.
        import subprocess
        common = [sys.executable, str(Path(__file__).resolve()), "--model-path", str(args.model_path),
                  "--output-dir", str(args.output_dir), "--checkpoint-dir", str(args.checkpoint_dir),
                  "--cache-dir", str(args.cache_dir), "--layers", args.layers,
                  "--qjl-d-factor", str(args.qjl_d_factor), "--group-size", str(args.group_size),
                  "--pairing", args.pairing, "--ranks", ",".join(map(str, args.ranks)),
                  "--projection-seed", str(args.projection_seed), "--pairing-seed", str(args.pairing_seed),
                  "--bundle-seed", str(args.bundle_seed), "--ridge", str(args.ridge),
                  "--svd-method", args.svd_method,
                  "--svd-oversample", str(args.svd_oversample),
                  "--svd-niter", str(args.svd_niter), "--svd-seed", str(args.svd_seed),
                  "--row-chunk", str(args.row_chunk),
                  "--refine-row-chunk", str(args.refine_row_chunk),
                  "--refine-sweeps", str(args.refine_sweeps), "--refine-block", str(args.refine_block),
                  "--task", args.task, "--eval-limit", str(args.eval_limit),
                  "--eval-batch-size", str(args.eval_batch_size), "--backend", args.backend,
                  "--cases", ",".join(args.cases),
                  "--calibration-backend", args.calibration_backend,
                  "--hidden-cache", args.hidden_cache]
        if not args.length_bucketing:
            common.append("--no-length-bucketing")
        if args.profile:
            common.append("--profile")
        if args.calibration_jsonl is not None:
            common += ["--calibration-jsonl", str(args.calibration_jsonl)]
        if args.heldout_jsonl is not None:
            common += ["--heldout-jsonl", str(args.heldout_jsonl)]
        common += ["--calibration-size", str(args.calibration_size),
                   "--heldout-size", str(args.heldout_size),
                   "--calibration-batch-size", str(args.calibration_batch_size),
                   "--max-length", str(args.max_length)]
        if args.refine_all_bits:
            common.append("--refine-all-bits")
        if args.resume:
            common.append("--resume")
        if not cache_ready:
            if args.calibration_jsonl is None or args.heldout_jsonl is None:
                raise FileNotFoundError("calibration cache missing; pass both JSONL paths or run calibrate")
            subprocess.run(common + ["--mode", "calibrate"], check=True)
        subprocess.run(common + ["--mode", "compress"], check=True)
        if not args.skip_eval:
            for case in args.cases:
                ranks = args.ranks if case in ("C", "D", "E") else [0]
                for rank in ranks:
                    if args.resume and completed_eval_matches(args, case, rank):
                        continue
                    subprocess.run(common + ["--mode", "eval", "--case", case,
                                             "--rank", str(rank)], check=True)
        subprocess.run(common + ["--mode", "report"], check=True)
        return
    if args.mode == "calibrate":
        cache_manifest = args.cache_dir / "calibration_manifest.json"
        if args.calibration_jsonl is None and cache_manifest.exists():
            cached = json.loads(cache_manifest.read_text())
            paths = [Path(cached[f"{split}_jsonl"]) for split in ("calibration", "heldout")]
            if all(path.is_file() for path in paths):
                args.calibration_jsonl, args.heldout_jsonl = paths
        if args.calibration_jsonl is None:
            prepare_default_texts(args)
        if args.resume and cache_complete(args, select(args.layers, 24)):
            print("[calibrate] compatible covariance cache reused", flush=True)
            return
    if args.mode == "eval" and args.resume:
        rank = args.rank
        if args.case in ("C", "D", "E") and not args.explicit_rank:
            rank = max(json.loads((args.checkpoint_dir / "manifest.json").read_text())["ranks"])
        if completed_eval_matches(args, args.case, rank):
            print(f"[eval] compatible result reused: {eval_result_path(args, args.case, rank)}",
                  flush=True)
            return
    if args.mode == "compress":
        if not cache_complete(args, select(args.layers, 24)):
            raise FileNotFoundError("compatible calibration and held-out covariance shards are required")
    model, tokenizer = load_model(args.model_path, args.gpu_memory_fraction)
    layers = select(args.layers, len(model.model.layers))
    if args.mode == "calibrate":
        calibrate(args, model, tokenizer, layers)
    elif args.mode == "compress":
        compress(args, model, layers)
    else:
        evaluate(args, model, tokenizer, layers)


if __name__ == "__main__":
    main()
