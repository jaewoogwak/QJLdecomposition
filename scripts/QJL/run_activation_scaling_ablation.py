#!/usr/bin/env python3
"""Compare original row-L2 and optimized row scales on fixed A/B bundle codes."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from diagnose_codebook_base import atomic_json, select
from rescode.evaluation import evaluate_fewshot
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.activation_bundle_model import checkpoint_sizes, load_checkpoint
from rescode.qjl.activation_scaling_ablation import (
    layer_metrics, replace_row_scales, validate_source,
)
from run_activation_bundle import _model_identity
from run_base_bundling import load_model, save_csv, save_torch


CASES = ("A0", "A", "B0", "B")
TASKS = "arc_easy,arc_challenge"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "compress", "eval", "report"),
                        default="all")
    parser.add_argument("--case", choices=("A0", "B0"))
    parser.add_argument("--model-path", type=Path,
                        default=Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat"))
    parser.add_argument("--source-checkpoint-dir", type=Path, default=Path(
        "results/activation_bundle/optimized_exact_k3_rank_sweep/checkpoint"))
    parser.add_argument("--baseline-results-dir", type=Path, default=Path(
        "results/activation_bundle/optimized_exact_k3_r256"))
    parser.add_argument("--cache-dir", type=Path, default=Path(
        "results/activation_bundle/shared_cache_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "results/activation_bundle/scaling_ablation"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--backend", choices=("triton", "reference"), default="triton")
    parser.add_argument("--task", default=TASKS)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--gpu-memory-fraction", type=float, default=.94)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "eval" and args.case is None:
        parser.error("--case A0 or B0 is required in eval mode")
    if (args.eval_batch_size < 1 or args.eval_limit < 0 or
            not 0 < args.gpu_memory_fraction <= 1):
        parser.error("invalid evaluation or GPU memory option")
    for name in ("model_path", "source_checkpoint_dir", "baseline_results_dir",
                 "cache_dir", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.checkpoint_dir = (args.checkpoint_dir or
                           args.output_dir / "checkpoint").resolve()
    return args


def _sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(args, layers):
    source_path = args.source_checkpoint_dir / "manifest.json"
    baseline_path = args.baseline_results_dir / "checkpoint/manifest.json"
    cache_path = args.cache_dir / "calibration_manifest.json"
    source = json.loads(source_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    cache = json.loads(cache_path.read_text())
    validate_source(source, baseline, cache, layers)
    if str(args.model_path) != source["model_path"]:
        raise ValueError("model path differs from the source checkpoint")
    if _model_identity(args.model_path) != cache["model_identity"]:
        raise ValueError("model revision differs from the calibration cache")
    for split in ("calibration", "heldout"):
        dataset = Path(cache[f"{split}_jsonl"])
        if not dataset.is_file() or _sha256(dataset) != cache[f"{split}_sha256"]:
            raise ValueError(f"{split} JSONL differs from the calibration cache")
        for layer in layers:
            if not (args.cache_dir / f"{split}_layer_{layer}.pt").is_file():
                raise FileNotFoundError(f"{split} covariance layer {layer} is missing")
    for layer in layers:
        for case in ("A", "B"):
            path = args.source_checkpoint_dir / f"base_{case}_layer_{layer}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
    for case in ("A", "B"):
        path = args.baseline_results_dir / case / "eval_metrics.json"
        metrics = json.loads(path.read_text())
        request = (metrics["task"], metrics["eval_limit"],
                   metrics["eval_batch_size"], metrics["backend"])
        wanted = (args.task, args.eval_limit, args.eval_batch_size, args.backend)
        if request != wanted:
            raise ValueError(f"existing {case} evaluation settings differ: {request} != {wanted}")
    return source, _sha256(source_path), _sha256(baseline_path)


def ablation_manifest(args, layers):
    source, source_hash, baseline_hash = provenance(args, layers)
    config = dict(source)
    config.update({"layers": layers, "rank": 0, "ranks": [0], "cases": ["A", "B"],
                   "scaling_ablation": "original_weight_row_l2",
                   "source_checkpoint_dir": str(args.source_checkpoint_dir),
                   "source_manifest_sha256": source_hash,
                   "baseline_manifest_sha256": baseline_hash})
    return config


def _layer_files(root, layer):
    return [root / f"base_{case}_layer_{layer}.pt" for case in ("A", "B")] + [
        root / f"per_expert_layer_{layer}.csv"]


@torch.no_grad()
def compress(args, layers):
    config = ablation_manifest(args, layers)
    root = args.checkpoint_dir
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != config:
            raise ValueError("ablation checkpoint config differs; choose another output directory")
        if not args.resume:
            raise FileExistsError("checkpoint exists; pass --resume")
    else:
        atomic_json(manifest_path, config)
    model, _ = load_model(args.model_path, args.gpu_memory_fraction)
    banks = {}
    with tqdm(layers, desc="[scaling]", unit="layer", dynamic_ncols=True) as progress:
        for layer in progress:
            progress.set_postfix_str(f"layer {layer}", refresh=False)
            block = model.model.layers[layer].mlp
            if args.resume and all(path.is_file() for path in _layer_files(root, layer)):
                block.experts = nn.ModuleList(nn.Identity() for _ in block.experts)
                continue
            states = {case: torch.load(args.source_checkpoint_dir /
                        f"base_{case}_layer_{layer}.pt", map_location="cpu",
                        weights_only=False) for case in ("A", "B")}
            replace_row_scales(states, block)
            calibration = torch.load(args.cache_dir / f"calibration_layer_{layer}.pt",
                                     map_location="cpu", weights_only=False)
            heldout = torch.load(args.cache_dir / f"heldout_layer_{layer}.pt",
                                 map_location="cpu", weights_only=False)
            rows = layer_metrics(block, states, calibration, heldout, config, banks, layer)
            for case, state in states.items():
                save_torch(root / f"base_{case}_layer_{layer}.pt", state)
            save_csv(root / f"per_expert_layer_{layer}.csv", rows)
            block.experts = nn.ModuleList(nn.Identity() for _ in block.experts)
            del states, calibration, heldout, rows
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report(args, layers)


def evaluation_signature(args, case, manifest_path):
    return {"case": case, "task": args.task, "eval_limit": args.eval_limit,
            "eval_batch_size": args.eval_batch_size, "backend": args.backend,
            "model_path": str(args.model_path),
            "checkpoint_manifest_sha256": _sha256(manifest_path)}


def evaluate(args, layers):
    case = args.case
    manifest_path = args.checkpoint_dir / "manifest.json"
    signature = evaluation_signature(args, case, manifest_path)
    output = args.output_dir / case / "eval_metrics.json"
    if output.is_file():
        old = json.loads(output.read_text())
        if old.get("request_signature") != signature:
            raise ValueError(f"saved evaluation settings differ: {output}")
        if args.resume:
            print(f"[eval] reused {output}", flush=True)
            return
        raise FileExistsError(f"evaluation exists; pass --resume: {output}")
    model, tokenizer = load_model(args.model_path, args.gpu_memory_fraction)
    base_case = case[0]
    config = load_checkpoint(model, args.checkpoint_dir, base_case, args.backend, 0)
    if config["layers"] != layers or config.get("scaling_ablation") != "original_weight_row_l2":
        raise ValueError("loaded checkpoint is not this row-L2 ablation")
    storage = checkpoint_sizes(args.checkpoint_dir, config, base_case, 0)
    case_dir = output.parent
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = evaluate_fewshot(model, tokenizer, args.task, 0,
        eval_batch_size=args.eval_batch_size, output_path=str(case_dir / "lm_eval.txt"),
        limit=args.eval_limit or None)
    seconds = time.perf_counter() - started
    summary = report_eval_summary(result, args.task, str(case_dir / "lm_eval.txt"),
                                  header=f"Scaling ablation {case}")
    atomic_json(output, {"case": case, "rank": 0, "scaling": "original_weight_row_l2",
                         "request_signature": signature, "task": args.task,
                         "eval_limit": args.eval_limit,
                         "eval_batch_size": args.eval_batch_size, "backend": args.backend,
                         "evaluation_wall_seconds": seconds, "task_summary": summary,
                         "task_results": result.get("results", {}), "storage": storage,
                         "gpu_process_peak_allocated_bytes": [
                             torch.cuda.max_memory_allocated(i)
                             for i in range(torch.cuda.device_count())]})
    print(f"[eval] saved {output}", flush=True)


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    return None if value in (None, "", "None") else float(value)


def _mean(rows, key):
    values = [_float(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _baseline_rows(args, layers):
    old = _read_csv(args.baseline_results_dir / "per_expert.csv")
    selected = set(layers)
    result = []
    for row in old:
        if row["case"] not in ("A", "B") or int(row["rank"]) != 0 or int(row["layer"]) not in selected:
            continue
        cal = _float(row["activation_loss"])
        held = _float(row["heldout_activation_loss"])
        result.append({"layer": int(row["layer"]), "projection": row["projection"],
                       "expert": int(row["expert"]), "group": int(row["group"]),
                       "case": row["case"], "rank": 0,
                       "scaling": "activation_optimized",
                       "weight_relative_error": _float(row["weight_relative_error"]),
                       "calibration_activation_loss": cal,
                       "heldout_activation_loss": held,
                       "calibration_relative_l2_error": math.sqrt(cal) if cal is not None else None,
                       "heldout_relative_l2_error": math.sqrt(held) if held is not None else None,
                       "calibration_count": int(row["calibration_count"]),
                       "heldout_count": None})
    if len(result) != len(layers) * 3 * 60 * 2:
        raise ValueError("existing A/B per-expert metrics are incomplete")
    return result


def _summary_row(case, metrics, rows, storage):
    summary = metrics["task_summary"]
    tasks = {task: (acc, norm) for task, acc, norm in summary["tasks"]}
    return {"case": case, "rank": 0, "scaling": ("original_weight_row_l2" if case.endswith("0")
                                                else "activation_optimized"),
            "average_accuracy": summary["avg"], "average_acc_norm": summary["avg_norm"],
            "arc_easy_acc": tasks["arc_easy"][0], "arc_easy_acc_norm": tasks["arc_easy"][1],
            "arc_challenge_acc": tasks["arc_challenge"][0],
            "arc_challenge_acc_norm": tasks["arc_challenge"][1],
            "mean_weight_relative_error": _mean(rows, "weight_relative_error"),
            "mean_calibration_activation_loss": _mean(rows, "calibration_activation_loss"),
            "mean_heldout_activation_loss": _mean(rows, "heldout_activation_loss"),
            "mean_calibration_relative_l2_error": _mean(rows, "calibration_relative_l2_error"),
            "mean_heldout_relative_l2_error": _mean(rows, "heldout_relative_l2_error"),
            "shared_sketch_bytes": storage["shared_sketch_bytes"],
            "row_scale_bytes": storage["row_scale_bytes"],
            "actual_checkpoint_bytes": storage["actual_checkpoint_bytes"],
            "effective_bpw": storage["effective_bpw"],
            "bf16_storage_reduction": storage["bf16_storage_reduction"]}


def report(args, layers):
    root = args.checkpoint_dir
    if not (root / "manifest.json").is_file():
        return
    new_rows = []
    for layer in layers:
        path = root / f"per_expert_layer_{layer}.csv"
        if not path.is_file():
            return
        new_rows.extend(_read_csv(path))
    old_rows = _baseline_rows(args, layers)
    all_rows = new_rows + old_rows
    all_rows.sort(key=lambda row: (int(row["layer"]), row["projection"],
                                   int(row["expert"]), CASES.index(row["case"])))
    save_csv(args.output_dir / "per_expert.csv", all_rows)
    old_summary = {row["case"]: row for row in _read_csv(
        args.baseline_results_dir / "summary.csv") if row["case"] in ("A", "B")}
    if set(old_summary) != {"A", "B"}:
        raise ValueError("existing A/B summary is incomplete")
    summaries = []
    for case in CASES:
        rows = [row for row in all_rows if row["case"] == case]
        if len(rows) != len(layers) * 3 * 60:
            raise ValueError(f"{case} per-expert metrics are incomplete")
        if case.endswith("0"):
            path = args.output_dir / case / "eval_metrics.json"
            if not path.is_file():
                continue
            metrics = json.loads(path.read_text())
            config = json.loads((root / "manifest.json").read_text())
            storage = checkpoint_sizes(root, config, case[0], 0)
        else:
            metrics = json.loads((args.baseline_results_dir / case / "eval_metrics.json").read_text())
            storage = {key: _float(old_summary[case][key]) for key in (
                "shared_sketch_bytes", "row_scale_bytes", "actual_checkpoint_bytes",
                "effective_bpw", "bf16_storage_reduction")}
        summaries.append(_summary_row(case, metrics, rows, storage))
    if not summaries:
        return
    save_csv(args.output_dir / "summary.csv", summaries)
    by_case = {row["case"]: row for row in summaries}
    deltas = {}
    for original, optimized in (("A0", "A"), ("B0", "B")):
        if original in by_case and optimized in by_case:
            deltas[original + "_to_" + optimized] = {
                key: by_case[optimized][key] - by_case[original][key]
                for key in ("average_accuracy", "average_acc_norm",
                            "mean_weight_relative_error",
                            "mean_calibration_activation_loss",
                            "mean_heldout_activation_loss")}
    atomic_json(args.output_dir / "summary.json", {
        "experiment": "fixed-code row-scaling ablation", "layers": layers,
        "group_size": 3, "qjl_d_factor": 2, "rank": 0,
        "source_checkpoint_dir": str(args.source_checkpoint_dir),
        "baseline_results_dir": str(args.baseline_results_dir),
        "cache_dir": str(args.cache_dir), "cases": by_case, "optimized_minus_row_l2": deltas})
    lines = ["# Activation-aware bundled Base: row scaling ablation", "",
             "A0/A share majority-vote codes; B0/B share activation-aware codes.",
             "Only the expert-specific row scale changes. A/B evaluation and",
             "per-expert measurements are reused from `optimized_exact_k3_r256`.", "",
             "All metrics below use 24 layers, group size 3, D=2n, rank 0,",
             "zero-shot full ARC-Easy and ARC-Challenge. Error columns are",
             "unweighted means over layer × projection × expert.", "",
             "| Case | Row scale | Mean acc | Mean acc_norm | Weight rel. L2 | Cal. loss | Heldout loss | Cal. rel. L2 | Heldout rel. L2 | BPW |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        if case not in by_case:
            continue
        row = by_case[case]
        lines.append("| " + " | ".join([
            case, row["scaling"],
            *(f"{row[key]:.4f}" for key in (
                "average_accuracy", "average_acc_norm", "mean_weight_relative_error",
                "mean_calibration_activation_loss", "mean_heldout_activation_loss",
                "mean_calibration_relative_l2_error", "mean_heldout_relative_l2_error",
                "effective_bpw"))]) + " |")
    lines += ["", "Per-task accuracy and storage bytes are in `summary.csv`/`summary.json`.",
              "`per_expert.csv` contains the underlying errors. New A0/B0 packed",
              "checkpoints are in `checkpoint/`; covariance caches and the",
              "original model are excluded from BPW.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(lines))


def main(argv=None):
    args = arguments(argv)
    layers = select(args.layers, 24)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "all":
        common = [sys.executable, str(Path(__file__).resolve()),
                  "--model-path", str(args.model_path),
                  "--source-checkpoint-dir", str(args.source_checkpoint_dir),
                  "--baseline-results-dir", str(args.baseline_results_dir),
                  "--cache-dir", str(args.cache_dir),
                  "--output-dir", str(args.output_dir),
                  "--checkpoint-dir", str(args.checkpoint_dir),
                  "--layers", args.layers, "--backend", args.backend,
                  "--task", args.task, "--eval-batch-size", str(args.eval_batch_size),
                  "--eval-limit", str(args.eval_limit),
                  "--gpu-memory-fraction", str(args.gpu_memory_fraction)]
        if args.resume:
            common.append("--resume")
        subprocess.run(common + ["--mode", "compress"], check=True)
        for case in ("A0", "B0"):
            subprocess.run(common + ["--mode", "eval", "--case", case], check=True)
        subprocess.run(common + ["--mode", "report"], check=True)
        return
    if args.mode == "compress":
        compress(args, layers)
    elif args.mode == "eval":
        provenance(args, layers)
        evaluate(args, layers)
        report(args, layers)
    else:
        provenance(args, layers)
        report(args, layers)


if __name__ == "__main__":
    main()
