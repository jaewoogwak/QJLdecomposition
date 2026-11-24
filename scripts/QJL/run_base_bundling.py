#!/usr/bin/env python3
"""Compress or evaluate pairwise QJL Base with an optional BF16 SVD residual."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from diagnose_codebook_base import atomic_json, select
from rescode.evaluation import evaluate_fewshot
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.base_bundle_model import (FORMAT, compress_layer, checkpoint_sizes,
                                            load_checkpoint, resident_bytes)

TASKS = "arc_easy,arc_challenge"
ALL_TASKS = "winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("compress", "eval"))
    p.add_argument("--eval-variant", choices=("original", "base-only", "base-plus-lowrank"),
                   help="Evaluation model: original BF16, shared QJL Base, or Base plus SVD residual")
    p.add_argument("--case", dest="legacy_case", choices=("original", "A", "B"),
                   help=argparse.SUPPRESS)
    p.add_argument("--model-path", type=Path,
                   default=Path("/home/jaewoo/models/Qwen1.5-MoE-A2.7B-Chat"))
    p.add_argument("--layers", default="all")
    p.add_argument("--qjl-d-factor", type=float, default=2.0)
    p.add_argument("--rank", type=int, default=512)
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--pairing-seed", type=int, default=0)
    p.add_argument("--bundle-seed", type=int, default=0)
    p.add_argument("--row-chunk", type=int, default=128)
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/base_bundling/pairwise_d2n_r512"))
    p.add_argument("--backend", choices=("triton", "reference"), default="triton")
    p.add_argument("--task", default=TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-fraction", type=float, default=.94)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)
    legacy_variants = {"original": "original", "A": "base-only",
                       "B": "base-plus-lowrank"}
    if args.legacy_case is not None:
        legacy = legacy_variants[args.legacy_case]
        if args.eval_variant is not None and args.eval_variant != legacy:
            p.error("--case and --eval-variant select different evaluation models")
        args.eval_variant = legacy
    if args.mode == "eval" and args.eval_variant is None:
        p.error("--eval-variant is required in eval mode")
    if args.mode == "compress" and args.eval_variant is not None:
        p.error("compress creates the common Base and rank factor shards; omit --eval-variant")
    args.case = {"original": "original", "base-only": "A",
                 "base-plus-lowrank": "B"}.get(args.eval_variant)
    if (args.qjl_d_factor <= 0 or args.rank < 1 or args.row_chunk < 1
            or args.eval_limit < 0 or args.eval_batch_size < 1
            or not 0 < args.gpu_memory_fraction <= 1):
        p.error("invalid dimension, rank, row chunk, evaluation, or memory option")
    return args


def load_model(source: Path, memory_fraction: float):
    if not (source / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"model snapshot missing: {source}")
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * memory_fraction)
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    model = AutoModelForCausalLM.from_pretrained(str(source), torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True,
        max_memory=max_memory)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
    return model, tokenizer


def save_torch(path: Path, payload):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def save_spectra(path: Path, spectra: dict):
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **spectra)
    os.replace(tmp, path)


def read_rows(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_plots(checkpoint_dir: Path, layers: list[int], out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spectra = {key: [] for key in ("gate", "up", "down")}
    for layer in layers:
        with np.load(checkpoint_dir / f"spectrum_layer_{layer}.npz") as data:
            for name in data.files:
                key = name.split("_", 1)[0]
                values = data[name].astype(np.float64)
                energy = values ** 2
                spectra[key].append(energy / max(energy.sum(), 1e-30))
    out.mkdir(parents=True, exist_ok=True)
    for name, transform, ylabel in (("spectrum.png", lambda x: x,
                                     "Mean normalized singular energy"),
                                    ("cumulative_energy.png", np.cumsum,
                                     "Mean cumulative energy")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, arrays in spectra.items():
            mean = np.mean(np.stack([transform(a) for a in arrays]), axis=0)
            ax.plot(np.arange(1, len(mean) + 1), mean, label=key)
        if name == "spectrum.png":
            ax.set_yscale("log")
        ax.set_xlabel("Rank")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)


def collect_analysis(checkpoint_dir: Path, config: dict, out: Path):
    rows = [row for layer in config["layers"] for row in
            read_rows(checkpoint_dir / f"per_expert_layer_{layer}.csv")]
    expected = 60 * 3 * len(config["layers"])
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} matrix records, got {len(rows)}")
    save_csv(out / "per_expert.csv", rows)
    write_plots(checkpoint_dir, config["layers"], out / "plots")
    return rows


def compress(model, args, checkpoint_dir: Path, out: Path):
    layers = select(args.layers, len(model.model.layers))
    config = {"format": FORMAT, "model_path": str(args.model_path.resolve()),
              "layers": layers, "qjl_d_factor": args.qjl_d_factor, "rank": args.rank,
              "projection_seed": args.projection_seed, "pairing_seed": args.pairing_seed,
              "bundle_seed": args.bundle_seed, "row_chunk": args.row_chunk}
    manifest = checkpoint_dir / "manifest.json"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        if previous != config:
            raise ValueError("checkpoint manifest settings differ; use another output directory")
        if not args.resume:
            raise FileExistsError("checkpoint exists; pass --resume to verify completed layers")
    else:
        atomic_json(manifest, config)
    banks = {}
    model._base_bundle_banks = banks
    for layer in layers:
        paths = [checkpoint_dir / f"base_layer_{layer}.pt",
                 checkpoint_dir / f"rank_layer_{layer}.pt",
                 checkpoint_dir / f"per_expert_layer_{layer}.csv",
                 checkpoint_dir / f"spectrum_layer_{layer}.npz"]
        if args.resume and all(path.is_file() for path in paths):
            model.model.layers[layer].mlp.experts = nn.ModuleList(nn.Identity() for _ in range(60))
            print(f"[base-bundle] layer {layer}: resumed", flush=True)
            continue
        start = time.perf_counter()
        base, factors, records, spectra = compress_layer(
            model.model.layers[layer].mlp, layer, config, banks)
        save_torch(paths[0], base)
        save_torch(paths[1], factors)
        save_csv(paths[2], records)
        save_spectra(paths[3], spectra)
        # Compression and evaluation are separate processes. Free original
        # routed weights after their layer has been safely checkpointed.
        model.model.layers[layer].mlp.experts = nn.ModuleList(nn.Identity() for _ in range(60))
        del base, factors, records, spectra
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[base-bundle] layer {layer}: 180 matrices in "
              f"{time.perf_counter() - start:.1f}s", flush=True)
    rows = collect_analysis(checkpoint_dir, config, out)
    for case in ("A", "B"):
        atomic_json(out / f"storage_{case}.json", checkpoint_sizes(checkpoint_dir, config, case))
    return config, rows


def original_storage(model, source: Path, layers: list[int]):
    count = sum(expert.gate_proj.weight.numel() + expert.up_proj.weight.numel()
                + expert.down_proj.weight.numel()
                for layer in layers for expert in model.model.layers[layer].mlp.experts)
    disk = sum(path.stat().st_size for path in source.glob("*.safetensors"))
    return {"routed_weight_count": count, "routed_bf16_bytes": 2 * count,
            "actual_checkpoint_bytes": disk, "effective_bpw": 16.0,
            "bf16_storage_reduction": 0.0, "shared_sketch_bytes": 0,
            "row_norm_bytes": 0, "lowrank_residual_bytes": 0}


def summarize_evaluations(out: Path):
    rows = []
    for case in ("original", "A", "B"):
        path = out / case / "eval_metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        summary = metrics["task_summary"]
        row = {"case": case, "average_accuracy": summary["avg"],
               "average_acc_norm": summary["avg_norm"],
               "evaluation_wall_seconds": metrics["evaluation_wall_seconds"]}
        for task, acc, norm in summary["tasks"]:
            row[f"{task}_acc"] = acc
            row[f"{task}_acc_norm"] = norm
        row.update(metrics["storage"])
        rows.append(row)
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        save_csv(out / "summary.csv", [{key: row.get(key) for key in fields}
                                        for row in rows])
        atomic_json(out / "summary.json", rows)
    return rows


def write_report(out: Path, config: dict | None, rows: list[dict]):
    import statistics
    lines = ["# Pairwise QJL Base + low-rank residual", "",
             "A uses one shared packed sketch per pair and expert-specific FP32 norms.",
             "B adds BF16 SVD factors for the residual after bundling.", ""]
    if config:
        lines += [f"Layers: {config['layers']}; D factor: {config['qjl_d_factor']}; "
                  f"rank: {config['rank']}.", ""]
    if rows:
        lines += ["| Case | Mean acc | Mean acc_norm | Checkpoint bytes | Effective BPW |",
                  "|---|---:|---:|---:|---:|"]
        for row in rows:
            size = row["actual_checkpoint_bytes"]
            lines.append(f"| {row['case']} | {row['average_accuracy']:.4f} | "
                         f"{row['average_acc_norm']:.4f} | {size} | "
                         f"{row['effective_bpw']:.3f} |")
        lines += ["", "Original checkpoint bytes count all model safetensors; A/B "
                  "checkpoint bytes count only compressed routed layers. "
                  "Effective BPW and BF16 reduction use routed weights in the selected layers.", ""]
    else:
        lines += ["Evaluation has not been run.", ""]
    analysis_path = out / "per_expert.csv"
    if analysis_path.is_file():
        analysis = read_rows(analysis_path)
        lines += [f"Analyzed {len(analysis)} residual matrices. Means across matrices:", ""]
        for field, label in (("residual_frobenius_ratio", "Residual/weight Frobenius ratio"),
                             ("pre_bundle_bit_agreement", "Pre-bundle bit agreement"),
                             ("post_bundle_bit_preservation", "Post-bundle bit preservation"),
                             ("energy_rank_512", "Rank-512 FP32 residual energy retention"),
                             ("optimal_weight_error_rank_512", "Optimal rank-512 weight error"),
                             ("bf16_factor_weight_error", "BF16 factor weight error")):
            lines.append(f"- {label}: {statistics.fmean(float(r[field]) for r in analysis):.6f}")
        lines += [""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main(argv=None):
    args = arguments(argv)
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (args.checkpoint_dir or out / "checkpoint").resolve()
    source = args.model_path.resolve()
    model, tokenizer = load_model(source, args.gpu_memory_fraction)
    if args.mode == "compress":
        config, _ = compress(model, args, checkpoint_dir, out)
        write_report(out, config, summarize_evaluations(out))
        return
    if args.case == "original":
        layers = select(args.layers, len(model.model.layers))
        storage = original_storage(model, source, layers)
        config = None
        runtime = {"gpu_shared_sketch_bytes": 0, "gpu_row_norm_bytes": 0,
                   "gpu_lowrank_bytes": 0, "gpu_projection_bytes": 0}
    else:
        config = load_checkpoint(model, checkpoint_dir, args.case, args.backend)
        storage = checkpoint_sizes(checkpoint_dir, config, args.case)
        runtime = None
    case_dir = out / args.case
    case_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = evaluate_fewshot(model, tokenizer, args.task, 0,
        eval_batch_size=args.eval_batch_size, output_path=str(case_dir / "lm_eval.txt"),
        limit=args.eval_limit or None)
    elapsed = time.perf_counter() - start
    if runtime is None:
        runtime = resident_bytes(model)
    metrics = {"case": args.case, "eval_variant": args.eval_variant,
               "source": str(source),
               "checkpoint_dir": str(checkpoint_dir) if config else None,
               "task": args.task, "eval_limit": args.eval_limit,
               "eval_batch_size": args.eval_batch_size, "backend": args.backend,
               "evaluation_wall_seconds": elapsed,
               "task_summary": report_eval_summary(result, args.task,
                   str(case_dir / "lm_eval.txt"), header=f"Base bundle case {args.case}"),
               "task_results": result.get("results", {}),
               "storage": storage, "runtime": runtime,
               "gpu_process_allocated_bytes": (
                   [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
                   if torch.cuda.is_available() else []),
               "gpu_process_peak_allocated_bytes": (
                   [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
                   if torch.cuda.is_available() else [])}
    atomic_json(case_dir / "eval_metrics.json", metrics)
    rows = summarize_evaluations(out)
    write_report(out, config, rows)
    print(f"[base-bundle] saved {case_dir / 'eval_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
