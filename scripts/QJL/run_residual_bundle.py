#!/usr/bin/env python3
"""Compress or evaluate pairwise Exponent-Mantissa Base + QJL residuals."""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diagnose_codebook_base import atomic_json, select
from rescode.evaluation import evaluate_fewshot
from rescode.merging.eval_summary import report_eval_summary
from rescode.qjl.residual_bundle_model import (
    checkpoint_sizes, compress_model, gpu_resident_breakdown, load_checkpoint,
    save_checkpoint)

MODEL_PATH = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B-Chat"
              / "snapshots/ec052fda178e241c7c443468d2fa1db6618996be")
TASKS = "winogrande,arc_easy,arc_challenge,boolq,hellaswag,mmlu,openbookqa,rte"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("compress", "eval"))
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--bundle-method", choices=("none", "majority", "magnitude-aware"),
                        default="none")
    parser.add_argument("--pairing", choices=("random", "similarity"), default="similarity")
    parser.add_argument("--qjl-d-factor", type=float, default=2.0)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--pairing-seed", type=int, default=0)
    parser.add_argument("--bundle-seed", type=int, default=0)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qjl_residual_bundle/run")
    parser.add_argument("--backend", choices=("triton", "reference"), default="triton")
    parser.add_argument("--task", default=TASKS)
    parser.add_argument("--eval-limit", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    args = parser.parse_args(argv)
    if args.mode == "eval" and args.checkpoint is None:
        parser.error("--checkpoint is required for eval")
    if args.qjl_d_factor <= 0 or args.row_chunk <= 0 or args.eval_limit < 0:
        parser.error("QJL factor and row chunk must be positive; eval limit must be nonnegative")
    if args.eval_batch_size < 1 or not 0 < args.gpu_memory_fraction <= 1:
        parser.error("invalid eval batch size or GPU memory fraction")
    return args


def load_model(source, fraction):
    if not (source / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"model snapshot missing: {source}")
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * fraction)
                      for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
    model = AutoModelForCausalLM.from_pretrained(
        str(source), torch_dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True, max_memory=max_memory)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
    return model, tokenizer


def run_eval(model, tokenizer, args, out, metrics):
    if args.skip_eval:
        return
    path = out / "lm_eval.txt"
    result = evaluate_fewshot(model, tokenizer, args.task, 0,
                              eval_batch_size=args.eval_batch_size, output_path=str(path),
                              limit=args.eval_limit or None)
    metrics["task_summary"] = report_eval_summary(
        result, args.task, str(path), header="Pairwise residual bundle evaluation")
    metrics["task_results"] = result.get("results", {})
    atomic_json(out / "accuracy.json", metrics["task_results"])


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.model_path.expanduser().resolve()
    model, tokenizer = load_model(source, args.gpu_memory_fraction)
    metrics = {"mode": args.mode, "source": str(source), "backend": args.backend,
               "task": args.task, "eval_limit": args.eval_limit,
               "eval_batch_size": args.eval_batch_size}
    checkpoint = args.checkpoint or out / "compact_qjl_state.pt"
    if args.mode == "compress":
        layers = select(args.layers, len(model.model.layers))
        config = {"format": "rescode-residual-pair-bundle-v1", "layers": layers,
                  "bundle_method": args.bundle_method, "pairing": args.pairing,
                  "qjl_d_factor": args.qjl_d_factor, "projection_seed": args.projection_seed,
                  "pairing_seed": args.pairing_seed, "bundle_seed": args.bundle_seed,
                  "row_chunk": args.row_chunk, "backend": args.backend,
                  "model": str(source)}
        started = time.perf_counter()
        state, records = compress_model(model, config)
        metrics["compression_wall_seconds"] = time.perf_counter() - started
        sizes = save_checkpoint(state, config, checkpoint)
        metrics.update(sizes)
        weight_count = sum(r["rows"] * r["cols"] for r in records)
        metrics.update({"weight_count": weight_count,
                        "checkpoint_bits_per_weight": 8 * sizes["checkpoint_bytes"] / weight_count,
                        "bf16_compression_ratio": 2 * weight_count / sizes["checkpoint_bytes"],
                        "base_bits_per_weight": 8 * sizes["base_tensor_bytes"] / weight_count,
                        "qjl_sketch_bits_per_weight": 8 * sizes["qjl_sketch_bytes"] / weight_count,
                        "norm_bits_per_weight": 8 * sizes["norm_bytes"] / weight_count})
        atomic_json(out / "storage.json", {**sizes, "weight_count": weight_count,
                                            "checkpoint_bits_per_weight": metrics["checkpoint_bits_per_weight"],
                                            "bf16_compression_ratio": metrics["bf16_compression_ratio"]})
        if not args.skip_eval:
            run_eval(model, tokenizer, args, out, metrics)
    else:
        payload = load_checkpoint(model, checkpoint, backend=args.backend)
        config = payload["config"]
        metrics["checkpoint_config"] = config
        metrics.update(checkpoint_sizes(payload["layers"], checkpoint))
        run_eval(model, tokenizer, args, out, metrics)
    metrics["checkpoint"] = str(checkpoint)
    metrics.update(gpu_resident_breakdown(model))
    if torch.cuda.is_available():
        metrics["gpu_process_allocated_bytes"] = [
            torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
        metrics["gpu_process_peak_allocated_bytes"] = [
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    atomic_json(out / "metrics.json", metrics)
    print(f"[bundle] saved {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
