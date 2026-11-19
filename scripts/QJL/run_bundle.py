#!/usr/bin/env python3
"""Compress/evaluate ResCode-grouped majority QJL with low-rank corrections."""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rescode.evaluation import evaluate_fewshot, get_calib_dataloder
from rescode.merging.calib_stats import load_or_collect_calib_stats
from rescode.merging.eval_summary import report_eval_summary
from rescode.merging.expert_clusters import form_clusters, residual_params_fn_for
from rescode.qjl.bundled_model import (checkpoint_sizes, compress_model,
    gpu_resident_bytes, load_checkpoint, save_checkpoint, validate_groups)
from rescode.qjl.calibration import collect_projection_samples


MODEL = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
TASKS = "arc_easy,arc_challenge,hellaswag,winogrande,boolq,mmlu,openbookqa,rte"


def parse_layers(value, n_layers):
    layers = list(range(n_layers)) if value == "all" else sorted({int(s) for s in value.split(",")})
    if not layers or any(x < 0 or x >= n_layers for x in layers):
        raise ValueError(f"invalid layer selection: {value}")
    return layers


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("compress", "eval"), required=True)
    p.add_argument("--model-path", default=MODEL)
    p.add_argument("--groups-path", type=Path)
    p.add_argument("--calib-stats-cache", type=Path)
    p.add_argument("--train-data-path", type=Path, default=Path("rescode/data/wikipedia-calib.jsonl"))
    p.add_argument("--layers", default="all")
    p.add_argument("--num-groups", type=int, default=30)
    p.add_argument("--singleton-ratio", type=float, default=0.15)
    p.add_argument("--rank-for-grouping", type=int, default=128)
    p.add_argument("--linkage", default="average")
    p.add_argument("--budget-aware", choices=("true", "false"), default="true")
    p.add_argument("--n-sentences", type=int, default=2400)
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--calib-seqlen", type=int, default=2048)
    p.add_argument("--calib-batch-size", type=int, default=1)
    p.add_argument("--qjl-d", type=int, default=2048)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--bundle-mode", choices=("majority", "none"), default="majority")
    p.add_argument("--residual-mode", choices=("lowrank", "exact"), default="lowrank")
    p.add_argument("--backend", choices=("triton", "reference", "triton_legacy"), default="triton")
    p.add_argument("--projection-seed", type=int, default=0)
    p.add_argument("--svd-seed", type=int, default=42)
    p.add_argument("--svd-oversample", type=int, default=8)
    p.add_argument("--svd-niter", type=int, default=2)
    p.add_argument("--diagnostic-samples", type=int, default=0,
                   help="Routed calibration samples per expert; 0 skips the extra model pass")
    p.add_argument("--output-dir", type=Path, default=Path("results/qjl_bundle/run"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--task", default=TASKS)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--gpu-memory-fraction", type=float, default=0.94)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()
    if args.mode == "eval" and not args.checkpoint:
        p.error("--checkpoint required in eval mode")
    if args.mode == "compress" and args.residual_mode == "exact" and not args.skip_eval:
        p.error("exact residual is diagnostic-only; use --skip-eval")
    if args.qjl_d < 1 or args.rank < 1 or args.svd_oversample < 0 or args.svd_niter < 0:
        p.error("D/rank must be positive and SVD parameters nonnegative")
    if not 0 <= args.singleton_ratio < 1 or args.num_groups < 2:
        p.error("invalid grouping parameters")
    return args


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, default=str)
    os.replace(tmp, path)


def calibration_loader(tokenizer, args):
    if not args.train_data_path.is_file():
        raise FileNotFoundError(f"calibration file missing: {args.train_data_path}; "
                                "supply --groups-path to reuse grouping or prepare the dataset")
    return get_calib_dataloder(dataset="c4", tokenizer=tokenizer,
        max_block_size=args.calib_seqlen, n_blocks_for_stat=args.n_sentences,
        batch_size=args.calib_batch_size, num_workers=2, seed=args.calib_seed,
        data_path=str(args.train_data_path))


def _compatible_cache(path, args):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    key = saved.get("key", {})
    expected = {"model_name": MODEL, "num_groups": args.num_groups,
                "singleton_ratio": args.singleton_ratio,
                "rank": args.rank_for_grouping, "targets": "gate_up_down",
                "budget_aware": args.budget_aware, "linkage": args.linkage,
                "calib_seed": args.calib_seed}
    if any(key.get(k) != v for k, v in expected.items()):
        raise ValueError(f"group cache settings differ from CLI: {path}")
    return saved


def load_groups(model, tokenizer, args, selected):
    path = args.groups_path
    if path is None:
        candidates = list(Path("results/qjl_direct/shared/qwen").glob("groups-*.pt"))
        compatible = []
        for candidate in candidates:
            try:
                _compatible_cache(candidate, args)
                compatible.append(candidate)
            except ValueError:
                pass
        if len(compatible) > 1:
            raise ValueError(f"multiple compatible group caches: {compatible}; use --groups-path")
        if compatible:
            path = compatible[0]
    if path is not None:
        saved = _compatible_cache(path, args)
        groups = {int(k): v for k, v in saved["groups"].items()}
        print(f"[bundle] reused ResCode grouping: {path}", flush=True)
        return {i: groups[i] for i in selected}, str(path), saved.get("summary")
    loader = calibration_loader(tokenizer, args)
    stats = load_or_collect_calib_stats(model, loader, "qwen",
        str(args.calib_stats_cache) if args.calib_stats_cache else None)
    expert = model.model.layers[selected[0]].mlp.experts[0]
    m, n = expert.gate_proj.weight.shape
    groups, summary = form_clusters(stats, num_groups=args.num_groups,
        singleton_ratio=args.singleton_ratio, expert_params=3 * m * n,
        residual_params_fn=residual_params_fn_for(n, m, args.rank_for_grouping, 3),
        linkage=args.linkage, budget_aware=args.budget_aware == "true")
    path = Path("results/qjl_bundle/grouping/groups.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite grouping: {path}")
    torch.save({"key": {"model_name": MODEL, "num_groups": args.num_groups,
        "singleton_ratio": args.singleton_ratio, "rank": args.rank_for_grouping,
        "targets": "gate_up_down", "budget_aware": args.budget_aware,
        "linkage": args.linkage, "calib_seed": args.calib_seed},
        "groups": groups, "summary": summary,
        "saliency": {k: v["saliency"] for k, v in stats.items()}}, path)
    return {i: groups[i] for i in selected}, str(path), summary


def routed_samples(model, tokenizer, args, groups):
    if args.diagnostic_samples == 0:
        return {}, "disabled"
    loader = calibration_loader(tokenizer, args)
    _, heldout, probes = collect_projection_samples(model, loader, "qwen", groups,
        max_train=0, max_heldout=args.diagnostic_samples)
    return {layer: {proj: {i: x for i, (x, _) in per_expert.items()}
            for proj, per_expert in projects.items()} for layer, projects in heldout.items()}, probes


def probe(model, tokenizer):
    ids = tokenizer("The answer is", return_tensors="pt")["input_ids"]
    ids = ids.to(model.get_input_embeddings().weight.device)
    with torch.no_grad():
        return model(input_ids=ids).logits.float().cpu()


def summarize(records):
    output = {}
    for projection in ("gate", "up", "down", "all"):
        rows = [r for r in records if projection == "all" or r["projection"] == projection]
        output[projection] = {"count": len(rows)}
        for field in ("sign_agreement", "group_pairwise_sign_agreement",
                      "majority_flip_rate", "residual_frobenius_ratio",
                      "svd_energy_retention", "svd_residual_relative_error",
                      "weight_relative_error",
                      "activation_output_relative_error", "runtime_rounding_relative_error",
                      "expert_output_relative_error"):
            values = [r[field] for r in rows if r[field] is not None]
            output[projection][field] = {"mean": statistics.fmean(values) if values else None,
                                          "max": max(values) if values else None,
                                          "count": len(values)}
    return output


def original_bytes(model, groups):
    elements = 0
    for layer in groups:
        block = model.model.layers[layer].mlp
        for expert in block.experts:
            elements += sum(getattr(expert, name).weight.numel()
                            for name in ("gate_proj", "up_proj", "down_proj"))
    return 2 * elements, elements


def main():
    args = arguments()
    os.chdir(ROOT)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "config.json", vars(args))
    checkpoint = args.checkpoint or out / "bundle_qjl_state.pt"
    if args.mode == "compress" and args.residual_mode == "lowrank" and checkpoint.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint}")
    started = time.perf_counter()
    max_memory = None
    if torch.cuda.is_available():
        max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory *
                             args.gpu_memory_fraction) for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "128GiB"
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
        torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        trust_remote_code=True, max_memory=max_memory)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    metrics = {"model": MODEL, "source": args.model_path, "mode": args.mode,
               "model_load_seconds": time.perf_counter() - started}
    if args.mode == "compress":
        selected = parse_layers(args.layers, len(model.model.layers))
        stage_start = time.perf_counter()
        groups, group_path, group_summary = load_groups(model, tokenizer, args, selected)
        metrics["grouping_or_cache_seconds"] = time.perf_counter() - stage_start
        for layer, layer_groups in groups.items():
            validate_groups(layer_groups, len(model.model.layers[layer].mlp.experts))
            if next(model.model.layers[layer].mlp.experts[0].parameters()).device.type != "cuda":
                raise RuntimeError(f"layer {layer} expert is not on GPU; more GPU memory is required")
        raw_bytes, raw_elements = original_bytes(model, groups)
        stage_start = time.perf_counter()
        samples, probes = routed_samples(model, tokenizer, args, groups)
        metrics["activation_collection_seconds"] = time.perf_counter() - stage_start
        config = {"model": MODEL, "layers": selected, "qjl_d": args.qjl_d,
            "rank": args.rank, "bundle_mode": args.bundle_mode,
            "residual_mode": args.residual_mode, "backend": args.backend,
            "projection_seed": args.projection_seed, "svd_seed": args.svd_seed,
            "svd_oversample": args.svd_oversample, "svd_niter": args.svd_niter,
            "grouping_path": group_path, "grouping_summary": group_summary,
            "svd_method": "torch.svd_lowrank"}
        before_probe = probe(model, tokenizer)
        start = time.perf_counter()
        data, records, timing = compress_model(model, groups, config, samples, probes if isinstance(probes, dict) else None)
        metrics["compression_seconds"] = time.perf_counter() - start
        metrics["grouping_path"] = group_path
        metrics["grouping_rescode_parameter_estimate"] = group_summary
        metrics["summary"] = summarize(records)
        metrics["moe_output_relative_errors"] = [r for r in timing if r.get("group") is None]
        metrics["original_routed_bf16_bytes"] = raw_bytes
        if records:
            with (out / "diagnostics.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
        atomic_json(out / "diagnostics.json", {"records": records, "timing": timing,
                    "activation_diagnostics": "calibration" if samples else "unavailable"})
        if args.residual_mode == "lowrank":
            stage_start = time.perf_counter()
            sizes = save_checkpoint(data, config, checkpoint)
            metrics["checkpoint_save_seconds"] = time.perf_counter() - stage_start
            metrics.update(sizes)
            metrics["checkpoint"] = str(checkpoint)
            metrics["effective_bits_per_routed_weight"] = 8 * sizes["checkpoint_bytes"] / raw_elements
            metrics["shared_sign_bits_per_routed_weight"] = 8 * sizes["shared_packed_sign_bytes"] / raw_elements
            metrics["row_norm_bits_per_routed_weight"] = 8 * sizes["expert_row_norm_bytes"] / raw_elements
            metrics["lowrank_bits_per_routed_weight"] = 8 * sizes["lowrank_factor_bytes"] / raw_elements
            metrics["singleton_bits_per_routed_weight"] = 8 * sizes["singleton_bf16_bytes"] / raw_elements
            metrics["routed_bf16_compression_ratio"] = 1 - sizes["checkpoint_bytes"] / raw_bytes
            metrics["vs_int4_bytes_ratio"] = sizes["checkpoint_bytes"] / (raw_elements / 2)
            metrics["vs_int8_bytes_ratio"] = sizes["checkpoint_bytes"] / raw_elements
            compressed_probe = probe(model, tokenizer)
            stage_start = time.perf_counter()
            load_checkpoint(model, checkpoint, backend=args.backend)
            metrics["checkpoint_reload_seconds"] = time.perf_counter() - stage_start
            reloaded_probe = probe(model, tokenizer)
            parity = float((compressed_probe - reloaded_probe).abs().max())
            metrics["checkpoint_reload_probe_max_abs"] = parity
            if parity != 0:
                raise AssertionError(f"checkpoint reload changed logits: {parity}")
        else:
            metrics["diagnostic_only"] = True
            metrics["checkpoint"] = None
        metrics["original_to_compressed_probe_relative_l2"] = float(
            torch.linalg.vector_norm(probe(model, tokenizer) - before_probe) /
            torch.linalg.vector_norm(before_probe).clamp_min(1e-30))
    else:
        payload = load_checkpoint(model, checkpoint, backend=args.backend)
        metrics["checkpoint"] = str(checkpoint)
        metrics["checkpoint_config"] = payload["config"]
        metrics.update(checkpoint_sizes(payload, checkpoint.stat().st_size))
    metrics.update(gpu_resident_bytes(model))
    ids = tokenizer("The answer is", return_tensors="pt")["input_ids"].to(
        model.get_input_embeddings().weight.device)
    with torch.no_grad():
        model(input_ids=ids)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            model(input_ids=ids)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    metrics["inference_forward_median_ms"] = statistics.median(times) * 1000
    if torch.cuda.is_available():
        metrics["gpu_peak_allocated_bytes"] = [torch.cuda.max_memory_allocated(i)
                                               for i in range(torch.cuda.device_count())]
        metrics["gpu_process_allocated_bytes"] = [torch.cuda.memory_allocated(i)
                                                  for i in range(torch.cuda.device_count())]
    if not args.skip_eval:
        start = time.perf_counter()
        results = evaluate_fewshot(model, tokenizer, args.task, 0,
            eval_batch_size=args.eval_batch_size, output_path=str(out / "lm_eval.txt"),
            limit=args.eval_limit or None)
        metrics["eval_seconds"] = time.perf_counter() - start
        metrics["task_summary"] = report_eval_summary(results, args.task,
            str(out / "lm_eval.txt"), header="Majority Bundled QJL + low-rank")
        metrics["task_results"] = results.get("results", {})
    atomic_json(out / "metrics.json", metrics)
    if args.mode == "compress":
        atomic_json(out / "summary.json", {
            "configuration": config, "summary": metrics["summary"],
            "original_routed_bf16_bytes": metrics["original_routed_bf16_bytes"],
            "checkpoint_bytes": metrics.get("checkpoint_bytes"),
            "effective_bits_per_routed_weight": metrics.get("effective_bits_per_routed_weight"),
            "routed_bf16_compression_ratio": metrics.get("routed_bf16_compression_ratio"),
            "moe_output_relative_errors": metrics["moe_output_relative_errors"]})
    print(f"[bundle] metrics: {out / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
