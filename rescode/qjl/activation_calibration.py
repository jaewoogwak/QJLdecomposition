"""Layer-at-a-time routed covariance collection for Qwen2-MoE.

Only one decoder layer owns GPU covariance accumulators at a time. Hidden
states are passed to the next layer from GPU, CPU, or disk storage; a resumed
in-memory run safely replays completed layers without collecting covariance.
"""

from __future__ import annotations

import gc
import os
import resource
import shutil
import time
from pathlib import Path

import torch
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

FORMAT = "activation-routed-covariance-v2"


def atomic_torch(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def token_batches(tokenizer, texts, batch_size=8, max_length=512, bucket=True):
    """Tokenize once; bucketing changes only batch order, not token contents."""
    encoded = tokenizer(texts, padding=False, truncation=True, max_length=max_length)
    order = list(range(len(texts)))
    if bucket:
        order.sort(key=lambda i: (len(encoded["input_ids"][i]), i))
    batches = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        items = [{"input_ids": encoded["input_ids"][i],
                  "attention_mask": encoded["attention_mask"][i]} for i in indices]
        batch = tokenizer.pad(items, padding=True, return_tensors="pt")
        batches.append({"input_ids": batch["input_ids"],
                        "attention_mask": batch["attention_mask"],
                        "sample_indices": indices})
    return batches


def layer_inputs(backbone, hidden, attention_mask):
    """Mirror Qwen2MoeModel.forward's no-cache mask and position construction."""
    batch_size, seq_length = hidden.shape[:2]
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=hidden.device).unsqueeze(0)
    mask = attention_mask.to(hidden.device)
    implementation = backbone._attn_implementation
    if implementation == "flash_attention_2":
        mask = mask if bool((attention_mask == 0).any()) else None
    elif implementation == "sdpa":
        mask = _prepare_4d_causal_attention_mask_for_sdpa(
            mask, (batch_size, seq_length), hidden, 0,
            sliding_window=backbone.config.sliding_window)
    else:
        mask = _prepare_4d_causal_attention_mask(
            mask, (batch_size, seq_length), hidden, 0,
            sliding_window=backbone.config.sliding_window)
    return mask, position_ids


class RoutedLayerCovariance:
    """GPU FP32 uncentered moments from actual routed expert inputs."""

    def __init__(self, block, *, profile=False):
        self.block = block
        self.n_experts = len(block.experts)
        self.profile = profile
        self.mask = None
        self.selected = None
        self.sums = {"gate": {}, "down": {}}
        self.counts = {"gate": {}, "down": {}}
        self.events = []
        self.cpu_addmm_seconds = 0.
        self.handles = [block.gate.register_forward_hook(self._router_hook)]
        for expert_id, expert in enumerate(block.experts):
            self.handles.append(expert.gate_proj.register_forward_pre_hook(
                self._input_hook(expert_id, "gate")))
            self.handles.append(expert.down_proj.register_forward_pre_hook(
                self._input_hook(expert_id, "down")))

    def begin_batch(self, attention_mask):
        self.mask = attention_mask.reshape(-1).to(
            device=self.block.gate.weight.device, dtype=torch.bool)
        self.selected = None

    def _router_hook(self, _module, _inputs, logits):
        selected = torch.topk(torch.softmax(logits.float(), dim=-1),
                              self.block.top_k, dim=-1).indices
        if self.mask is None or self.mask.numel() != selected.shape[0]:
            raise RuntimeError("router output and attention mask have different token counts")
        self.selected = selected
        current = torch.bincount(selected[self.mask].reshape(-1), minlength=self.n_experts)
        # Accumulate expected totals without reading a CUDA scalar per batch.
        if not hasattr(self, "expected_total"):
            self.expected_total = current.clone()
        else:
            self.expected_total.add_(current)

    def _input_hook(self, expert_id, projection):
        def hook(_module, inputs):
            if self.selected is None:
                raise RuntimeError("expert input arrived before router output")
            _, token = torch.where(self.selected.T == expert_id)
            x = inputs[0]
            if token.numel() != x.shape[0]:
                raise RuntimeError("routed input order differs from router selection")
            valid = self.mask[token]
            x = x[valid].detach().float()
            if not x.numel():
                return
            sums = self.sums[projection]
            if expert_id not in sums:
                sums[expert_id] = torch.zeros(x.shape[1], x.shape[1],
                                             dtype=torch.float32, device=x.device)
                self.counts[projection][expert_id] = 0
            if self.profile and x.is_cuda:
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                sums[expert_id].addmm_(x.T, x)
                end.record()
                self.events.append((begin, end))
            else:
                started = time.perf_counter() if self.profile else None
                sums[expert_id].addmm_(x.T, x)
                if started is not None:
                    self.cpu_addmm_seconds += time.perf_counter() - started
            self.counts[projection][expert_id] += x.shape[0]
        return hook

    def snapshot(self):
        if not hasattr(self, "expected_total"):
            raise RuntimeError("no routed batches were observed")
        expected = self.expected_total.cpu().tolist()
        experts = {}
        started = time.perf_counter()
        for expert_id in range(self.n_experts):
            expert = {}
            for projection in ("gate", "down"):
                count = self.counts[projection].get(expert_id, 0)
                if count != expected[expert_id]:
                    raise RuntimeError(f"expert {expert_id} {projection}: "
                                       f"observed {count} routed inputs, expected {expected[expert_id]}")
                if count:
                    covariance = (self.sums[projection][expert_id] / count).cpu()
                else:
                    dim = getattr(self.block.experts[expert_id], projection + "_proj").in_features
                    covariance = torch.zeros(dim, dim, dtype=torch.float32)
                expert[projection] = {"covariance": covariance, "count": count, "probe": None}
            expert["up"] = expert["gate"]
            experts[expert_id] = expert
        transfer_seconds = time.perf_counter() - started
        accumulation_seconds = self.cpu_addmm_seconds
        if self.events:
            torch.cuda.synchronize(self.sums["gate"][next(iter(self.sums["gate"]))].device)
            accumulation_seconds += sum(begin.elapsed_time(end) for begin, end in self.events) / 1000
        return {"format": FORMAT, "experts": experts,
                "profile": {"covariance_addmm_seconds": accumulation_seconds,
                            "covariance_transfer_seconds": transfer_seconds}}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _storage_mode(requested, backbone, batches):
    if requested != "auto":
        return requested
    device = backbone.embed_tokens.weight.device
    bytes_needed = sum(batch["input_ids"].numel() for split in batches.values()
                       for batch in split) * backbone.config.hidden_size * 2
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        if free > bytes_needed * 3:
            return "gpu"
    try:
        import psutil
        if psutil.virtual_memory().available > bytes_needed * 4:
            return "cpu"
    except ImportError:
        return "cpu"
    return "disk"


def _hidden_dir(cache_dir, split, layer):
    return cache_dir / "hidden" / split / f"layer_{layer}"


def _disk_hidden_paths(directory, count):
    if not (directory / "COMPLETE").is_file():
        return None
    paths = [directory / f"batch_{i:04d}.pt" for i in range(count)]
    return paths if all(path.is_file() for path in paths) else None


@torch.no_grad()
def collect_layerwise(backbone, batches, layers, cache_dir, *, hidden_cache="auto",
                      resume=False, profile=False, progress=print):
    """Process every batch through layer 0, then layer 1, and so on.

    Completed covariance files are never recomputed on resume. In-memory
    hidden states are safely replayed through completed layers after restart;
    disk mode instead resumes from the last complete hidden shard.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    selected = set(layers)
    if not selected:
        raise ValueError("at least one layer is required")
    mode = _storage_mode(hidden_cache, backbone, batches)
    if mode not in ("gpu", "cpu", "disk"):
        raise ValueError(f"unknown hidden cache mode: {mode}")
    current = {split: None for split in batches}
    profiles = []
    for layer_idx in range(max(selected) + 1):
        layer = backbone.layers[layer_idx]
        next_hidden = {}
        for split, split_batches in batches.items():
            cov_path = cache_dir / f"{split}_layer_{layer_idx}.pt"
            completed_cov = layer_idx not in selected or (resume and cov_path.is_file())
            output_dir = _hidden_dir(cache_dir, split, layer_idx)
            ready_hidden = (_disk_hidden_paths(output_dir, len(split_batches))
                            if mode == "disk" and resume else None)
            if ready_hidden is not None and completed_cov:
                next_hidden[split] = ready_hidden
                progress(f"[calibrate] {split} layer {layer_idx}: resumed hidden state")
                continue
            collector = (RoutedLayerCovariance(layer.mlp, profile=profile)
                         if layer_idx in selected and not completed_cov else None)
            if mode == "disk":
                shutil.rmtree(output_dir, ignore_errors=True)
                output_dir.mkdir(parents=True)
            outputs = []
            forward_seconds = cache_seconds = 0.
            if profile and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            try:
                for batch_id, batch in enumerate(split_batches):
                    if current[split] is None:
                        embedding_device = backbone.embed_tokens.weight.device
                        hidden = backbone.embed_tokens(batch["input_ids"].to(embedding_device))
                    else:
                        prior = current[split][batch_id]
                        hidden = (torch.load(prior, map_location="cpu", weights_only=True)
                                  if isinstance(prior, Path) else prior)
                    device = next(layer.parameters()).device
                    hidden = hidden.to(device)
                    attention_mask, position_ids = layer_inputs(backbone, hidden,
                                                                 batch["attention_mask"])
                    if collector is not None:
                        collector.begin_batch(batch["attention_mask"])
                    if profile and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    output = layer(hidden, attention_mask=attention_mask,
                                   position_ids=position_ids, past_key_value=None,
                                   output_attentions=False, output_router_logits=False,
                                   use_cache=False)[0].detach()
                    if profile and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    forward_seconds += time.perf_counter() - started
                    started = time.perf_counter()
                    if mode == "disk":
                        path = output_dir / f"batch_{batch_id:04d}.pt"
                        atomic_torch(path, output.cpu())
                        outputs.append(path)
                    elif mode == "cpu":
                        outputs.append(output.cpu())
                    else:
                        outputs.append(output)
                    cache_seconds += time.perf_counter() - started
                    progress(f"[calibrate] {split} layer {layer_idx}: "
                             f"{batch_id + 1}/{len(split_batches)}", flush=True)
                snapshot = collector.snapshot() if collector is not None else None
                if snapshot is not None:
                    atomic_torch(cov_path, snapshot)
                if mode == "disk":
                    (output_dir / "COMPLETE").write_text(str(len(outputs)))
                next_hidden[split] = outputs
                hidden_bytes = (sum(path.stat().st_size for path in outputs)
                                if mode == "disk" else
                                sum(tensor.numel() * tensor.element_size() for tensor in outputs))
                profiles.append({"split": split, "layer": layer_idx,
                                 "covariance_reused": completed_cov,
                                 "forward_seconds": forward_seconds,
                                 "hidden_cache_seconds": cache_seconds,
                                 "hidden_cache_bytes": hidden_bytes,
                                 "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                                 "covariance_addmm_seconds": (snapshot["profile"]["covariance_addmm_seconds"]
                                                               if snapshot else 0.),
                                 "covariance_transfer_seconds": (snapshot["profile"]["covariance_transfer_seconds"]
                                                                  if snapshot else 0.),
                                 "gpu_peak_allocated_bytes": ([torch.cuda.max_memory_allocated(i)
                                                               for i in range(torch.cuda.device_count())]
                                                              if profile else [])})
            finally:
                if collector is not None:
                    collector.close()
            if mode == "disk" and layer_idx > 0:
                # The preceding shard is no longer needed after this split's
                # next layer has completed and its output is durable.
                shutil.rmtree(_hidden_dir(cache_dir, split, layer_idx - 1), ignore_errors=True)
        current = next_hidden
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return profiles, mode
