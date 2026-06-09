#!/usr/bin/env python3
"""Shared routines for gated-attention MIL training."""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared import (
    _binary_logits,
    compute_binary_metrics_from_probs,
    copy_custom_model_files,
    copy_tokenizer_files,
    write_json,
)


def setup_dist(require_cuda: bool = True) -> Tuple[int, int, torch.device, bool]:
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MIL training.")
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return local_rank, world_size, device, dist.is_initialized()


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in model.state_dict().items():
        cpu_value = value.detach().cpu()
        if cpu_value.is_floating_point():
            cpu_value = cpu_value.float()
        out[key] = cpu_value
    return out


def save_mil_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    filename: str = "best_mil_model.pt",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state_dict(model), output_dir / filename)


def save_mil_artifacts(
    output_dir: Path,
    backbone_path: str,
    tokenizer: Any,
    config: Any,
    meta: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(output_dir))
    config.save_pretrained(str(output_dir))
    copy_custom_model_files(backbone_path, output_dir)
    write_json(output_dir / "model_meta.json", meta)


def evaluate_mil(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    local_rank: int,
    scan_chunk: int,
    use_amp: bool,
    compute_frag_metrics: bool = True,
) -> Tuple[float, Dict[str, float], Optional[Dict[str, float]]]:
    model.eval()
    seq_probs = []
    seq_labels = []
    frag_probs = []
    frag_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    steps = 0

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating", disable=(local_rank != 0)):
            labels = batch["labels"].to(device)
            with autocast_context(use_amp):
                seq_logits, _, frag_logits_list, _ = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    sub_chunk_size=scan_chunk,
                    return_frag_logits=compute_frag_metrics,
                    return_hidden=False,
                )
            bce_logits = _binary_logits(seq_logits)
            loss = criterion(bce_logits, labels.float())
            total_loss += float(loss.item())
            steps += 1
            probs = torch.sigmoid(bce_logits).float().cpu().numpy().tolist()
            seq_probs.extend(probs)
            seq_labels.extend(labels.cpu().numpy().astype(int).tolist())

            if compute_frag_metrics and frag_logits_list is not None:
                labels_cpu = labels.cpu().numpy().astype(int).tolist()
                for idx, frag_logits in enumerate(frag_logits_list):
                    frag_bce = _binary_logits(frag_logits.to(device))
                    frag_p = torch.sigmoid(frag_bce).float().cpu().numpy().tolist()
                    frag_probs.extend(frag_p)
                    frag_labels.extend([labels_cpu[idx]] * len(frag_p))

    seq_metrics = compute_binary_metrics_from_probs(np.asarray(seq_probs), np.asarray(seq_labels), threshold=0.5)
    frag_metrics = None
    if compute_frag_metrics:
        frag_metrics = compute_binary_metrics_from_probs(np.asarray(frag_probs), np.asarray(frag_labels), threshold=0.5)
    return total_loss / max(1, steps), seq_metrics, frag_metrics

