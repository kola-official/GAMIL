#!/usr/bin/env python3
"""Gated-attention MIL KD training for 6-layer ViraLM students."""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from experiment_config import TRAIN_CSV_DIR
from mil_model import ViraLM_MIL_Gated
from mil_train_common import (
    autocast_context,
    barrier,
    cleanup_dist,
    evaluate_mil,
    save_mil_artifacts,
    save_mil_checkpoint,
    setup_dist,
)
from shared import (
    _binary_logits,
    last_hidden,
    load_or_build_tokenized_dataset,
    sequence_collate_fn,
    SequenceGroupedDataset,
    setup_logging,
    str2bool,
    validate_staged_teacher,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 6L gated-attention MIL KD model")
    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--resume-mil-state", default="")
    parser.add_argument("--data-path", default=str(TRAIN_CSV_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--learning-rate-backbone", type=float, default=1e-5)
    parser.add_argument("--learning-rate-head", type=float, default=1e-4)
    parser.add_argument("--scan-chunk", type=int, default=int(os.environ.get("SCAN_CHUNK", "16")))
    parser.add_argument("--grad-chunk", type=int, default=int(os.environ.get("GRAD_CHUNK", "8")))
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-distil", type=float, default=7.0)
    parser.add_argument("--alpha-cor", type=float, default=1.0)
    parser.add_argument("--lambda-seq", type=float, default=float(os.environ.get("LAMBDA_SEQ", "0.5")))
    parser.add_argument("--lambda-frag", type=float, default=float(os.environ.get("LAMBDA_FRAG", "1.0")))
    parser.add_argument("--tokenizer-batch-size", type=int, default=4096)
    parser.add_argument("--tokenize-num-proc", type=int, default=16)
    parser.add_argument("--tokenized-cache-dir", default="")
    parser.add_argument("--tokenized-cache-format", default="arrow")
    parser.add_argument("--token-cache-model-ref", default="")
    parser.add_argument("--rebuild-tokenized-cache", default="False")
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", default="True")
    parser.add_argument("--require-cuda", default="True")
    parser.add_argument("--eval-frag-metrics", default=os.environ.get("EVAL_FRAG_METRICS", "1"))
    parser.add_argument("--smoke-test", default=os.environ.get("SMOKE_TEST", "0"))
    parser.add_argument("--smoke-train-groups", type=int, default=int(os.environ.get("SMOKE_TRAIN_GROUPS", "8")))
    parser.add_argument("--smoke-dev-groups", type=int, default=int(os.environ.get("SMOKE_DEV_GROUPS", "4")))
    return parser.parse_args()


def flatten_batch(batch, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    max_len = max(ids.size(1) for ids in batch["input_ids"])
    padded_ids = [F.pad(ids, (0, max_len - ids.size(1)), value=0) for ids in batch["input_ids"]]
    padded_mask = [F.pad(mask, (0, max_len - mask.size(1)), value=0) for mask in batch["attention_mask"]]
    return torch.cat(padded_ids, dim=0).to(device), torch.cat(padded_mask, dim=0).to(device), [ids.size(0) for ids in batch["input_ids"]]


def head_forward_from_hidden(model: ViraLM_MIL_Gated, h_batch: torch.Tensor, counts: Sequence[int]) -> torch.Tensor:
    """Vectorized gated-attention MIL head over flattened fragments."""
    if not counts:
        return h_batch.new_empty((0, model.num_classes))
    device = h_batch.device
    seq_lens = torch.as_tensor(list(counts), device=device, dtype=torch.long)
    seq_ids = torch.repeat_interleave(torch.arange(len(counts), device=device), seq_lens)

    e_all = model.attention_w(
        torch.tanh(model.attention_V(h_batch)) * torch.sigmoid(model.attention_U(h_batch))
    ).squeeze(-1).float()
    seq_max = torch.full((len(counts),), torch.finfo(e_all.dtype).min, device=device, dtype=e_all.dtype)
    seq_max.scatter_reduce_(0, seq_ids, e_all, reduce="amax", include_self=True)
    exp_all = torch.exp(e_all - seq_max[seq_ids])
    seq_sum = torch.zeros(len(counts), device=device, dtype=e_all.dtype)
    seq_sum.scatter_add_(0, seq_ids, exp_all)
    attn = (exp_all / seq_sum[seq_ids]).to(h_batch.dtype)

    weighted_h = attn.unsqueeze(1) * h_batch
    anchors = h_batch.new_zeros((len(counts), h_batch.size(1)))
    anchors.scatter_add_(0, seq_ids.unsqueeze(1).expand(-1, h_batch.size(1)), weighted_h)
    return model.seq_classifier(anchors)


def tokenwise_hidden_cosine_loss(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.bool()
    if mask.numel() == 0 or not bool(mask.any()):
        return student_hidden.new_zeros(())
    s_norm = F.normalize(student_hidden.float(), p=2, dim=-1)
    t_norm = F.normalize(teacher_hidden.float(), p=2, dim=-1)
    token_loss = 1.0 - (s_norm * t_norm).sum(dim=-1)
    return token_loss.masked_select(mask).mean()


def load_data(args: argparse.Namespace, tokenizer, is_smoke: bool):
    data_path = Path(args.data_path)
    tokenized_cache_dir = args.tokenized_cache_dir or str(data_path / "tokenized_cache")
    cache_model_ref = args.token_cache_model_ref or args.backbone_model
    train_hf = load_or_build_tokenized_dataset(
        "train",
        str(data_path / "train.csv"),
        tokenizer,
        cache_model_ref,
        args.model_max_length,
        args.tokenizer_batch_size if not is_smoke else min(args.tokenizer_batch_size, 512),
        args.tokenize_num_proc if not is_smoke else min(args.tokenize_num_proc, 2),
        tokenized_cache_dir,
        args.tokenized_cache_format,
        str2bool(args.rebuild_tokenized_cache),
        None,
    )
    dev_hf = load_or_build_tokenized_dataset(
        "dev",
        str(data_path / "dev.csv"),
        tokenizer,
        cache_model_ref,
        args.model_max_length,
        args.tokenizer_batch_size if not is_smoke else min(args.tokenizer_batch_size, 512),
        args.tokenize_num_proc if not is_smoke else min(args.tokenize_num_proc, 2),
        tokenized_cache_dir,
        args.tokenized_cache_format,
        str2bool(args.rebuild_tokenized_cache),
        None,
    )
    return (
        SequenceGroupedDataset(train_hf, max_groups=args.smoke_train_groups if is_smoke else 0),
        SequenceGroupedDataset(dev_hf, max_groups=args.smoke_dev_groups if is_smoke else 0),
    )


def main() -> None:
    setup_logging()
    args = parse_args()
    is_smoke = str2bool(args.smoke_test)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    teacher_basename = Path(args.teacher_model).name
    if teacher_basename == "viralm-r":
        validate_staged_teacher("viralm_r", args.teacher_model)
    elif teacher_basename == "viralm-o":
        validate_staged_teacher("viralm_o", args.teacher_model)

    local_rank, world_size, device, is_distributed = setup_dist(require_cuda=str2bool(args.require_cuda))
    is_rank0 = local_rank == 0
    use_amp = str2bool(args.fp16) and torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained(args.backbone_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    config = AutoConfig.from_pretrained(args.backbone_model, trust_remote_code=True)
    train_dataset, dev_dataset = load_data(args, tokenizer, is_smoke)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, 2) if is_smoke else args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=sequence_collate_fn,
        num_workers=0 if is_smoke else args.dataloader_workers,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=min(args.eval_batch_size, 4) if is_smoke else args.eval_batch_size,
        shuffle=False,
        collate_fn=sequence_collate_fn,
        num_workers=0 if is_smoke else args.dataloader_workers,
    )

    backbone = AutoModelForSequenceClassification.from_pretrained(
        args.backbone_model,
        num_labels=2,
        trust_remote_code=True,
    ).to(device)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model,
        num_labels=2,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    model = ViraLM_MIL_Gated(backbone, hidden_size=int(getattr(config, "hidden_size", 768)), num_classes=2).to(device)
    if args.resume_mil_state and Path(args.resume_mil_state).exists():
        state = torch.load(args.resume_mil_state, map_location=device)
        model.load_state_dict(state, strict=False)
        if is_rank0:
            print(f"Resumed MIL state from {args.resume_mil_state}")
    model.unfreeze_backbone()
    if hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if n.startswith("backbone.")], "lr": args.learning_rate_backbone},
            {"params": [p for n, p in model.named_parameters() if not n.startswith("backbone.")], "lr": args.learning_rate_head},
        ]
    )

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    mm = model.module if hasattr(model, "module") else model

    output_dir = Path(args.output_dir)
    if is_rank0:
        save_mil_artifacts(
            output_dir,
            args.backbone_model,
            tokenizer,
            config,
            {
                "run_name": args.run_name,
                "kind": "gated_mil_kd",
                "backbone_model": args.backbone_model,
                "teacher_model": args.teacher_model,
                "resume_mil_state": args.resume_mil_state,
                "model_max_length": args.model_max_length,
                "batch_size": args.batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "scan_chunk": args.scan_chunk,
                "grad_chunk": args.grad_chunk,
                "temperature": args.temperature,
                "alpha_distil": args.alpha_distil,
                "alpha_cor": args.alpha_cor,
                "lambda_seq": args.lambda_seq,
                "lambda_frag": args.lambda_frag,
                "smoke_test": is_smoke,
            },
        )
    barrier()

    criterion_none = nn.BCEWithLogitsLoss(reduction="none")
    kd_loss = nn.KLDivLoss(reduction="batchmean")
    best_seq_f1 = -1.0
    epochs = 1 if is_smoke else args.epochs

    for epoch in range(epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=not is_rank0)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(pbar):
            labels = batch["labels"].to(device)
            flat_ids, flat_mask, counts = flatten_batch(batch, device)
            n_total = flat_ids.size(0)
            total_bs = max(1, args.grad_accum_steps * args.batch_size * world_size)
            step_start = time.time()

            with torch.no_grad():
                h_parts = []
                with autocast_context(use_amp):
                    for start in range(0, n_total, args.scan_chunk):
                        c_ids = flat_ids[start : start + args.scan_chunk]
                        c_mask = flat_mask[start : start + args.scan_chunk]
                        s_out = mm.backbone(input_ids=c_ids, attention_mask=c_mask, output_hidden_states=True)
                        s_hidden = last_hidden(s_out.hidden_states)
                        if s_hidden is None:
                            raise RuntimeError("Student did not return hidden states")
                        h_parts.append(s_hidden[:, 0, :].detach().contiguous())
                        del s_out, s_hidden
                h_batch = torch.cat(h_parts, dim=0).detach()

            scan_time = time.time() - step_start
            sync_now = ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(train_loader))
            no_sync = getattr(model, "no_sync", contextlib.nullcontext)
            sync_context = contextlib.nullcontext if sync_now or not is_distributed else no_sync

            with sync_context():
                h_for_head = h_batch.detach().requires_grad_(True)
                with autocast_context(use_amp):
                    seq_logits = head_forward_from_hidden(mm, h_for_head, counts)
                    seq_loss = args.lambda_seq * criterion_none(_binary_logits(seq_logits), labels.float()).sum() / total_bs
                seq_loss.backward()
                grad_h = h_for_head.grad.detach()

                last_chunk_loss = torch.zeros((), device=device)
                for start in range(0, n_total, args.grad_chunk):
                    c_ids = flat_ids[start : start + args.grad_chunk]
                    c_mask = flat_mask[start : start + args.grad_chunk]
                    n_chunk = c_ids.size(0)
                    with torch.no_grad():
                        with autocast_context(use_amp):
                            t_out = teacher(input_ids=c_ids, attention_mask=c_mask, output_hidden_states=True)
                            t_hidden = last_hidden(t_out.hidden_states)
                            if t_hidden is None:
                                raise RuntimeError("Teacher did not return hidden states")
                            t_hidden = t_hidden.detach()
                            t_logits = t_out.logits.detach()
                            del t_out
                    with autocast_context(use_amp):
                        s_out = mm.backbone(input_ids=c_ids, attention_mask=c_mask, output_hidden_states=True)
                        s_hidden = last_hidden(s_out.hidden_states)
                        if s_hidden is None:
                            raise RuntimeError("Student did not return hidden states")
                        h_c = s_hidden[:, 0, :]
                        z_c = s_out.logits
                        s_log_p = F.log_softmax(z_c.float() / args.temperature, dim=-1)
                        t_p = F.softmax(t_logits.float() / args.temperature, dim=-1)
                        loss_kd = kd_loss(s_log_p, t_p) * (args.temperature ** 2)
                        loss_cor = tokenwise_hidden_cosine_loss(s_hidden, t_hidden, c_mask)
                        loss_seq_proj = (grad_h[start : start + n_chunk].to(h_c.dtype) * h_c).sum()
                        last_chunk_loss = loss_seq_proj + args.lambda_frag * (
                            args.alpha_distil * loss_kd + args.alpha_cor * loss_cor
                        ) / total_bs
                    last_chunk_loss.backward()
                    del s_out, s_hidden, t_hidden, t_logits

            if sync_now:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if is_rank0 and step % 10 == 0:
                pbar.set_postfix(
                    {
                        "seq": f"{float(seq_loss.item()):.4f}",
                        "chunk": f"{float(last_chunk_loss.item()):.4f}",
                        "scan": f"{scan_time:.2f}s",
                    }
                )

        if is_rank0:
            compute_frag_metrics = str2bool(args.eval_frag_metrics)
            val_loss, seq_m, frag_m = evaluate_mil(
                mm,
                dev_loader,
                device,
                local_rank,
                args.scan_chunk,
                use_amp,
                compute_frag_metrics=compute_frag_metrics,
            )
            print(
                f"Epoch {epoch + 1} val_loss={val_loss:.6f} seq_f1={seq_m['f1']:.6f}"
                + (f" frag_f1={frag_m['f1']:.6f}" if frag_m else "")
            )
            save_mil_checkpoint(mm, output_dir, filename=f"mil_model_epoch{epoch + 1}.pt")
            if seq_m["f1"] > best_seq_f1:
                best_seq_f1 = float(seq_m["f1"])
                save_mil_checkpoint(mm, output_dir, filename="best_mil_model.pt")
        barrier()

    cleanup_dist()


if __name__ == "__main__":
    main()
