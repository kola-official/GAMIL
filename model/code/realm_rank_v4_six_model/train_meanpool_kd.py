#!/usr/bin/env python3
"""Mean-pool/fragment-level KD training for 6-layer ViraLM students."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from experiment_config import LOG_ROOT, OUTPUT_ROOT, TRAIN_CSV_DIR
from shared import (
    DataCollatorForSupervisedDataset,
    _binary_logits,
    build_compute_metrics,
    copy_custom_model_files,
    last_hidden,
    load_or_build_tokenized_dataset,
    preprocess_logits_for_metrics,
    safe_save_model_for_hf_trainer,
    setup_logging,
    str2bool,
    subset_hf_dataset_by_sources,
    validate_staged_teacher,
    write_json,
)


class MeanPoolKDTrainer(Trainer):
    def __init__(
        self,
        teacher_model: torch.nn.Module,
        *args: Any,
        temperature: float = 2.0,
        alpha_bce: float = 2.0,
        alpha_distil: float = 7.0,
        alpha_cos: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.to(self.args.device)

        self.temperature = float(temperature)
        self.alpha_bce = float(alpha_bce)
        self.alpha_distil = float(alpha_distil)
        self.alpha_cos = float(alpha_cos)
        self.bce_loss_fct = nn.BCEWithLogitsLoss()
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")
        self.cos_loss_fct = nn.CosineEmbeddingLoss()

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        return self.optimizer

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        inputs = dict(inputs)
        inputs["output_hidden_states"] = True
        labels = inputs.pop("labels", None)
        attention_mask = inputs.get("attention_mask")

        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        student_hidden = last_hidden(student_outputs.hidden_states)

        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits
            teacher_hidden = last_hidden(teacher_outputs.hidden_states)

        if labels is not None:
            loss_bce = self.bce_loss_fct(_binary_logits(student_logits), labels.float())
        else:
            loss_bce = torch.zeros((), device=student_logits.device)

        student_log_probs = F.log_softmax(student_logits.float() / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / self.temperature, dim=-1)
        loss_distil = self.kl_loss_fct(student_log_probs, teacher_probs)

        if self.alpha_cos > 0.0 and student_hidden is not None and teacher_hidden is not None:
            if attention_mask is not None:
                active = attention_mask.reshape(-1) == 1
                student_tokens = student_hidden.reshape(-1, student_hidden.size(-1))[active].float()
                teacher_tokens = teacher_hidden.reshape(-1, teacher_hidden.size(-1))[active].float()
            else:
                student_tokens = student_hidden.reshape(-1, student_hidden.size(-1)).float()
                teacher_tokens = teacher_hidden.reshape(-1, teacher_hidden.size(-1)).float()
            if student_tokens.numel() == 0:
                loss_cos = torch.zeros((), device=student_logits.device)
            else:
                target = torch.ones(student_tokens.size(0), device=student_tokens.device)
                loss_cos = self.cos_loss_fct(student_tokens, teacher_tokens, target)
        else:
            loss_cos = torch.zeros((), device=student_logits.device)

        loss = (
            self.alpha_bce * loss_bce
            + self.alpha_distil * (self.temperature ** 2) * loss_distil
            + self.alpha_cos * loss_cos
        )

        if return_outputs:
            try:
                student_outputs.hidden_states = None
                student_outputs.attentions = None
            except Exception:
                pass
            return loss, student_outputs
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 6L mean-pool KD ViraLM model")
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--data-path", default=str(TRAIN_CSV_DIR))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--dataloader-num-workers", type=int, default=16)
    parser.add_argument("--tokenizer-batch-size", type=int, default=4096)
    parser.add_argument("--tokenize-num-proc", type=int, default=16)
    parser.add_argument("--tokenized-cache-dir", default="")
    parser.add_argument("--tokenized-cache-format", default="arrow")
    parser.add_argument("--token-cache-model-ref", default="")
    parser.add_argument("--rebuild-tokenized-cache", default="False")
    parser.add_argument("--sequence-eval-threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha-bce", type=float, default=2.0)
    parser.add_argument("--alpha-distil", type=float, default=7.0)
    parser.add_argument("--alpha-cos", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", default="True")
    parser.add_argument("--require-cuda", default="True")
    parser.add_argument("--ddp-find-unused-parameters", default="False")
    parser.add_argument("--smoke-test", default=os.environ.get("SMOKE_TEST", "0"))
    parser.add_argument("--smoke-train-groups", type=int, default=int(os.environ.get("SMOKE_TRAIN_GROUPS", "8")))
    parser.add_argument("--smoke-dev-groups", type=int, default=int(os.environ.get("SMOKE_DEV_GROUPS", "4")))
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    is_smoke = str2bool(args.smoke_test)

    if str2bool(args.require_cuda) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training entry.")

    teacher_basename = Path(args.teacher_model).name
    if teacher_basename == "viralm-r":
        validate_staged_teacher("viralm_r_v4_final", args.teacher_model)
    elif teacher_basename == "viralm-o":
        validate_staged_teacher("viralm_o", args.teacher_model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    data_path = Path(args.data_path)
    train_csv = data_path / "train.csv"
    dev_csv = data_path / "dev.csv"
    tokenized_cache_dir = args.tokenized_cache_dir or str(data_path / "tokenized_cache")
    cache_model_ref = args.token_cache_model_ref or args.student_model

    train_dataset = load_or_build_tokenized_dataset(
        "train",
        str(train_csv),
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
    dev_dataset = load_or_build_tokenized_dataset(
        "dev",
        str(dev_csv),
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

    if is_smoke:
        train_dataset = subset_hf_dataset_by_sources(train_dataset, args.smoke_train_groups)
        dev_dataset = subset_hf_dataset_by_sources(dev_dataset, args.smoke_dev_groups)

    dev_source_ids = [str(x) for x in dev_dataset["source_id"]]
    metric_fn = build_compute_metrics(dev_source_ids, args.sequence_eval_threshold)

    student = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        num_labels=2,
        trust_remote_code=True,
    )
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher_model,
        num_labels=2,
        trust_remote_code=True,
    )

    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp16 = str2bool(args.fp16) and torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size if not is_smoke else min(args.per_device_train_batch_size, 4),
        per_device_eval_batch_size=args.per_device_eval_batch_size if not is_smoke else min(args.per_device_eval_batch_size, 8),
        gradient_accumulation_steps=1,
        num_train_epochs=1.0 if is_smoke else args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1 if is_smoke else args.logging_steps,
        save_total_limit=1 if is_smoke else args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=fp16,
        dataloader_pin_memory=True,
        dataloader_num_workers=0 if is_smoke else args.dataloader_num_workers,
        remove_unused_columns=False,
        group_by_length=False,
        ddp_find_unused_parameters=str2bool(args.ddp_find_unused_parameters),
        report_to=[],
        seed=args.seed,
    )

    trainer = MeanPoolKDTrainer(
        teacher_model=teacher,
        model=student,
        tokenizer=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=metric_fn,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
        temperature=args.temperature,
        alpha_bce=args.alpha_bce,
        alpha_distil=args.alpha_distil,
        alpha_cos=args.alpha_cos,
    )

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer, str(output_dir))
    copy_custom_model_files(args.student_model, output_dir)
    results = trainer.evaluate(eval_dataset=dev_dataset)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "results" / "val_results.json", results)
    write_json(
        output_dir / "model_meta.json",
        {
            "run_name": args.run_name,
            "kind": "meanpool_kd",
            "student_model": args.student_model,
            "teacher_model": args.teacher_model,
            "model_max_length": args.model_max_length,
            "temperature": args.temperature,
            "alpha_bce": args.alpha_bce,
            "alpha_distil": args.alpha_distil,
            "alpha_cos": args.alpha_cos,
            "smoke_test": is_smoke,
        },
    )


if __name__ == "__main__":
    main()
