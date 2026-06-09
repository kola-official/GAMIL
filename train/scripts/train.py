#!/usr/bin/env python3
"""Train ViraL_Mamba with directory-built train/dev CSV inputs.

Defaults aligned to requested settings:
- learning_rate = 1e-5
- warmup_steps = 50
- optimizer = Adam
- loss = binary cross-entropy (BCEWithLogitsLoss)
"""

import csv
import faulthandler
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import transformers
from datasets import Dataset as HFDataset
from datasets import load_dataset, load_from_disk
from filelock import FileLock
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset as TorchDataset, Sampler


LOGGER = logging.getLogger(__name__)

LOCKED_LEARNING_RATE = 1e-5
LOCKED_WARMUP_STEPS = 50
SUPPORTED_TOKENIZED_CACHE_FORMATS = {"arrow", "parquet"}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="model")


@dataclass
class DataArguments:
    data_path: str = field(default="prepared_data", metadata={"help": "Path containing train.csv and dev.csv"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="viralm_train")
    model_max_length: int = field(default=512)
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=32)
    num_train_epochs: int = field(default=3)
    fp16: bool = field(default=True)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=3800)
    eval_steps: int = field(default=3800)
    evaluation_strategy: str = field(default="epoch")
    eval_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    warmup_steps: int = field(default=50)
    learning_rate: float = field(default=1e-5)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="f1")
    greater_is_better: bool = field(default=True)
    output_dir: str = field(default="output/viralm_train")
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=16)
    tokenizer_batch_size: int = field(default=4096)
    tokenize_num_proc: int = field(default=16)
    tokenized_cache_dir: str = field(default="")
    tokenized_cache_format: str = field(default="arrow")
    rebuild_tokenized_cache: bool = field(default=False)
    train_batch_plan_path: str = field(default="")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(default=True)
    sequence_eval_threshold: float = field(default=0.5)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    require_cuda: bool = field(default=False)
    report_to: str = field(default="none")
    seed: int = field(default=42)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str) -> None:
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {}
        for key, value in state_dict.items():
            cpu_value = value.detach().cpu()
            if cpu_value.is_floating_point():
                cpu_value = cpu_value.float()
            cpu_state_dict[key] = cpu_value
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def _normalize_cache_format(cache_format: str) -> str:
    normalized = cache_format.strip().lower()
    if normalized not in SUPPORTED_TOKENIZED_CACHE_FORMATS:
        raise ValueError(
            f"tokenized_cache_format must be one of {sorted(SUPPORTED_TOKENIZED_CACHE_FORMATS)}, got: {cache_format}"
        )
    return normalized


def _build_cache_key(csv_path: str, model_name_or_path: str, max_length: int) -> str:
    stat = os.stat(csv_path)
    model_ref = os.path.abspath(model_name_or_path) if os.path.exists(model_name_or_path) else model_name_or_path
    payload = {
        "csv_path": os.path.abspath(csv_path),
        "csv_size": stat.st_size,
        "csv_mtime_ns": stat.st_mtime_ns,
        "model_ref": model_ref,
        "max_length": max_length,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _portable_model_ref(model_name_or_path: str) -> str:
    if os.path.exists(model_name_or_path):
        return os.path.basename(os.path.abspath(model_name_or_path))
    return model_name_or_path


def _find_portable_tokenized_cache(
    split_root: str,
    csv_path: str,
    model_name_or_path: str,
    max_length: int,
    cache_format: str,
) -> Optional[str]:
    if not os.path.isdir(split_root):
        return None

    csv_name = os.path.basename(csv_path)
    csv_size = os.path.getsize(csv_path)
    model_ref = _portable_model_ref(model_name_or_path)

    for cache_key in sorted(os.listdir(split_root)):
        candidate_root = os.path.join(split_root, cache_key)
        meta_path = os.path.join(candidate_root, "meta.json")
        if not os.path.isfile(meta_path):
            continue

        try:
            with open(meta_path) as handle:
                metadata = json.load(handle)
        except Exception:
            continue

        if metadata.get("cache_format") != cache_format:
            continue
        if int(metadata.get("max_length", -1)) != int(max_length):
            continue
        if os.path.basename(str(metadata.get("csv_path", ""))) != csv_name:
            continue
        if int(metadata.get("csv_size", -1)) != int(csv_size):
            continue
        if _portable_model_ref(str(metadata.get("model_name_or_path", ""))) != model_ref:
            continue

        if cache_format == "arrow":
            cache_path = os.path.join(candidate_root, "arrow")
            if os.path.isdir(cache_path):
                return cache_path
        else:
            cache_path = os.path.join(candidate_root, "data.parquet")
            if os.path.isfile(cache_path):
                return cache_path

    return None


def _to_binary_float(value: Union[str, int, float]) -> float:
    return float(int(float(value)))


def _build_tokenized_from_csv(
    csv_path: str,
    split_name: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int,
    tokenizer_batch_size: int,
    tokenize_num_proc: int,
    cache_dir: Optional[str],
    rebuild_tokenized_cache: bool,
) -> HFDataset:
    start_time = time.time()
    raw_dataset = load_dataset("csv", data_files=csv_path, split="train", cache_dir=cache_dir)
    if "sequence" not in raw_dataset.column_names or "label" not in raw_dataset.column_names:
        raise ValueError(f"CSV must contain 'sequence' and 'label' columns: {csv_path}")

    LOGGER.info(
        "Tokenizing %s split from %s (rows=%s, batch_size=%s, num_proc=%s)",
        split_name,
        csv_path,
        len(raw_dataset),
        tokenizer_batch_size,
        tokenize_num_proc,
    )

    def _tokenize_batch(
        examples: Dict[str, Sequence[Union[str, int, float]]],
        indices: Sequence[int],
    ) -> Dict[str, Sequence[Union[List[int], float, str]]]:
        sequences = [str(seq).strip().upper() for seq in examples["sequence"]]
        encoded = tokenizer(
            sequences,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = [_to_binary_float(label) for label in examples["label"]]

        source_ids_raw = examples.get("source_id")
        if source_ids_raw is None:
            encoded["source_id"] = [f"row_{int(idx)}" for idx in indices]
        else:
            encoded["source_id"] = [str(x) for x in source_ids_raw]
        return encoded

    tokenized_dataset = raw_dataset.map(
        _tokenize_batch,
        batched=True,
        with_indices=True,
        batch_size=tokenizer_batch_size,
        num_proc=tokenize_num_proc if tokenize_num_proc > 1 else None,
        writer_batch_size=max(1000, tokenizer_batch_size),
        remove_columns=raw_dataset.column_names,
        desc=f"Tokenizing {split_name}",
        load_from_cache_file=not rebuild_tokenized_cache,
    )
    LOGGER.info(
        "Tokenized %s split completed (rows=%s, elapsed=%.1fs)",
        split_name,
        len(tokenized_dataset),
        time.time() - start_time,
    )
    return tokenized_dataset


def load_or_build_tokenized_dataset(
    split_name: str,
    csv_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    model_name_or_path: str,
    max_length: int,
    tokenizer_batch_size: int,
    tokenize_num_proc: int,
    tokenized_cache_dir: str,
    tokenized_cache_format: str,
    rebuild_tokenized_cache: bool,
    cache_dir: Optional[str],
) -> HFDataset:
    stage_start_time = time.time()
    cache_key = _build_cache_key(csv_path, model_name_or_path, max_length)
    split_root = os.path.join(tokenized_cache_dir, split_name)
    split_cache_root = os.path.join(split_root, cache_key)
    active_cache_root = split_cache_root
    active_cache_key = cache_key
    os.makedirs(split_root, exist_ok=True)
    tokenized_cache_format = _normalize_cache_format(tokenized_cache_format)
    lock_file = os.path.join(split_root, ".build.lock")

    with FileLock(lock_file):
        if tokenized_cache_format == "arrow":
            arrow_dir = os.path.join(split_cache_root, "arrow")
            if os.path.isdir(arrow_dir) and not rebuild_tokenized_cache:
                LOGGER.info("Loading tokenized %s split from Arrow cache: %s", split_name, arrow_dir)
                tokenized_dataset = load_from_disk(arrow_dir)
            else:
                reusable_arrow_dir = None
                if not rebuild_tokenized_cache:
                    reusable_arrow_dir = _find_portable_tokenized_cache(
                        split_root=split_root,
                        csv_path=csv_path,
                        model_name_or_path=model_name_or_path,
                        max_length=max_length,
                        cache_format=tokenized_cache_format,
                    )
                if reusable_arrow_dir is not None:
                    active_cache_root = os.path.dirname(reusable_arrow_dir)
                    active_cache_key = os.path.basename(active_cache_root)
                    LOGGER.info(
                        "Loading tokenized %s split from portable Arrow cache: %s",
                        split_name,
                        reusable_arrow_dir,
                    )
                    tokenized_dataset = load_from_disk(reusable_arrow_dir)
                else:
                    os.makedirs(split_cache_root, exist_ok=True)
                    active_cache_root = split_cache_root
                    active_cache_key = cache_key
                    if rebuild_tokenized_cache and os.path.isdir(arrow_dir):
                        shutil.rmtree(arrow_dir)
                    LOGGER.info("Building tokenized %s split and saving Arrow cache...", split_name)
                    tokenized_dataset = _build_tokenized_from_csv(
                        csv_path=csv_path,
                        split_name=split_name,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        tokenizer_batch_size=tokenizer_batch_size,
                        tokenize_num_proc=tokenize_num_proc,
                        cache_dir=cache_dir,
                        rebuild_tokenized_cache=rebuild_tokenized_cache,
                    )
                    tokenized_dataset.save_to_disk(arrow_dir)
        else:
            parquet_file = os.path.join(split_cache_root, "data.parquet")
            if os.path.isfile(parquet_file) and not rebuild_tokenized_cache:
                LOGGER.info("Loading tokenized %s split from Parquet cache: %s", split_name, parquet_file)
                tokenized_dataset = load_dataset("parquet", data_files=parquet_file, split="train", cache_dir=cache_dir)
            else:
                reusable_parquet_file = None
                if not rebuild_tokenized_cache:
                    reusable_parquet_file = _find_portable_tokenized_cache(
                        split_root=split_root,
                        csv_path=csv_path,
                        model_name_or_path=model_name_or_path,
                        max_length=max_length,
                        cache_format=tokenized_cache_format,
                    )
                if reusable_parquet_file is not None:
                    active_cache_root = os.path.dirname(reusable_parquet_file)
                    active_cache_key = os.path.basename(active_cache_root)
                    LOGGER.info(
                        "Loading tokenized %s split from portable Parquet cache: %s",
                        split_name,
                        reusable_parquet_file,
                    )
                    tokenized_dataset = load_dataset(
                        "parquet",
                        data_files=reusable_parquet_file,
                        split="train",
                        cache_dir=cache_dir,
                    )
                else:
                    os.makedirs(split_cache_root, exist_ok=True)
                    active_cache_root = split_cache_root
                    active_cache_key = cache_key
                    if rebuild_tokenized_cache and os.path.isfile(parquet_file):
                        os.remove(parquet_file)
                    LOGGER.info("Building tokenized %s split and writing Parquet cache...", split_name)
                    tokenized_dataset = _build_tokenized_from_csv(
                        csv_path=csv_path,
                        split_name=split_name,
                        tokenizer=tokenizer,
                        max_length=max_length,
                        tokenizer_batch_size=tokenizer_batch_size,
                        tokenize_num_proc=tokenize_num_proc,
                        cache_dir=cache_dir,
                        rebuild_tokenized_cache=rebuild_tokenized_cache,
                    )
                    tokenized_dataset.to_parquet(parquet_file)
                    tokenized_dataset = load_dataset("parquet", data_files=parquet_file, split="train", cache_dir=cache_dir)

    required_columns = {"input_ids", "attention_mask", "labels"}
    missing_columns = required_columns.difference(tokenized_dataset.column_names)
    if missing_columns:
        raise ValueError(f"Tokenized dataset missing required columns {sorted(missing_columns)}: {csv_path}")

    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True,
    )

    metadata = {
        "csv_path": os.path.abspath(csv_path),
        "csv_size": os.path.getsize(csv_path),
        "cache_format": tokenized_cache_format,
        "cache_key": active_cache_key,
        "model_name_or_path": model_name_or_path,
        "portable_model_ref": _portable_model_ref(model_name_or_path),
        "max_length": max_length,
        "tokenizer_batch_size": tokenizer_batch_size,
        "tokenize_num_proc": tokenize_num_proc,
        "num_rows": len(tokenized_dataset),
    }
    with open(os.path.join(active_cache_root, "meta.json"), "w") as handle:
        json.dump(metadata, handle, indent=2)

    LOGGER.info(
        "Tokenized %s split ready (rows=%s, elapsed=%.1fs)",
        split_name,
        len(tokenized_dataset),
        time.time() - stage_start_time,
    )

    return tokenized_dataset


class SupervisedDataset(TorchDataset):
    """Dataset for binary sequence classification with CSV input."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        tokenizer_batch_size: int = 4096,
    ):
        super().__init__()
        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        if tokenizer_batch_size <= 0:
            raise ValueError("tokenizer_batch_size must be > 0")

        seq_batch: List[str] = []
        label_batch: List[float] = []
        processed = 0

        with open(data_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            if "sequence" not in reader.fieldnames or "label" not in reader.fieldnames:
                raise ValueError(f"CSV must contain 'sequence' and 'label' columns: {data_path}")

            for row in reader:
                seq_batch.append(row["sequence"].strip().upper())
                label_batch.append(float(int(row["label"])))
                if len(seq_batch) >= tokenizer_batch_size:
                    self._append_encoded_batch(tokenizer, max_length, seq_batch, label_batch)
                    processed += len(seq_batch)
                    if processed % (tokenizer_batch_size * 50) == 0:
                        LOGGER.info("Tokenized %s samples from %s", processed, data_path)
                    seq_batch = []
                    label_batch = []

        if seq_batch:
            self._append_encoded_batch(tokenizer, max_length, seq_batch, label_batch)

        if not self.input_ids:
            raise ValueError(f"No usable samples found in {data_path}")

    def _append_encoded_batch(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int,
        sequences: Sequence[str],
        labels: Sequence[float],
    ) -> None:
        encoded = tokenizer(
            list(sequences),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        input_ids_list = encoded["input_ids"]
        attention_mask_list = encoded["attention_mask"]
        if isinstance(input_ids_list, torch.Tensor):
            input_ids_list = input_ids_list.tolist()
        if isinstance(attention_mask_list, torch.Tensor):
            attention_mask_list = attention_mask_list.tolist()

        for input_ids, attention_mask, label in zip(input_ids_list, attention_mask_list, labels):
            self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            self.attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [x["input_ids"] for x in instances]
        attention_mask = [x["attention_mask"] for x in instances]
        labels = torch.stack([x["labels"] for x in instances]).float()

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PlannedBatchSampler(Sampler[List[int]]):
    """Yield precomputed balanced batches of dataset row indices."""

    def __init__(self, plan_path: str, expected_batch_size: int, dataset_size: int):
        plan = np.load(plan_path, allow_pickle=False)
        if plan.ndim != 2:
            raise ValueError(f"Batch plan must be a 2D array, got shape={plan.shape}: {plan_path}")
        if plan.shape[1] != expected_batch_size:
            raise ValueError(
                f"Batch plan batch_size={plan.shape[1]} does not match per_device_train_batch_size="
                f"{expected_batch_size}: {plan_path}"
            )
        if plan.size == 0:
            raise ValueError(f"Batch plan is empty: {plan_path}")
        if int(plan.min()) < 0 or int(plan.max()) >= dataset_size:
            raise ValueError(
                f"Batch plan index range [{int(plan.min())}, {int(plan.max())}] is outside dataset size "
                f"{dataset_size}: {plan_path}"
            )
        self.plan = plan.astype(np.int64, copy=False)

    def __iter__(self):
        for batch in self.plan:
            yield batch.tolist()

    def __len__(self) -> int:
        return int(self.plan.shape[0])


def _binary_logits(logits: torch.Tensor) -> torch.Tensor:
    """Map model logits to a single binary logit for BCE.

    If model outputs two logits, use logit margin (positive - negative).
    """
    if logits.ndim == 2 and logits.size(-1) == 2:
        return logits[:, 1] - logits[:, 0]
    if logits.ndim == 2 and logits.size(-1) == 1:
        return logits[:, 0]
    return logits.view(-1)


class BCETrainer(transformers.Trainer):
    """Trainer with explicit Adam optimizer and BCE loss."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._logged_first_batch_device = False

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

    def get_train_dataloader(self) -> DataLoader:
        plan_path = str(getattr(self.args, "train_batch_plan_path", "") or "").strip()
        if not plan_path:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset when using train_batch_plan_path")
        if not os.path.isfile(plan_path):
            raise FileNotFoundError(f"train_batch_plan_path not found: {plan_path}")

        batch_sampler = PlannedBatchSampler(
            plan_path=plan_path,
            expected_batch_size=int(self.args.per_device_train_batch_size),
            dataset_size=len(self.train_dataset),
        )
        LOGGER.info(
            "Using planned train batches from %s (batches=%s, batch_size=%s); group_by_length is bypassed.",
            plan_path,
            len(batch_sampler),
            self.args.per_device_train_batch_size,
        )

        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        return self.accelerator.prepare(dataloader)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        if not self._logged_first_batch_device:
            labels_preview = inputs.get("labels")
            LOGGER.info(
                "First batch device: model=%s input_ids=%s attention_mask=%s labels=%s",
                next(model.parameters()).device,
                inputs["input_ids"].device if "input_ids" in inputs else "n/a",
                inputs["attention_mask"].device if "attention_mask" in inputs else "n/a",
                labels_preview.device if isinstance(labels_preview, torch.Tensor) else "n/a",
            )
            self._logged_first_batch_device = True

        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = _binary_logits(outputs.logits)
        loss = BCEWithLogitsLoss()(logits, labels)
        return (loss, outputs) if return_outputs else loss


def preprocess_logits_for_metrics(
    logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]],
    _: Any,
) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    return _binary_logits(logits).detach()


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    del eval_pred
    raise RuntimeError("compute_metrics should be created via build_compute_metrics(...) at runtime")


def _compute_binary_metrics_from_probs(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    preds = (probs >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / total

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def build_compute_metrics(
    eval_source_ids: Optional[List[str]],
    threshold: float,
):
    """Build metric function aligned with deployment: aggregate fragments by source_id."""

    def _compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
        labels = np.asarray(labels).astype(np.int64)

        fragment_metrics = _compute_binary_metrics_from_probs(probs, labels, threshold=threshold)

        if not eval_source_ids or len(eval_source_ids) != len(probs):
            if eval_source_ids is not None:
                LOGGER.warning(
                    "Falling back to fragment-level metrics because source_id count (%s) != prediction count (%s)",
                    len(eval_source_ids),
                    len(probs),
                )
            return fragment_metrics

        per_source_probs: Dict[str, List[float]] = {}
        per_source_label: Dict[str, int] = {}
        conflicting_label_sources = 0

        for sid, prob, label in zip(eval_source_ids, probs.tolist(), labels.tolist()):
            sid = str(sid)
            per_source_probs.setdefault(sid, []).append(float(prob))
            label_i = int(label)
            if sid in per_source_label and per_source_label[sid] != label_i:
                conflicting_label_sources += 1
            else:
                per_source_label[sid] = label_i

        if conflicting_label_sources > 0:
            LOGGER.warning(
                "Detected %s source_ids with inconsistent fragment labels in eval set; using first observed label.",
                conflicting_label_sources,
            )

        source_ids = list(per_source_probs.keys())
        # Deployment aggregation: fragment probabilities from the same source_id
        # are averaged into one sequence-level probability. This aggregation is
        # metric/inference-only and does not participate in training gradients.
        seq_probs = np.asarray([np.mean(per_source_probs[sid]) for sid in source_ids], dtype=np.float32)
        seq_labels = np.asarray([per_source_label[sid] for sid in source_ids], dtype=np.int64)

        seq_metrics = _compute_binary_metrics_from_probs(seq_probs, seq_labels, threshold=threshold)

        # Keep legacy metric names mapped to sequence-level values so checkpoint selection
        # follows deployment-aligned sequence aggregation.
        return {
            "accuracy": seq_metrics["accuracy"],
            "f1": seq_metrics["f1"],
            "precision": seq_metrics["precision"],
            "recall": seq_metrics["recall"],
            "seq_accuracy": seq_metrics["accuracy"],
            "seq_f1": seq_metrics["f1"],
            "seq_precision": seq_metrics["precision"],
            "seq_recall": seq_metrics["recall"],
            "fragment_accuracy": fragment_metrics["accuracy"],
            "fragment_f1": fragment_metrics["f1"],
            "fragment_precision": fragment_metrics["precision"],
            "fragment_recall": fragment_metrics["recall"],
            "eval_threshold": float(threshold),
            "eval_num_sequences": float(len(source_ids)),
            "eval_num_fragments": float(len(probs)),
        }

    return _compute_metrics


def train() -> None:
    faulthandler.enable(file=sys.stderr, all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.tokenizer_batch_size <= 0:
        raise ValueError("tokenizer_batch_size must be > 0")
    if training_args.tokenize_num_proc <= 0:
        raise ValueError("tokenize_num_proc must be > 0")
    if not (0.0 <= training_args.sequence_eval_threshold <= 1.0):
        raise ValueError("sequence_eval_threshold must be in [0, 1]")
    training_args.tokenized_cache_format = _normalize_cache_format(training_args.tokenized_cache_format)

    if abs(training_args.learning_rate - LOCKED_LEARNING_RATE) > 1e-12:
        LOGGER.warning(
            "Overriding learning_rate=%s to locked value %s",
            training_args.learning_rate,
            LOCKED_LEARNING_RATE,
        )
    training_args.learning_rate = LOCKED_LEARNING_RATE

    if training_args.warmup_steps != LOCKED_WARMUP_STEPS:
        LOGGER.warning(
            "Overriding warmup_steps=%s to locked value %s",
            training_args.warmup_steps,
            LOCKED_WARMUP_STEPS,
        )
    training_args.warmup_steps = LOCKED_WARMUP_STEPS

    eval_strategy_value = (
        training_args.evaluation_strategy.value
        if hasattr(training_args.evaluation_strategy, "value")
        else str(training_args.evaluation_strategy)
    )
    if eval_strategy_value.lower() != "epoch":
        LOGGER.warning(
            "Overriding evaluation_strategy=%s to 'epoch' for paper-aligned convergence schedule.",
            training_args.evaluation_strategy,
        )
    training_args.evaluation_strategy = "epoch"
    training_args.eval_strategy = "epoch"
    save_strategy_value = (
        training_args.save_strategy.value
        if hasattr(training_args.save_strategy, "value")
        else str(training_args.save_strategy)
    )
    if save_strategy_value.lower() != "epoch":
        LOGGER.warning(
            "Overriding save_strategy=%s to 'epoch' to stay consistent with load_best_model_at_end.",
            training_args.save_strategy,
        )
    training_args.save_strategy = "epoch"

    cuda_available = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count() if cuda_available else 0
    if training_args.require_cuda and not cuda_available:
        raise RuntimeError(
            "`require_cuda=True` but CUDA is unavailable. "
            "Please check your driver/CUDA runtime and CUDA_VISIBLE_DEVICES."
        )
    if training_args.fp16 and not cuda_available:
        raise RuntimeError("`fp16=True` requires CUDA, but CUDA is unavailable.")

    if cuda_available:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "benchmark"):
            torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    train_csv = os.path.join(data_args.data_path, "train.csv")
    dev_csv = os.path.join(data_args.data_path, "dev.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv: {train_csv}")
    if not os.path.exists(dev_csv):
        raise FileNotFoundError(f"Missing dev.csv: {dev_csv}")

    LOGGER.info("Training with Adam optimizer + BCEWithLogitsLoss")
    LOGGER.info("Configured learning rate=%s, warmup_steps=%s", training_args.learning_rate, training_args.warmup_steps)
    LOGGER.info(
        "Runtime: cuda_available=%s, cuda_count=%s, fp16=%s, batch(train/eval)=%s/%s, workers=%s, pin_memory=%s, tokenizer_batch=%s, tokenize_num_proc=%s",
        cuda_available,
        cuda_count,
        training_args.fp16,
        training_args.per_device_train_batch_size,
        training_args.per_device_eval_batch_size,
        training_args.dataloader_num_workers,
        training_args.dataloader_pin_memory,
        training_args.tokenizer_batch_size,
        training_args.tokenize_num_proc,
    )
    LOGGER.info(
        "Policy lock: learning_rate=%s, warmup_steps=%s, evaluation_strategy=%s, save_strategy=%s",
        training_args.learning_rate,
        training_args.warmup_steps,
        training_args.evaluation_strategy,
        training_args.save_strategy,
    )
    LOGGER.info(
        "CUDA_VISIBLE_DEVICES=%s, TOKENIZERS_PARALLELISM=%s, torch_num_threads=%s",
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        os.environ.get("TOKENIZERS_PARALLELISM"),
        torch.get_num_threads(),
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token

    tokenized_cache_dir = training_args.tokenized_cache_dir.strip() or os.path.join(data_args.data_path, "tokenized_cache")
    os.makedirs(tokenized_cache_dir, exist_ok=True)
    LOGGER.info(
        "Token cache: dir=%s, format=%s, rebuild=%s",
        tokenized_cache_dir,
        training_args.tokenized_cache_format,
        training_args.rebuild_tokenized_cache,
    )

    train_dataset = load_or_build_tokenized_dataset(
        split_name="train",
        csv_path=train_csv,
        tokenizer=tokenizer,
        model_name_or_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        tokenizer_batch_size=training_args.tokenizer_batch_size,
        tokenize_num_proc=training_args.tokenize_num_proc,
        tokenized_cache_dir=tokenized_cache_dir,
        tokenized_cache_format=training_args.tokenized_cache_format,
        rebuild_tokenized_cache=training_args.rebuild_tokenized_cache,
        cache_dir=training_args.cache_dir,
    )
    val_dataset = load_or_build_tokenized_dataset(
        split_name="dev",
        csv_path=dev_csv,
        tokenizer=tokenizer,
        model_name_or_path=model_args.model_name_or_path,
        max_length=training_args.model_max_length,
        tokenizer_batch_size=training_args.tokenizer_batch_size,
        tokenize_num_proc=training_args.tokenize_num_proc,
        tokenized_cache_dir=tokenized_cache_dir,
        tokenized_cache_format=training_args.tokenized_cache_format,
        rebuild_tokenized_cache=training_args.rebuild_tokenized_cache,
        cache_dir=training_args.cache_dir,
    )

    # Sequence-level eval requires source_id; old token caches may not include it.
    if "source_id" not in train_dataset.column_names or "source_id" not in val_dataset.column_names:
        LOGGER.warning(
            "Tokenized cache missing source_id; rebuilding tokenized cache to enable sequence-level metrics."
        )
        train_dataset = load_or_build_tokenized_dataset(
            split_name="train",
            csv_path=train_csv,
            tokenizer=tokenizer,
            model_name_or_path=model_args.model_name_or_path,
            max_length=training_args.model_max_length,
            tokenizer_batch_size=training_args.tokenizer_batch_size,
            tokenize_num_proc=training_args.tokenize_num_proc,
            tokenized_cache_dir=tokenized_cache_dir,
            tokenized_cache_format=training_args.tokenized_cache_format,
            rebuild_tokenized_cache=True,
            cache_dir=training_args.cache_dir,
        )
        val_dataset = load_or_build_tokenized_dataset(
            split_name="dev",
            csv_path=dev_csv,
            tokenizer=tokenizer,
            model_name_or_path=model_args.model_name_or_path,
            max_length=training_args.model_max_length,
            tokenizer_batch_size=training_args.tokenizer_batch_size,
            tokenize_num_proc=training_args.tokenize_num_proc,
            tokenized_cache_dir=tokenized_cache_dir,
            tokenized_cache_format=training_args.tokenized_cache_format,
            rebuild_tokenized_cache=True,
            cache_dir=training_args.cache_dir,
        )

    if "source_id" not in train_dataset.column_names or "source_id" not in val_dataset.column_names:
        raise ValueError(
            "source_id column is required for sequence-level evaluation but is missing from tokenized dataset."
        )

    val_source_ids = [str(x) for x in val_dataset["source_id"]]
    metric_fn = build_compute_metrics(
        eval_source_ids=val_source_ids,
        threshold=training_args.sequence_eval_threshold,
    )

    LOGGER.info(
        "Evaluation metrics aligned to deployment: sequence-level mean aggregation by source_id, threshold=%s",
        training_args.sequence_eval_threshold,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        num_labels=2,
        trust_remote_code=True,
    )

    trainer = BCETrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=metric_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    LOGGER.info(
        "Trainer resolved device=%s, n_gpu=%s, local_rank=%s",
        trainer.args.device,
        trainer.args.n_gpu,
        trainer.args.local_rank,
    )
    LOGGER.info("Model parameter device before train()=%s", next(trainer.model.parameters()).device)

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        os.makedirs(results_path, exist_ok=True)
        results = trainer.evaluate(eval_dataset=val_dataset)
        with open(os.path.join(results_path, "val_results.json"), "w") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    train()
