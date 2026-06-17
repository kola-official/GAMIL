#!/usr/bin/env python3
"""Shared utilities for Realm-Rank six-model training and benchmark scripts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
from torch.utils.data import Dataset

from experiment_config import CUSTOM_CODE_FILES, REQUIRED_STAGED_FILES, TOKENIZER_FILES

try:
    from datasets import Dataset as HFDataset
    from datasets import load_dataset, load_from_disk
except Exception:  # pragma: no cover - training env has datasets; default env may not.
    HFDataset = Any  # type: ignore
    load_dataset = None  # type: ignore
    load_from_disk = None  # type: ignore


LOGGER = logging.getLogger(__name__)
SUPPORTED_TOKENIZED_CACHE_FORMATS = {"arrow", "parquet"}
HEADER_FIELD_RE = re.compile(r"(\S+)=([^\s]+)")


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def str2bool(value: Union[str, bool, int]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def require_file(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def require_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(str(path))
    return path


def read_json(path: Union[str, Path]) -> dict:
    with open(path) as handle:
        return json.load(handle)


def write_json(path: Union[str, Path], payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_staged_teacher(name: str, model_dir: Union[str, Path]) -> None:
    model_dir = require_dir(model_dir)
    missing = [item for item in REQUIRED_STAGED_FILES if not (model_dir / item).is_file()]
    if missing:
        raise FileNotFoundError(f"{name} staged teacher missing files: {missing}")

    if name == "viralm_r":
        state_path = model_dir / "trainer_state.json"
        require_file(state_path)
        state = read_json(state_path)
        best = str(state.get("best_model_checkpoint", ""))
        if "checkpoint-48740" not in best:
            raise ValueError(
                f"{name} best checkpoint should be checkpoint-48740, got: {best or '<missing>'}"
            )


def validate_all_teachers(teacher_paths: Dict[str, Union[str, Path]]) -> None:
    for name, path in teacher_paths.items():
        validate_staged_teacher(name, path)


def copy_files_if_present(src_dir: Union[str, Path], dst_dir: Union[str, Path], names: Iterable[str]) -> None:
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, dst_dir / name)


def copy_custom_model_files(src_dir: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    copy_files_if_present(src_dir, dst_dir, CUSTOM_CODE_FILES)


def copy_tokenizer_files(src_dir: Union[str, Path], dst_dir: Union[str, Path]) -> None:
    copy_files_if_present(src_dir, dst_dir, TOKENIZER_FILES)


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
            metadata = read_json(meta_path)
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


def _require_datasets() -> None:
    if load_dataset is None or load_from_disk is None:
        raise RuntimeError("The 'datasets' package is required. Run with the vl conda environment.")


def _build_tokenized_from_csv(
    csv_path: str,
    split_name: str,
    tokenizer: Any,
    max_length: int,
    tokenizer_batch_size: int,
    tokenize_num_proc: int,
    cache_dir: Optional[str],
    rebuild_tokenized_cache: bool,
) -> HFDataset:
    _require_datasets()
    start_time = time.time()
    raw_dataset = load_dataset("csv", data_files=csv_path, split="train", cache_dir=cache_dir)  # type: ignore[misc]
    if "sequence" not in raw_dataset.column_names or "label" not in raw_dataset.column_names:
        raise ValueError(f"CSV must contain 'sequence' and 'label' columns: {csv_path}")

    LOGGER.info(
        "Tokenizing %s from %s (rows=%s, batch_size=%s, num_proc=%s)",
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
        encoded = tokenizer(sequences, truncation=True, max_length=max_length, padding=False)
        encoded["labels"] = [_to_binary_float(label) for label in examples["label"]]
        source_ids_raw = examples.get("source_id")
        genomes_raw = examples.get("genome")
        sources_raw = examples.get("source")
        if source_ids_raw is None:
            encoded["source_id"] = [f"row_{int(idx)}" for idx in indices]
        else:
            encoded["source_id"] = [str(x) for x in source_ids_raw]
        if genomes_raw is not None:
            encoded["genome"] = [str(x) for x in genomes_raw]
        if sources_raw is not None:
            encoded["source"] = [str(x) for x in sources_raw]
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
        "Tokenized %s completed (rows=%s, elapsed=%.1fs)",
        split_name,
        len(tokenized_dataset),
        time.time() - start_time,
    )
    return tokenized_dataset


def load_or_build_tokenized_dataset(
    split_name: str,
    csv_path: str,
    tokenizer: Any,
    model_name_or_path: str,
    max_length: int,
    tokenizer_batch_size: int,
    tokenize_num_proc: int,
    tokenized_cache_dir: str,
    tokenized_cache_format: str,
    rebuild_tokenized_cache: bool,
    cache_dir: Optional[str],
) -> HFDataset:
    _require_datasets()
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
                LOGGER.info("Loading tokenized %s from Arrow cache: %s", split_name, arrow_dir)
                tokenized_dataset = load_from_disk(arrow_dir)  # type: ignore[misc]
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
                    LOGGER.info("Loading tokenized %s from portable Arrow cache: %s", split_name, reusable_arrow_dir)
                    tokenized_dataset = load_from_disk(reusable_arrow_dir)  # type: ignore[misc]
                else:
                    os.makedirs(split_cache_root, exist_ok=True)
                    active_cache_root = split_cache_root
                    active_cache_key = cache_key
                    if rebuild_tokenized_cache and os.path.isdir(arrow_dir):
                        shutil.rmtree(arrow_dir)
                    tokenized_dataset = _build_tokenized_from_csv(
                        csv_path,
                        split_name,
                        tokenizer,
                        max_length,
                        tokenizer_batch_size,
                        tokenize_num_proc,
                        cache_dir,
                        rebuild_tokenized_cache,
                    )
                    tokenized_dataset.save_to_disk(arrow_dir)
        else:
            parquet_file = os.path.join(split_cache_root, "data.parquet")
            if os.path.isfile(parquet_file) and not rebuild_tokenized_cache:
                LOGGER.info("Loading tokenized %s from Parquet cache: %s", split_name, parquet_file)
                tokenized_dataset = load_dataset("parquet", data_files=parquet_file, split="train", cache_dir=cache_dir)  # type: ignore[misc]
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
                    tokenized_dataset = load_dataset("parquet", data_files=reusable_parquet_file, split="train", cache_dir=cache_dir)  # type: ignore[misc]
                else:
                    os.makedirs(split_cache_root, exist_ok=True)
                    active_cache_root = split_cache_root
                    active_cache_key = cache_key
                    if rebuild_tokenized_cache and os.path.isfile(parquet_file):
                        os.remove(parquet_file)
                    tokenized_dataset = _build_tokenized_from_csv(
                        csv_path,
                        split_name,
                        tokenizer,
                        max_length,
                        tokenizer_batch_size,
                        tokenize_num_proc,
                        cache_dir,
                        rebuild_tokenized_cache,
                    )
                    tokenized_dataset.to_parquet(parquet_file)
                    tokenized_dataset = load_dataset("parquet", data_files=parquet_file, split="train", cache_dir=cache_dir)  # type: ignore[misc]

    required_columns = {"input_ids", "attention_mask", "labels", "source_id"}
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
    write_json(os.path.join(active_cache_root, "meta.json"), metadata)
    LOGGER.info(
        "Tokenized %s ready (rows=%s, elapsed=%.1fs)",
        split_name,
        len(tokenized_dataset),
        time.time() - stage_start_time,
    )
    return tokenized_dataset


class DataCollatorForSupervisedDataset:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def subset_hf_dataset_by_sources(dataset: HFDataset, max_sources: int, max_rows: int = 0) -> HFDataset:
    if max_sources <= 0 and max_rows <= 0:
        return dataset
    selected: List[int] = []
    seen = set()
    source_ids = [str(x) for x in dataset["source_id"]]
    for idx, sid in enumerate(source_ids):
        if max_sources > 0 and sid not in seen and len(seen) >= max_sources:
            continue
        seen.add(sid)
        selected.append(idx)
        if max_rows > 0 and len(selected) >= max_rows:
            break
    if not selected:
        raise ValueError("smoke subset selection produced no rows")
    LOGGER.info(
        "Subset dataset from rows=%s sources=%s to rows=%s sources=%s",
        len(dataset),
        len(set(source_ids)),
        len(selected),
        len(seen),
    )
    return dataset.select(selected)


class SequenceGroupedDataset(Dataset):
    """Group tokenized CSV fragments by source_id/genome for MIL."""

    def __init__(self, hf_dataset: HFDataset, max_groups: int = 0):
        self.hf_dataset = hf_dataset
        self.sid_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, sid in enumerate(hf_dataset["source_id"]):
            sid_str = str(sid)
            if max_groups > 0 and sid_str not in self.sid_to_indices and len(self.sid_to_indices) >= max_groups:
                continue
            self.sid_to_indices[sid_str].append(idx)
        self.unique_sids = list(self.sid_to_indices.keys())
        if not self.unique_sids:
            raise ValueError("no source groups found")
        LOGGER.info(
            "Grouped %s fragments into %s source groups",
            sum(len(v) for v in self.sid_to_indices.values()),
            len(self.unique_sids),
        )

    def __len__(self) -> int:
        return len(self.unique_sids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid = self.unique_sids[idx]
        indices = self.sid_to_indices[sid]
        fragments = self.hf_dataset[indices]
        return {
            "input_ids": fragments["input_ids"],
            "attention_mask": fragments["attention_mask"],
            "label": float(fragments["labels"][0]),
            "source_id": sid,
        }


def sequence_collate_fn(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    input_ids = []
    attention_mask = []
    labels = []
    source_ids = []
    for item in batch:
        ids_tensors = [torch.as_tensor(x, dtype=torch.long) for x in item["input_ids"]]
        mask_tensors = [torch.as_tensor(x, dtype=torch.long) for x in item["attention_mask"]]
        input_ids.append(torch.nn.utils.rnn.pad_sequence(ids_tensors, batch_first=True, padding_value=0))
        attention_mask.append(torch.nn.utils.rnn.pad_sequence(mask_tensors, batch_first=True, padding_value=0))
        labels.append(float(item["label"]))
        source_ids.append(str(item["source_id"]))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.as_tensor(labels, dtype=torch.float),
        "source_ids": source_ids,
    }


def _binary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.size(-1) == 2:
        return logits[:, 1] - logits[:, 0]
    if logits.ndim == 2 and logits.size(-1) == 1:
        return logits[:, 0]
    return logits.view(-1)


def last_hidden(hidden_states: Any) -> Optional[torch.Tensor]:
    if hidden_states is None:
        return None
    if isinstance(hidden_states, (tuple, list)):
        return hidden_states[-1]
    return hidden_states


def compute_binary_metrics_from_probs(
    probs: Union[np.ndarray, Sequence[float]],
    labels: Union[np.ndarray, Sequence[int]],
    threshold: float = 0.5,
) -> Dict[str, float]:
    probs_arr = np.asarray(probs, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    preds = (probs_arr >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels_arr == 1)))
    tn = int(np.sum((preds == 0) & (labels_arr == 0)))
    fp = int(np.sum((preds == 1) & (labels_arr == 0)))
    fn = int(np.sum((preds == 0) & (labels_arr == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    out = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_score": f1,
        "accuracy": accuracy,
    }
    out.update(compute_auc_metrics(probs_arr, labels_arr))
    return out


def compute_auc_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auroc": float(roc_auc_score(labels, probs)),
            "auprc": float(average_precision_score(labels, probs)),
        }
    except Exception:
        return {"auroc": _fallback_auroc(probs, labels), "auprc": _fallback_auprc(probs, labels)}


def _fallback_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(probs)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(probs) + 1)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fallback_auprc(probs: np.ndarray, labels: np.ndarray) -> float:
    positives = int((labels == 1).sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-probs)
    sorted_labels = labels[order]
    tp = 0
    precisions = []
    for rank, label in enumerate(sorted_labels, start=1):
        if int(label) == 1:
            tp += 1
            precisions.append(tp / rank)
    return float(np.sum(precisions) / positives)


def build_compute_metrics(eval_source_ids: Optional[List[str]], threshold: float):
    def _compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
        labels_arr = np.asarray(labels).astype(np.int64)
        fragment_metrics = compute_binary_metrics_from_probs(probs, labels_arr, threshold=threshold)
        if not eval_source_ids or len(eval_source_ids) != len(probs):
            return {
                "accuracy": fragment_metrics["accuracy"],
                "f1": fragment_metrics["f1"],
                "precision": fragment_metrics["precision"],
                "recall": fragment_metrics["recall"],
                "fragment_f1": fragment_metrics["f1"],
            }
        per_source_probs: Dict[str, List[float]] = {}
        per_source_label: Dict[str, int] = {}
        for sid, prob, label in zip(eval_source_ids, probs.tolist(), labels_arr.tolist()):
            sid = str(sid)
            per_source_probs.setdefault(sid, []).append(float(prob))
            per_source_label.setdefault(sid, int(label))
        source_ids = list(per_source_probs.keys())
        seq_probs = np.asarray([np.mean(per_source_probs[sid]) for sid in source_ids], dtype=np.float32)
        seq_labels = np.asarray([per_source_label[sid] for sid in source_ids], dtype=np.int64)
        seq_metrics = compute_binary_metrics_from_probs(seq_probs, seq_labels, threshold=threshold)
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
            "eval_num_sequences": float(len(source_ids)),
            "eval_num_fragments": float(len(probs)),
        }

    return _compute_metrics


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple[torch.Tensor, Any]], _: Any) -> torch.Tensor:
    if isinstance(logits, tuple):
        logits = logits[0]
    return _binary_logits(logits).detach()


def safe_save_model_for_hf_trainer(trainer: Any, output_dir: str) -> None:
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


def fasta_opener(path: Union[str, Path]):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def parse_realm_rank_header(header_line: str) -> Dict[str, str]:
    header = header_line[1:].strip() if header_line.startswith(">") else header_line.strip()
    parts = header.split()
    record_id = parts[0]
    fields = {match.group(1): match.group(2) for match in HEADER_FIELD_RE.finditer(header)}
    fields["record_id"] = record_id
    fields.setdefault("source_id", fields.get("genome", record_id))
    fields.setdefault("genome", fields["source_id"])
    fields.setdefault("contig", fields.get("genome", record_id))
    fields.setdefault("source", "")
    fields.setdefault("label", "")
    fields.setdefault("start", "")
    fields.setdefault("end", "")
    fields.setdefault("length", "")
    fields["binary_label"] = "1" if is_viral_header(fields) else "0"
    return fields


def is_viral_header(fields: Dict[str, str]) -> bool:
    source = str(fields.get("source", "")).lower()
    label = str(fields.get("label", "")).lower()
    supergroup = str(fields.get("supergroup", "")).lower()
    return source == "virus" or supergroup == "virus" or label == "virus"


def iter_fasta_records(path: Union[str, Path]) -> Iterator[Tuple[Dict[str, str], str]]:
    meta: Optional[Dict[str, str]] = None
    seq_chunks: List[str] = []
    with fasta_opener(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if meta is not None:
                    yield meta, "".join(seq_chunks).upper()
                meta = parse_realm_rank_header(line)
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if meta is not None:
            yield meta, "".join(seq_chunks).upper()


def write_csv(path: Union[str, Path], rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    return count


def atomic_touch(path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")

