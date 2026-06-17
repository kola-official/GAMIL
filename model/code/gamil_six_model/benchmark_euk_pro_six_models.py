#!/usr/bin/env python3
"""Benchmark six trained models on the Realm-Rank euk/pro FASTA suite."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from experiment_config import MODEL_ROOT, OUTPUT_ROOT, PROCESSED_DATA_ROOT
from mil_model import ViraLM_MIL_Gated
from mil_train_common import autocast_context
from shared import _binary_logits, compute_auc_metrics, iter_fasta_records, str2bool, write_csv, write_json


DEFAULT_DATA_ROOT = PROCESSED_DATA_ROOT / "realm_rank_benchmark"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "benchmark_euk_pro"
DEFAULT_MODELS = [
    "viralm_o_6l_meanpool_kd",
    "viralm_r_6l_meanpool_kd",
    "viralm_o_6l_gated_mil_kd",
    "viralm_r_6l_gated_mil_kd",
    "viralm_o_12l_gated_mil",
    "viralm_r_12l_gated_mil",
]
DEFAULT_BENCHMARKS = ["bench-euk", "bench-pro"]
DEFAULT_LENGTHS = ["500", "1000", "2000", "10000", "20000", "mixed"]
VALID_DNA_RE = re.compile(r"[^ACGT]")

SEQUENCE_FIELDS = [
    "model_name",
    "model_kind",
    "benchmark",
    "fasta_stem",
    "fasta_file",
    "seq_name",
    "class_label",
    "label",
    "source",
    "supergroup",
    "realm",
    "length",
    "record_length",
    "prediction",
    "virus_score",
    "fragment_count",
    "missing_prediction",
]

FRAGMENT_FIELDS = [
    "model_name",
    "model_kind",
    "benchmark",
    "fasta_stem",
    "fasta_file",
    "seq_name",
    "fragment_name",
    "fragment_index",
    "start",
    "end",
    "fragment_length",
    "class_label",
    "label",
    "source",
    "supergroup",
    "realm",
    "prediction",
    "virus_score",
]

SEQUENCE_METRIC_FIELDS = [
    "model_name",
    "model_kind",
    "benchmark",
    "fasta_stem",
    "fasta_file",
    "level",
    "total_positive",
    "total_negative",
    "evaluated_total",
    "missing_total",
    "TP",
    "FP",
    "FN",
    "TN",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "result_csv",
    "auroc",
    "auprc",
]

FRAGMENT_METRIC_FIELDS = [
    "model_name",
    "model_kind",
    "benchmark",
    "fasta_stem",
    "fasta_file",
    "level",
    "total_positive_fragments",
    "total_negative_fragments",
    "TP",
    "FP",
    "FN",
    "TN",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "fragment_result_csv",
    "auroc",
    "auprc",
]


@dataclass(frozen=True)
class FastaSpec:
    benchmark: str
    length: str
    fasta_path: Path
    fasta_file: str
    fasta_stem: str
    expected_total: int = 0
    expected_positive: int = 0
    expected_negative: int = 0


class FragmentDataset(Dataset):
    def __init__(self, fragments: Sequence[Dict[str, Any]]):
        self.fragments = list(fragments)

    def __len__(self) -> int:
        return len(self.fragments)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.fragments[index]


class FragmentCollator:
    def __init__(self, tokenizer: Any, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized = self.tokenizer(
            [row["sequence"] for row in batch],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.model_max_length,
        )
        tokenized["meta"] = list(batch)
        return tokenized


class SequenceBagDataset(Dataset):
    def __init__(self, entries: Sequence[Dict[str, Any]]):
        self.entries = [entry for entry in entries if entry["fragments"]]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.entries[index]


class SequenceBagCollator:
    def __init__(self, tokenizer: Any, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        for entry in batch:
            tokenized = self.tokenizer(
                [fragment["sequence"] for fragment in entry["fragments"]],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.model_max_length,
            )
            input_ids.append(tokenized["input_ids"])
            attention_mask.append(tokenized["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "entries": list(batch)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark six trained models on Realm-Rank euk/pro FASTAs")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--lengths", nargs="+", default=DEFAULT_LENGTHS)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mil-batch-size", type=int, default=8)
    parser.add_argument("--scan-chunk", type=int, default=int(os.environ.get("SCAN_CHUNK", "48")))
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fp16", default="True")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=0, help="Debug limit per FASTA; 0 means all records")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-combined-predictions", action="store_true")
    return parser.parse_args()


def fasta_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".fasta.gz"):
        return name[: -len(".fasta.gz")]
    if name.endswith(".fa.gz"):
        return name[: -len(".fa.gz")]
    return path.stem


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_expected_counts(data_root: Path) -> Dict[Tuple[str, str], Tuple[int, int, int]]:
    counts_path = data_root / "qc" / "benchmark_counts.tsv"
    if not counts_path.is_file():
        return {}
    out: Dict[Tuple[str, str], Tuple[int, int, int]] = {}
    for row in read_tsv(counts_path):
        out[(row["benchmark"], row["file"])] = (
            int(row["total_count"]),
            int(row["positive_count"]),
            int(row["negative_count"]),
        )
    return out


def benchmark_fastas(data_root: Path, benchmarks: Sequence[str], lengths: Sequence[str]) -> List[FastaSpec]:
    if not data_root.is_dir():
        raise FileNotFoundError(str(data_root))
    expected = load_expected_counts(data_root)
    specs: List[FastaSpec] = []
    for benchmark in benchmarks:
        for length in lengths:
            fasta_path = data_root / benchmark / f"{benchmark}-{length}.fasta.gz"
            if not fasta_path.is_file():
                raise FileNotFoundError(str(fasta_path))
            total, pos, neg = expected.get((benchmark, fasta_path.name), (0, 0, 0))
            specs.append(
                FastaSpec(
                    benchmark=benchmark,
                    length=length,
                    fasta_path=fasta_path,
                    fasta_file=fasta_path.name,
                    fasta_stem=fasta_stem(fasta_path),
                    expected_total=total,
                    expected_positive=pos,
                    expected_negative=neg,
                )
            )
    return specs


def model_spec(model_name: str, model_root: Path) -> Tuple[str, Path]:
    model_dir = model_root / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(str(model_dir))
    if (model_dir / "best_mil_model.pt").is_file():
        return "mil", model_dir
    if (model_dir / "pytorch_model.bin").is_file():
        return "meanpool", model_dir
    raise FileNotFoundError(f"no supported checkpoint found under {model_dir}")


def is_valid_dna(sequence: str) -> bool:
    return VALID_DNA_RE.search(sequence) is None


def binary_label_from_meta(meta: Dict[str, str]) -> int:
    class_label = str(meta.get("class_label", "")).lower()
    if class_label == "positive":
        return 1
    if class_label == "negative":
        return 0
    return int(meta.get("binary_label", "0"))


def base_meta(meta: Dict[str, str], spec: FastaSpec, sequence: str) -> Dict[str, Any]:
    binary_label = binary_label_from_meta(meta)
    return {
        "benchmark": spec.benchmark,
        "fasta_stem": spec.fasta_stem,
        "fasta_file": spec.fasta_file,
        "seq_name": meta["record_id"],
        "class_label": "positive" if binary_label == 1 else "negative",
        "label": meta.get("label", ""),
        "source": meta.get("source", ""),
        "supergroup": meta.get("supergroup", ""),
        "realm": meta.get("realm", ""),
        "length": meta.get("length", ""),
        "record_length": len(sequence),
        "binary_label": binary_label,
    }


def split_fragments(
    entry_meta: Dict[str, Any],
    sequence: str,
    min_len: int = 500,
    fragment_len: int = 2000,
    min_tail_len: int = 500,
) -> List[Dict[str, Any]]:
    sequence = sequence.upper()
    seq_len = len(sequence)
    if seq_len < min_len:
        return []

    fragments: List[Dict[str, Any]] = []

    def add_fragment(fragment_index: int, start: int, end: int) -> int:
        piece = sequence[start:end]
        if not is_valid_dna(piece):
            return fragment_index
        fragments.append(
            {
                **entry_meta,
                "fragment_name": f"{entry_meta['seq_name']}_{start}_{end}",
                "fragment_index": fragment_index,
                "start": start,
                "end": end,
                "fragment_length": len(piece),
                "sequence": piece,
            }
        )
        return fragment_index + 1

    fragment_index = 0
    if seq_len >= fragment_len:
        last_pos = 0
        for start in range(0, seq_len - fragment_len + 1, fragment_len):
            end = start + fragment_len
            fragment_index = add_fragment(fragment_index, start, end)
            last_pos = end
        if seq_len - last_pos >= min_tail_len:
            add_fragment(fragment_index, last_pos, seq_len)
    else:
        add_fragment(0, 0, seq_len)
    return fragments


def load_entries(spec: FastaSpec, max_records: int = 0) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    positive = 0
    negative = 0
    for index, (meta, sequence) in enumerate(iter_fasta_records(spec.fasta_path)):
        if max_records > 0 and index >= max_records:
            break
        meta_row = base_meta(meta, spec, sequence)
        if int(meta_row["binary_label"]) == 1:
            positive += 1
        else:
            negative += 1
        entries.append({"meta": meta_row, "fragments": split_fragments(meta_row, sequence)})

    if max_records <= 0 and spec.expected_total:
        if len(entries) != spec.expected_total:
            raise ValueError(f"{spec.fasta_path} records={len(entries)} expected={spec.expected_total}")
        if positive != spec.expected_positive or negative != spec.expected_negative:
            raise ValueError(
                f"{spec.fasta_path} label counts pos={positive} neg={negative}, "
                f"expected pos={spec.expected_positive} neg={spec.expected_negative}"
            )
    return entries


def load_model(model_name: str, model_kind: str, model_dir: Path, device: torch.device, model_max_length: int):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    if model_kind == "meanpool":
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            num_labels=2,
            trust_remote_code=True,
        ).to(device)
    else:
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
        backbone = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True).to(device)
        model = ViraLM_MIL_Gated(backbone, hidden_size=int(getattr(config, "hidden_size", 768)), num_classes=2).to(device)
        state = torch.load(model_dir / "best_mil_model.pt", map_location="cpu")
        model.load_state_dict(state, strict=True)
    model.eval()
    return tokenizer, model


def sequence_output_row(
    model_name: str,
    model_kind: str,
    entry_meta: Dict[str, Any],
    score: Optional[float],
    fragment_count: int,
    threshold: float,
) -> Dict[str, Any]:
    missing = score is None
    prediction = "non-virus" if missing or float(score) <= threshold else "virus"
    return {
        **{field: entry_meta.get(field, "") for field in SEQUENCE_FIELDS},
        "model_name": model_name,
        "model_kind": model_kind,
        "prediction": prediction,
        "virus_score": "" if missing else float(score),
        "fragment_count": "" if missing else int(fragment_count),
        "missing_prediction": 1 if missing else 0,
    }


def fragment_output_row(
    model_name: str,
    model_kind: str,
    fragment: Dict[str, Any],
    score: float,
    threshold: float,
) -> Dict[str, Any]:
    return {
        **{field: fragment.get(field, "") for field in FRAGMENT_FIELDS},
        "model_name": model_name,
        "model_kind": model_kind,
        "prediction": "virus" if float(score) > threshold else "non-virus",
        "virus_score": float(score),
    }


def predict_meanpool(
    model_name: str,
    model_kind: str,
    tokenizer: Any,
    model: torch.nn.Module,
    entries: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fragments = [fragment for entry in entries for fragment in entry["fragments"]]
    fragment_rows: List[Dict[str, Any]] = []
    seq_scores: "OrderedDict[str, List[float]]" = OrderedDict()

    if fragments:
        loader = DataLoader(
            FragmentDataset(fragments),
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            collate_fn=FragmentCollator(tokenizer, args.model_max_length),
            num_workers=max(0, int(args.dataloader_workers)),
            pin_memory=device.type == "cuda",
        )
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"{model_name} {entries[0]['meta']['fasta_stem']} fragments"):
                meta_rows = batch.pop("meta")
                tensor_batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with autocast_context(use_amp):
                    out = model(**tensor_batch)
                probs = torch.sigmoid(_binary_logits(out.logits)).float().detach().cpu().numpy().tolist()
                for fragment, score in zip(meta_rows, probs):
                    seq_scores.setdefault(str(fragment["seq_name"]), []).append(float(score))
                    fragment_rows.append(fragment_output_row(model_name, model_kind, fragment, float(score), args.threshold))

    sequence_rows = []
    for entry in entries:
        seq_name = str(entry["meta"]["seq_name"])
        scores = seq_scores.get(seq_name, [])
        score = float(np.mean(scores)) if scores else None
        sequence_rows.append(sequence_output_row(model_name, model_kind, entry["meta"], score, len(scores), args.threshold))
    return sequence_rows, fragment_rows


def predict_mil(
    model_name: str,
    model_kind: str,
    tokenizer: Any,
    model: ViraLM_MIL_Gated,
    entries: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    sequence_rows_by_name: Dict[str, Dict[str, Any]] = {}
    fragment_rows: List[Dict[str, Any]] = []
    dataset = SequenceBagDataset(entries)
    if len(dataset):
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.mil_batch_size)),
            shuffle=False,
            collate_fn=SequenceBagCollator(tokenizer, args.model_max_length),
            num_workers=0,
        )
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"{model_name} {entries[0]['meta']['fasta_stem']} bags"):
                with autocast_context(use_amp):
                    seq_logits, _, frag_logits_list, _ = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        sub_chunk_size=max(1, int(args.scan_chunk)),
                        return_frag_logits=True,
                        return_hidden=False,
                    )
                seq_probs = torch.sigmoid(_binary_logits(seq_logits)).float().detach().cpu().numpy().tolist()
                for idx, entry in enumerate(batch["entries"]):
                    score = float(seq_probs[idx])
                    fragments = entry["fragments"]
                    sequence_rows_by_name[str(entry["meta"]["seq_name"])] = sequence_output_row(
                        model_name,
                        model_kind,
                        entry["meta"],
                        score,
                        len(fragments),
                        args.threshold,
                    )
                    frag_logits = frag_logits_list[idx] if frag_logits_list is not None else None
                    if frag_logits is None:
                        frag_probs = [float("nan")] * len(fragments)
                    else:
                        frag_probs = torch.sigmoid(_binary_logits(frag_logits.float())).cpu().numpy().tolist()
                    for fragment, frag_score in zip(fragments, frag_probs):
                        fragment_rows.append(
                            fragment_output_row(model_name, model_kind, fragment, float(frag_score), args.threshold)
                        )

    sequence_rows = []
    for entry in entries:
        seq_name = str(entry["meta"]["seq_name"])
        row = sequence_rows_by_name.get(seq_name)
        if row is None:
            row = sequence_output_row(model_name, model_kind, entry["meta"], None, 0, args.threshold)
        sequence_rows.append(row)
    return sequence_rows, fragment_rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def compute_counts(rows: Sequence[Dict[str, Any]], score_field: str = "virus_score") -> Dict[str, Any]:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    labels = []
    probs = []
    positive = 0
    negative = 0
    for row in rows:
        is_positive = row.get("class_label") == "positive"
        pred_positive = row.get("prediction") == "virus"
        if is_positive:
            positive += 1
        else:
            negative += 1
        if is_positive and pred_positive:
            counts["TP"] += 1
        elif not is_positive and pred_positive:
            counts["FP"] += 1
        elif is_positive and not pred_positive:
            counts["FN"] += 1
        else:
            counts["TN"] += 1
        score = row.get(score_field, "")
        if score != "":
            labels.append(1 if is_positive else 0)
            probs.append(float(score))

    precision = counts["TP"] / (counts["TP"] + counts["FP"]) if counts["TP"] + counts["FP"] else 0.0
    recall = counts["TP"] / (counts["TP"] + counts["FN"]) if counts["TP"] + counts["FN"] else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = counts["TP"] + counts["FP"] + counts["FN"] + counts["TN"]
    accuracy = (counts["TP"] + counts["TN"]) / total if total else 0.0
    auc = compute_auc_metrics(np.asarray(probs, dtype=np.float64), np.asarray(labels, dtype=np.int64)) if probs else {
        "auroc": float("nan"),
        "auprc": float("nan"),
    }
    return {
        **counts,
        "positive": positive,
        "negative": negative,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "auroc": auc["auroc"],
        "auprc": auc["auprc"],
    }


def sequence_metric_row(
    model_name: str,
    model_kind: str,
    spec: FastaSpec,
    rows: Sequence[Dict[str, Any]],
    result_csv: Path,
) -> Dict[str, Any]:
    counts = compute_counts(rows)
    missing_total = sum(1 for row in rows if str(row.get("missing_prediction", "0")) == "1")
    return {
        "model_name": model_name,
        "model_kind": model_kind,
        "benchmark": spec.benchmark,
        "fasta_stem": spec.fasta_stem,
        "fasta_file": spec.fasta_file,
        "level": "sequence",
        "total_positive": counts["positive"],
        "total_negative": counts["negative"],
        "evaluated_total": len(rows) - missing_total,
        "missing_total": missing_total,
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "TN": counts["TN"],
        "precision": counts["precision"],
        "recall": counts["recall"],
        "f1_score": counts["f1_score"],
        "accuracy": counts["accuracy"],
        "result_csv": str(result_csv),
        "auroc": counts["auroc"],
        "auprc": counts["auprc"],
    }


def fragment_metric_row(
    model_name: str,
    model_kind: str,
    spec: FastaSpec,
    rows: Sequence[Dict[str, Any]],
    fragment_csv: Path,
) -> Dict[str, Any]:
    counts = compute_counts(rows)
    return {
        "model_name": model_name,
        "model_kind": model_kind,
        "benchmark": spec.benchmark,
        "fasta_stem": spec.fasta_stem,
        "fasta_file": spec.fasta_file,
        "level": "fragment",
        "total_positive_fragments": counts["positive"],
        "total_negative_fragments": counts["negative"],
        "TP": counts["TP"],
        "FP": counts["FP"],
        "FN": counts["FN"],
        "TN": counts["TN"],
        "precision": counts["precision"],
        "recall": counts["recall"],
        "f1_score": counts["f1_score"],
        "accuracy": counts["accuracy"],
        "fragment_result_csv": str(fragment_csv),
        "auroc": counts["auroc"],
        "auprc": counts["auprc"],
    }


def prediction_paths(output_dir: Path, model_name: str, fasta_stem_value: str) -> Tuple[Path, Path]:
    seq_path = output_dir / "sequence_predictions_by_file" / model_name / f"{fasta_stem_value}.csv"
    frag_path = output_dir / "fragment_predictions_by_file" / model_name / f"{fasta_stem_value}.csv"
    return seq_path, frag_path


def concatenate_csvs(paths: Sequence[Path], output_path: Path, fieldnames: Sequence[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            with open(path, newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                for row in reader:
                    writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()
    if Path(args.data_root).name != "realm_rank_benchmark":
        raise SystemExit(f"refusing unexpected data root: {args.data_root}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(max(1, int(args.threads)))
    if torch.cuda.is_available() and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = benchmark_fastas(Path(args.data_root), args.benchmarks, args.lengths)
    model_root = Path(args.model_root)
    device = torch.device(args.device)
    use_amp = str2bool(args.fp16) and device.type == "cuda"

    sequence_metrics: List[Dict[str, Any]] = []
    fragment_metrics: List[Dict[str, Any]] = []
    sequence_prediction_paths: List[Path] = []
    fragment_prediction_paths: List[Path] = []
    model_summaries = []
    start_time = time.time()

    for model_name in args.models:
        model_kind, model_dir = model_spec(model_name, model_root)
        print(json.dumps({"event": "model_start", "model_name": model_name, "model_kind": model_kind}, sort_keys=True), flush=True)
        pending = []
        for spec in specs:
            seq_path, frag_path = prediction_paths(output_dir, model_name, spec.fasta_stem)
            if args.overwrite or not (seq_path.is_file() and frag_path.is_file()):
                pending.append(spec)
        tokenizer = None
        model = None
        if pending:
            tokenizer, model = load_model(model_name, model_kind, model_dir, device, args.model_max_length)
        for spec in specs:
            seq_path, frag_path = prediction_paths(output_dir, model_name, spec.fasta_stem)
            if not args.overwrite and seq_path.is_file() and frag_path.is_file():
                print(
                    json.dumps(
                        {"event": "skip_existing", "model_name": model_name, "fasta_stem": spec.fasta_stem},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                sequence_rows = read_csv_rows(seq_path)
                fragment_rows = read_csv_rows(frag_path)
            else:
                if tokenizer is None or model is None:
                    tokenizer, model = load_model(model_name, model_kind, model_dir, device, args.model_max_length)
                entries = load_entries(spec, max_records=max(0, int(args.max_records)))
                if model_kind == "mil":
                    sequence_rows, fragment_rows = predict_mil(
                        model_name, model_kind, tokenizer, model, entries, args, device, use_amp
                    )
                else:
                    sequence_rows, fragment_rows = predict_meanpool(
                        model_name, model_kind, tokenizer, model, entries, args, device, use_amp
                    )
                write_csv(seq_path, sequence_rows, SEQUENCE_FIELDS)
                write_csv(frag_path, fragment_rows, FRAGMENT_FIELDS)

            sequence_prediction_paths.append(seq_path)
            fragment_prediction_paths.append(frag_path)
            sequence_metrics.append(sequence_metric_row(model_name, model_kind, spec, sequence_rows, seq_path))
            fragment_metrics.append(fragment_metric_row(model_name, model_kind, spec, fragment_rows, frag_path))
            print(
                json.dumps(
                    {
                        "event": "file_done",
                        "model_name": model_name,
                        "model_kind": model_kind,
                        "benchmark": spec.benchmark,
                        "fasta_stem": spec.fasta_stem,
                        "sequence_rows": len(sequence_rows),
                        "fragment_rows": len(fragment_rows),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        model_summaries.append({"model_name": model_name, "model_kind": model_kind, "model_dir": str(model_dir)})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(output_dir / "sequence_metrics_with_auc.csv", sequence_metrics, SEQUENCE_METRIC_FIELDS)
    write_csv(output_dir / "fragment_metrics_with_auc.csv", fragment_metrics, FRAGMENT_METRIC_FIELDS)
    if not args.no_combined_predictions:
        concatenate_csvs(sequence_prediction_paths, output_dir / "sequence_predictions.csv", SEQUENCE_FIELDS)
        concatenate_csvs(fragment_prediction_paths, output_dir / "fragment_predictions.csv", FRAGMENT_FIELDS)
    write_json(
        output_dir / "summary.json",
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_root": str(Path(args.data_root).resolve()),
            "aggregation": "original FASTA record; meanpool averages internal 2000bp fragments, MIL uses gated sequence logit",
            "fragment_len": 2000,
            "min_tail_len": 500,
            "min_sequence_len": 500,
            "threshold_rule": "virus if score > threshold",
            "threshold": float(args.threshold),
            "max_records_per_fasta": int(args.max_records),
            "elapsed_sec": round(time.time() - start_time, 3),
            "models": model_summaries,
            "fastas": [
                {
                    "benchmark": spec.benchmark,
                    "length": spec.length,
                    "fasta_path": str(spec.fasta_path),
                    "fasta_file": spec.fasta_file,
                    "fasta_stem": spec.fasta_stem,
                    "expected_total": spec.expected_total,
                    "expected_positive": spec.expected_positive,
                    "expected_negative": spec.expected_negative,
                }
                for spec in specs
            ],
            "outputs": {
                "sequence_metrics_with_auc": str(output_dir / "sequence_metrics_with_auc.csv"),
                "fragment_metrics_with_auc": str(output_dir / "fragment_metrics_with_auc.csv"),
                "sequence_predictions": str(output_dir / "sequence_predictions.csv"),
                "fragment_predictions": str(output_dir / "fragment_predictions.csv"),
            },
        },
    )
    print(
        json.dumps(
            {
                "event": "finished",
                "models": len(args.models),
                "fastas": len(specs),
                "sequence_metric_rows": len(sequence_metrics),
                "fragment_metric_rows": len(fragment_metrics),
                "output_dir": str(output_dir),
                "elapsed_sec": round(time.time() - start_time, 3),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
