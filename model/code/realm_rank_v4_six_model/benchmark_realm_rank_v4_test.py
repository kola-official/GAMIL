#!/usr/bin/env python3
"""Benchmark six trained models plus two frozen teachers on Realm-Rank v4 test."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from experiment_config import (
    BENCHMARK_MODEL_ORDER,
    BENCHMARK_ROOT,
    MODEL_ROOT,
    TEACHER_MODELS,
    TEST_FASTA,
)
from mil_model import ViraLM_MIL_Gated
from mil_train_common import autocast_context
from shared import (
    _binary_logits,
    compute_binary_metrics_from_probs,
    iter_fasta_records,
    setup_logging,
    str2bool,
    write_csv,
    write_json,
)


PREDICTION_FIELDS = [
    "model_name",
    "model_kind",
    "record_id",
    "genome",
    "contig",
    "source",
    "label_group",
    "start",
    "end",
    "length",
    "binary_label",
    "fragment_probability",
    "genome_probability",
    "prediction",
]

MODEL_METRIC_FIELDS = [
    "model_name",
    "model_kind",
    "level",
    "total_fragments",
    "total_genomes",
    "total_positive",
    "total_negative",
    "TP",
    "FP",
    "FN",
    "TN",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "auroc",
    "auprc",
    "prediction_csv",
]

GROUP_METRIC_FIELDS = [
    "model_name",
    "model_kind",
    "level",
    "group_field",
    "group_value",
    "total_genomes",
    "total_positive",
    "total_negative",
    "TP",
    "FP",
    "FN",
    "TN",
    "precision",
    "recall",
    "f1_score",
    "accuracy",
    "auroc",
    "auprc",
]


class FragmentDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


class FragmentCollator:
    def __init__(self, tokenizer, model_max_length: int):
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


class GenomeDataset(Dataset):
    def __init__(self, grouped_records: Sequence[Tuple[str, List[Dict[str, Any]]]]):
        self.grouped_records = list(grouped_records)

    def __len__(self) -> int:
        return len(self.grouped_records)

    def __getitem__(self, index: int) -> Tuple[str, List[Dict[str, Any]]]:
        return self.grouped_records[index]


class GenomeCollator:
    def __init__(self, tokenizer, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

    def __call__(self, batch: Sequence[Tuple[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
        input_ids = []
        attention_mask = []
        metas = []
        genomes = []
        for genome, records in batch:
            tokenized = self.tokenizer(
                [row["sequence"] for row in records],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.model_max_length,
            )
            input_ids.append(tokenized["input_ids"])
            attention_mask.append(tokenized["attention_mask"])
            metas.append(records)
            genomes.append(genome)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "metas": metas, "genomes": genomes}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Realm-Rank v4 six-model experiment")
    parser.add_argument("--test-fasta", default=str(TEST_FASTA))
    parser.add_argument("--output-dir", default=str(BENCHMARK_ROOT))
    parser.add_argument("--model-root", default=str(MODEL_ROOT))
    parser.add_argument("--models", nargs="+", default=BENCHMARK_MODEL_ORDER)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mil-batch-size", type=int, default=8)
    parser.add_argument("--scan-chunk", type=int, default=int(os.environ.get("SCAN_CHUNK", "48")))
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fp16", default="True")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-records", type=int, default=0, help="Debug/smoke limit before grouping")
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--expected-fragments", type=int, default=40930)
    parser.add_argument("--expected-viral-fragments", type=int, default=20492)
    parser.add_argument("--expected-nonviral-fragments", type=int, default=20438)
    parser.add_argument("--expected-genomes", type=int, default=0, help="0 disables genome-count assertion")
    return parser.parse_args()


def model_spec(model_name: str, model_root: Path) -> Tuple[str, Path]:
    if model_name == "viralm_o":
        return "meanpool", Path(TEACHER_MODELS["viralm_o"])
    if model_name == "viralm_r_v4_final":
        return "meanpool", Path(TEACHER_MODELS["viralm_r_v4_final"])
    path = model_root / model_name
    if (path / "best_mil_model.pt").is_file():
        return "mil", path
    return "meanpool", path


def load_records(path: str, max_records: int = 0) -> List[Dict[str, Any]]:
    records = []
    for meta, sequence in iter_fasta_records(path):
        row = dict(meta)
        row["sequence"] = sequence
        row["binary_label"] = int(meta["binary_label"])
        row["label_group"] = meta.get("label", "") or meta.get("source", "")
        records.append(row)
        if max_records > 0 and len(records) >= max_records:
            break
    return records


def grouped_by_genome(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for row in records:
        grouped.setdefault(str(row["genome"]), []).append(row)
    return list(grouped.items())


def validate_counts(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, int]:
    fragments = len(records)
    viral = sum(1 for row in records if int(row["binary_label"]) == 1)
    nonviral = fragments - viral
    genomes = len({str(row["genome"]) for row in records})
    counts = {"fragments": fragments, "viral_fragments": viral, "nonviral_fragments": nonviral, "genomes": genomes}
    if args.max_records <= 0:
        expected = {
            "fragments": args.expected_fragments,
            "viral_fragments": args.expected_viral_fragments,
            "nonviral_fragments": args.expected_nonviral_fragments,
        }
        for key, value in expected.items():
            if value and counts[key] != value:
                raise ValueError(f"Unexpected {key}: observed={counts[key]} expected={value}")
        if args.expected_genomes and counts["genomes"] != args.expected_genomes:
            raise ValueError(f"Unexpected genomes: observed={counts['genomes']} expected={args.expected_genomes}")
    return counts


def predict_meanpool(
    model_name: str,
    model_dir: Path,
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> List[Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=2,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    loader = DataLoader(
        FragmentDataset(records),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=FragmentCollator(tokenizer, args.model_max_length),
        num_workers=args.dataloader_workers,
    )
    fragment_rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{model_name} fragments"):
            meta = batch.pop("meta")
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(use_amp):
                out = model(**batch)
            probs = torch.sigmoid(_binary_logits(out.logits)).float().cpu().numpy().tolist()
            for row, prob in zip(meta, probs):
                fragment_rows.append({**row, "fragment_probability": float(prob)})

    genome_probs: Dict[str, float] = {}
    by_genome: Dict[str, List[float]] = defaultdict(list)
    for row in fragment_rows:
        by_genome[str(row["genome"])].append(float(row["fragment_probability"]))
    for genome, probs in by_genome.items():
        genome_probs[genome] = float(np.mean(probs))

    output_rows = []
    for row in fragment_rows:
        genome_prob = genome_probs[str(row["genome"])]
        output_rows.append(prediction_row(model_name, "meanpool", row, row["fragment_probability"], genome_prob, args.threshold))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_rows


def predict_mil(
    model_name: str,
    model_dir: Path,
    records: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> List[Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    backbone = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True).to(device)
    model = ViraLM_MIL_Gated(backbone, hidden_size=int(getattr(config, "hidden_size", 768)), num_classes=2).to(device)
    state = torch.load(model_dir / "best_mil_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    loader = DataLoader(
        GenomeDataset(grouped_by_genome(records)),
        batch_size=args.mil_batch_size,
        shuffle=False,
        collate_fn=GenomeCollator(tokenizer, args.model_max_length),
        num_workers=0,
    )

    output_rows: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"{model_name} genomes"):
            with autocast_context(use_amp):
                seq_logits, _, frag_logits_list, _ = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    sub_chunk_size=args.scan_chunk,
                    return_frag_logits=True,
                    return_hidden=False,
                )
            genome_probs = torch.sigmoid(_binary_logits(seq_logits)).float().cpu().numpy().tolist()
            for group_idx, records_for_genome in enumerate(batch["metas"]):
                frag_logits = frag_logits_list[group_idx] if frag_logits_list is not None else None
                if frag_logits is None:
                    frag_probs = [float("nan")] * len(records_for_genome)
                else:
                    frag_probs = torch.sigmoid(_binary_logits(frag_logits).float()).cpu().numpy().tolist()
                genome_prob = float(genome_probs[group_idx])
                for row, frag_prob in zip(records_for_genome, frag_probs):
                    output_rows.append(prediction_row(model_name, "mil", row, float(frag_prob), genome_prob, args.threshold))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_rows


def prediction_row(
    model_name: str,
    model_kind: str,
    row: Dict[str, Any],
    fragment_probability: float,
    genome_probability: float,
    threshold: float,
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "model_kind": model_kind,
        "record_id": row["record_id"],
        "genome": row["genome"],
        "contig": row["contig"],
        "source": row.get("source", ""),
        "label_group": row.get("label_group", row.get("label", "")),
        "start": row.get("start", ""),
        "end": row.get("end", ""),
        "length": row.get("length", ""),
        "binary_label": int(row["binary_label"]),
        "fragment_probability": fragment_probability,
        "genome_probability": genome_probability,
        "prediction": "virus" if genome_probability >= threshold else "non-virus",
    }


def genome_level_rows(prediction_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    genomes: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for row in prediction_rows:
        genome = str(row["genome"])
        if genome not in genomes:
            genomes[genome] = dict(row)
        else:
            existing = genomes[genome]
            if int(existing["binary_label"]) != int(row["binary_label"]):
                raise ValueError(f"Conflicting labels for genome {genome}")
    return list(genomes.values())


def metrics_row(
    model_name: str,
    model_kind: str,
    rows: Sequence[Dict[str, Any]],
    prediction_csv: str = "",
    group_field: str = "",
    group_value: str = "",
) -> Dict[str, Any]:
    labels = np.asarray([int(row["binary_label"]) for row in rows], dtype=np.int64)
    probs = np.asarray([float(row["genome_probability"]) for row in rows], dtype=np.float64)
    m = compute_binary_metrics_from_probs(probs, labels, threshold=0.5)
    out = {
        "model_name": model_name,
        "model_kind": model_kind,
        "level": "genome",
        "total_genomes": len(rows),
        "total_positive": int(labels.sum()),
        "total_negative": int((labels == 0).sum()),
        "TP": m["TP"],
        "FP": m["FP"],
        "FN": m["FN"],
        "TN": m["TN"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1_score": m["f1_score"],
        "accuracy": m["accuracy"],
        "auroc": m["auroc"],
        "auprc": m["auprc"],
    }
    if prediction_csv:
        out["prediction_csv"] = prediction_csv
        out["total_fragments"] = ""
    if group_field:
        out["group_field"] = group_field
        out["group_value"] = group_value
    return out


def grouped_metric_rows(
    model_name: str,
    model_kind: str,
    genome_rows: Sequence[Dict[str, Any]],
    group_field: str,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in genome_rows:
        grouped[str(row.get(group_field, ""))].append(row)
    rows = []
    for value in sorted(grouped):
        rows.append(metrics_row(model_name, model_kind, grouped[value], group_field=group_field, group_value=value))
    return rows


def main() -> None:
    setup_logging()
    args = parse_args()
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.test_fasta, args.max_records)
    counts = validate_counts(records, args)
    device = torch.device(args.device)
    use_amp = str2bool(args.fp16) and device.type == "cuda"

    by_model_rows = []
    by_source_rows = []
    by_label_rows = []
    model_summaries = []
    model_root = Path(args.model_root)

    for model_name in args.models:
        model_kind, model_dir = model_spec(model_name, model_root)
        if model_kind == "mil":
            required = model_dir / "best_mil_model.pt"
        else:
            required = model_dir / "pytorch_model.bin"
        if not required.is_file():
            if args.allow_missing_models:
                print(f"Skipping missing model {model_name}: {required}")
                continue
            raise FileNotFoundError(str(required))

        if model_kind == "mil":
            prediction_rows = predict_mil(model_name, model_dir, records, args, device, use_amp)
        else:
            prediction_rows = predict_meanpool(model_name, model_dir, records, args, device, use_amp)

        pred_path = pred_dir / f"{model_name}.csv"
        write_csv(pred_path, prediction_rows, PREDICTION_FIELDS)
        genome_rows = genome_level_rows(prediction_rows)
        model_metric = metrics_row(
            model_name,
            model_kind,
            genome_rows,
            prediction_csv=str(pred_path),
        )
        model_metric["total_fragments"] = len(prediction_rows)
        by_model_rows.append(model_metric)
        by_source_rows.extend(grouped_metric_rows(model_name, model_kind, genome_rows, "source"))
        by_label_rows.extend(grouped_metric_rows(model_name, model_kind, genome_rows, "label_group"))
        model_summaries.append({"model_name": model_name, "model_kind": model_kind, "model_dir": str(model_dir)})

    write_csv(output_dir / "test_metrics_by_model.csv", by_model_rows, MODEL_METRIC_FIELDS)
    write_csv(output_dir / "test_metrics_by_source.csv", by_source_rows, GROUP_METRIC_FIELDS)
    write_csv(output_dir / "test_metrics_by_label_group.csv", by_label_rows, GROUP_METRIC_FIELDS)
    write_json(
        output_dir / "summary.json",
        {
            "test_fasta": str(args.test_fasta),
            "counts": counts,
            "aggregation": "source_id=genome",
            "threshold": args.threshold,
            "model_count": len(by_model_rows),
            "models": model_summaries,
            "outputs": {
                "test_predictions": str(pred_dir),
                "test_metrics_by_model": str(output_dir / "test_metrics_by_model.csv"),
                "test_metrics_by_source": str(output_dir / "test_metrics_by_source.csv"),
                "test_metrics_by_label_group": str(output_dir / "test_metrics_by_label_group.csv"),
            },
        },
    )
    print(json.dumps({"counts": counts, "model_count": len(by_model_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
