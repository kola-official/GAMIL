#!/usr/bin/env python3
"""Compute sequence and fragment metrics from final merged eval outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = BUNDLE_DIR.parent.parent

SEQUENCE_METRIC_FIELDS = [
    "model_name",
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
]
FRAGMENT_METRIC_FIELDS = [
    "model_name",
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
]
SEQUENCE_PREDICTION_FIELDS = [
    "model_name",
    "benchmark",
    "fasta_stem",
    "seq_name",
    "class_label",
    "prediction",
    "virus_score",
    "fragment_count",
    "missing_prediction",
]
FRAGMENT_PREDICTION_FIELDS = [
    "model_name",
    "benchmark",
    "fasta_stem",
    "seq_name",
    "fragment_name",
    "fragment_index",
    "start",
    "end",
    "length",
    "class_label",
    "prediction",
    "virus_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate merged ViraLM outputs")
    parser.add_argument("--work-dir", default=str(BUNDLE_DIR / "work"))
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", str(PROJECT_DIR / "processed_data" / "realm_rank_test_v2")),
    )
    parser.add_argument("--output-dir", default=str(BUNDLE_DIR / "metrics"))
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_csv_map(path: Path, key_field: str) -> Dict[str, Dict[str, str]]:
    rows = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row[key_field]] = row
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_labels(path: Path) -> Dict[str, str]:
    labels = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            labels[row["query_id"]] = row["class_label"]
    return labels


def fasta_ids(path: Path) -> List[str]:
    ids = []
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "accuracy": round(accuracy, 6),
    }


def update_confusion(counts: Dict[str, int], class_label: str, prediction: str) -> None:
    true_positive = class_label == "positive"
    pred_positive = prediction == "virus"
    if true_positive and pred_positive:
        counts["TP"] += 1
    elif not true_positive and pred_positive:
        counts["FP"] += 1
    elif true_positive and not pred_positive:
        counts["FN"] += 1
    else:
        counts["TN"] += 1


def write_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def evaluate_group(
    group: Dict[str, str],
    labels: Dict[str, str],
    allow_missing: bool,
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    final_result = Path(group["final_result"])
    final_fragment = Path(group["final_fragment_result"])
    if not final_result.exists() or not final_fragment.exists():
        if allow_missing:
            return None, None, [], []
        missing = final_result if not final_result.exists() else final_fragment
        raise SystemExit(f"missing final output for {group['group_id']}: {missing}")

    ids = fasta_ids(Path(group["fasta_path"]))
    predictions = read_csv_map(final_result, "seq_name")
    seq_prediction_rows: List[Dict[str, object]] = []
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    total_positive = 0
    total_negative = 0
    missing_total = 0

    for seq_name in ids:
        class_label = labels.get(seq_name, "")
        if not class_label:
            raise SystemExit(f"metadata label missing for {seq_name}")
        if class_label == "positive":
            total_positive += 1
        else:
            total_negative += 1
        pred_row = predictions.get(seq_name)
        if pred_row is None:
            prediction = "non-virus"
            score = ""
            fragment_count = ""
            missing = 1
            missing_total += 1
        else:
            prediction = pred_row.get("prediction", "non-virus")
            score = pred_row.get("virus_score", "")
            fragment_count = pred_row.get("fragment_count", "")
            missing = 0
        update_confusion(counts, class_label, prediction)
        seq_prediction_rows.append(
            {
                "model_name": group["model_name"],
                "benchmark": group["benchmark"],
                "fasta_stem": group["fasta_stem"],
                "seq_name": seq_name,
                "class_label": class_label,
                "prediction": prediction,
                "virus_score": score,
                "fragment_count": fragment_count,
                "missing_prediction": missing,
            }
        )

    metric_values = compute_metrics(counts["TP"], counts["FP"], counts["FN"], counts["TN"])
    seq_metric = {
        "model_name": group["model_name"],
        "benchmark": group["benchmark"],
        "fasta_stem": group["fasta_stem"],
        "fasta_file": group["fasta_file"],
        "level": "sequence",
        "total_positive": total_positive,
        "total_negative": total_negative,
        "evaluated_total": len(predictions),
        "missing_total": missing_total,
        **counts,
        **metric_values,
        "result_csv": str(final_result),
    }

    fragment_rows = read_csv_rows(final_fragment)
    fragment_prediction_rows: List[Dict[str, object]] = []
    fragment_counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    positive_fragments = 0
    negative_fragments = 0
    for row in fragment_rows:
        seq_name = row["seq_name"]
        class_label = labels.get(seq_name, "")
        if not class_label:
            raise SystemExit(f"metadata label missing for fragment parent {seq_name}")
        prediction = row.get("prediction", "non-virus")
        if class_label == "positive":
            positive_fragments += 1
        else:
            negative_fragments += 1
        update_confusion(fragment_counts, class_label, prediction)
        fragment_prediction_rows.append(
            {
                "model_name": group["model_name"],
                "benchmark": group["benchmark"],
                "fasta_stem": group["fasta_stem"],
                "seq_name": seq_name,
                "fragment_name": row.get("fragment_name", ""),
                "fragment_index": row.get("fragment_index", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
                "length": row.get("length", ""),
                "class_label": class_label,
                "prediction": prediction,
                "virus_score": row.get("virus_score", ""),
            }
        )

    fragment_metric_values = compute_metrics(
        fragment_counts["TP"],
        fragment_counts["FP"],
        fragment_counts["FN"],
        fragment_counts["TN"],
    )
    fragment_metric = {
        "model_name": group["model_name"],
        "benchmark": group["benchmark"],
        "fasta_stem": group["fasta_stem"],
        "fasta_file": group["fasta_file"],
        "level": "fragment",
        "total_positive_fragments": positive_fragments,
        "total_negative_fragments": negative_fragments,
        **fragment_counts,
        **fragment_metric_values,
        "fragment_result_csv": str(final_fragment),
    }
    return seq_metric, fragment_metric, seq_prediction_rows, fragment_prediction_rows


def write_summary(path: Path, sequence_metrics: List[Dict[str, object]], groups_total: int) -> None:
    lines = []
    lines.append("# eval_euk_pro_o_vs_r metrics summary")
    lines.append("")
    lines.append(f"- Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Completed groups: {len(sequence_metrics)} / {groups_total}")
    lines.append("")
    if sequence_metrics:
        lines.append("| model | benchmark | file | TP | FP | FN | TN | precision | recall | f1 | accuracy |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sequence_metrics:
            lines.append(
                f"| {row['model_name']} | {row['benchmark']} | {row['fasta_stem']} | "
                f"{row['TP']} | {row['FP']} | {row['FN']} | {row['TN']} | "
                f"{row['precision']} | {row['recall']} | {row['f1_score']} | {row['accuracy']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    groups = read_tsv(work_dir / "groups.tsv")
    if args.group_id:
        groups = [row for row in groups if row["group_id"] == args.group_id]
        if not groups:
            raise SystemExit(f"group not found: {args.group_id}")
    labels = load_labels(data_root / "metadata" / "benchmark_fragments.tsv")

    sequence_metrics: List[Dict[str, object]] = []
    fragment_metrics: List[Dict[str, object]] = []
    sequence_predictions: List[Dict[str, object]] = []
    fragment_predictions: List[Dict[str, object]] = []

    for group in groups:
        seq_metric, frag_metric, seq_rows, frag_rows = evaluate_group(group, labels, args.allow_missing)
        if seq_metric is None:
            continue
        sequence_metrics.append(seq_metric)
        fragment_metrics.append(frag_metric)
        sequence_predictions.extend(seq_rows)
        fragment_predictions.extend(frag_rows)

    write_rows(output_dir / "sequence_metrics.csv", sequence_metrics, SEQUENCE_METRIC_FIELDS)
    write_rows(output_dir / "fragment_metrics.csv", fragment_metrics, FRAGMENT_METRIC_FIELDS)
    write_rows(output_dir / "sequence_predictions.csv", sequence_predictions, SEQUENCE_PREDICTION_FIELDS)
    write_rows(output_dir / "fragment_predictions.csv", fragment_predictions, FRAGMENT_PREDICTION_FIELDS)
    write_summary(output_dir / "summary.md", sequence_metrics, len(groups))
    print(
        json.dumps(
            {
                "sequence_metric_rows": len(sequence_metrics),
                "fragment_metric_rows": len(fragment_metrics),
                "sequence_prediction_rows": len(sequence_predictions),
                "fragment_prediction_rows": len(fragment_predictions),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
