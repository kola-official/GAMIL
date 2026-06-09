#!/usr/bin/env python3
"""Prepare binary train/dev CSVs from Realm-Rank FASTA fragments."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, TextIO


FASTA_WIDTH = 80
CSV_FIELDS = [
    "sequence",
    "label",
    "source_id",
    "source",
    "genome",
    "contig",
    "start",
    "end",
    "length",
    "split",
]


def log(message: str) -> None:
    print(f"[prepare-realm-rank-csv] {message}", flush=True)


def resolve_workers(value: int) -> int:
    if value > 0:
        return value
    cpus = os.cpu_count() or 1
    return max(1, min(16, cpus))


def split_for_genome(genome: str, seed: int, dev_fraction: float) -> str:
    payload = f"{seed}\x1f{genome}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(2**64)
    return "dev" if value < dev_fraction else "train"


def normalize_sequence(seq: str) -> str:
    return seq.upper().replace("U", "T")


def parse_header(header: str) -> dict[str, str]:
    parts = header.split()
    attrs = {"record_id": parts[0] if parts else "unknown"}
    for token in parts[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        attrs[key] = value
    return attrs


def binary_label(attrs: dict[str, str]) -> int:
    source = attrs.get("source", "")
    supergroup = attrs.get("supergroup", "")
    return 1 if source == "virus" or supergroup == "virus" else 0


def source_id_from_attrs(attrs: dict[str, str]) -> str:
    return attrs.get("genome") or attrs.get("source_id") or attrs.get("contig") or attrs["record_id"]


def convert_record(header: str, sequence: str, seed: int, dev_fraction: float) -> dict[str, object]:
    attrs = parse_header(header)
    source = attrs.get("source", "")
    genome = source_id_from_attrs(attrs)
    label = binary_label(attrs)
    split = split_for_genome(genome, seed=seed, dev_fraction=dev_fraction)
    return {
        "split": split,
        "sequence": normalize_sequence(sequence),
        "label": label,
        "source_id": genome,
        "source": source or "unknown",
        "genome": genome,
        "contig": attrs.get("contig", ""),
        "start": attrs.get("start", ""),
        "end": attrs.get("end", ""),
        "length": len(sequence),
    }


def convert_fixed_record(header: str, sequence: str, split: str) -> dict[str, object]:
    attrs = parse_header(header)
    source = attrs.get("source", "")
    genome = source_id_from_attrs(attrs)
    header_split = attrs.get("split", split)
    if header_split != split:
        raise RuntimeError(f"Header split mismatch for {attrs['record_id']}: header={header_split} expected={split}")
    return {
        "split": split,
        "sequence": normalize_sequence(sequence),
        "label": binary_label(attrs),
        "source_id": genome,
        "source": source or "unknown",
        "genome": genome,
        "contig": attrs.get("contig", ""),
        "start": attrs.get("start", ""),
        "end": attrs.get("end", ""),
        "length": len(sequence),
    }


def convert_batch(batch: list[tuple[str, str]], seed: int, dev_fraction: float) -> list[dict[str, object]]:
    return [convert_record(header, sequence, seed=seed, dev_fraction=dev_fraction) for header, sequence in batch]


def iter_fasta_records_from_handle(handle: TextIO) -> Iterator[tuple[str, str]]:
    header: str | None = None
    seq_parts: list[str] = []
    for raw_line in handle:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_parts)
            header = line[1:]
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def iter_fasta_records(path: Path, decompress_workers: int, use_pigz: bool) -> Iterator[tuple[str, str]]:
    proc: subprocess.Popen[str] | None = None
    handle: TextIO | None = None
    try:
        if path.suffix == ".gz":
            pigz = shutil.which("pigz") if use_pigz else None
            if pigz:
                cmd = [pigz, "-dc", "-p", str(max(1, decompress_workers)), str(path)]
                log("decompressor: " + " ".join(cmd))
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    bufsize=1024 * 1024,
                )
                if proc.stdout is None:
                    raise RuntimeError("pigz stdout pipe was not created")
                handle = proc.stdout
            else:
                log(f"decompressor: python gzip ({path})")
                handle = gzip.open(path, "rt")
        else:
            log(f"decompressor: plain text ({path})")
            handle = open(path, "rt")

        yield from iter_fasta_records_from_handle(handle)
    finally:
        exc_type = sys.exc_info()[0]
        if handle is not None:
            handle.close()
        if proc is not None:
            _, stderr = proc.communicate()
            if proc.returncode not in (0, None) and exc_type is None:
                raise RuntimeError(f"pigz failed with exit code {proc.returncode}: {stderr.strip()}")


def iter_batches(records: Iterator[tuple[str, str]], batch_size: int) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_converted_batches(
    records: Iterator[tuple[str, str]],
    batch_size: int,
    workers: int,
    seed: int,
    dev_fraction: float,
) -> Iterator[list[dict[str, object]]]:
    batches = iter_batches(records, batch_size=batch_size)
    if workers <= 1:
        for batch in batches:
            yield convert_batch(batch, seed=seed, dev_fraction=dev_fraction)
        return

    max_pending = max(2, workers * 2)
    pending = deque()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in batches:
            pending.append(executor.submit(convert_batch, batch, seed, dev_fraction))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def quotas_satisfied(
    total_written: int,
    label_counts: Counter[int],
    label_split_counts: Counter[tuple[int, str]],
    max_records: int,
    max_records_per_class: int,
    max_records_per_class_per_split: int,
) -> bool:
    if max_records > 0 and total_written >= max_records:
        return True
    if max_records_per_class_per_split > 0:
        required = (
            label_split_counts[(0, "train")] >= max_records_per_class_per_split
            and label_split_counts[(0, "dev")] >= max_records_per_class_per_split
            and label_split_counts[(1, "train")] >= max_records_per_class_per_split
            and label_split_counts[(1, "dev")] >= max_records_per_class_per_split
        )
        return required
    if max_records_per_class > 0:
        return label_counts[0] >= max_records_per_class and label_counts[1] >= max_records_per_class
    return False


def should_write_row(
    row: dict[str, object],
    total_written: int,
    label_counts: Counter[int],
    label_split_counts: Counter[tuple[int, str]],
    max_records: int,
    max_records_per_class: int,
    max_records_per_class_per_split: int,
) -> bool:
    label = int(row["label"])
    split = str(row["split"])
    if max_records > 0 and total_written >= max_records:
        return False
    if max_records_per_class_per_split > 0:
        return label_split_counts[(label, split)] < max_records_per_class_per_split
    if max_records_per_class > 0:
        return label_counts[label] < max_records_per_class
    return True


def write_csvs(args: argparse.Namespace) -> dict[str, object]:
    input_fasta = Path(args.input_fasta)
    output_dir = Path(args.output_dir)
    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")
    if not (0.0 < args.dev_fraction < 1.0):
        raise ValueError("--dev-fraction must be in (0, 1)")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    workers = resolve_workers(args.num_workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train.csv"
    dev_csv = output_dir / "dev.csv"
    train_tmp = output_dir / f"train.csv.tmp.{os.getpid()}"
    dev_tmp = output_dir / f"dev.csv.tmp.{os.getpid()}"

    started = time.time()
    label_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    label_split_counts: Counter[tuple[int, str]] = Counter()
    source_counts: Counter[str] = Counter()
    length_by_split: dict[str, list[int]] = {"train": [], "dev": []}
    genomes_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    genome_label: dict[str, int] = {}
    genome_label_conflicts = 0
    total_seen = 0
    total_written = 0

    records = iter_fasta_records(input_fasta, decompress_workers=workers, use_pigz=not args.no_pigz)
    converted_batches = iter_converted_batches(
        records=records,
        batch_size=args.batch_size,
        workers=workers,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
    )

    with open(train_tmp, "w", newline="") as train_handle, open(dev_tmp, "w", newline="") as dev_handle:
        writers = {
            "train": csv.writer(train_handle, lineterminator="\n"),
            "dev": csv.writer(dev_handle, lineterminator="\n"),
        }
        for writer in writers.values():
            writer.writerow(CSV_FIELDS)

        for batch in converted_batches:
            for row in batch:
                total_seen += 1
                if not should_write_row(
                    row=row,
                    total_written=total_written,
                    label_counts=label_counts,
                    label_split_counts=label_split_counts,
                    max_records=args.max_records,
                    max_records_per_class=args.max_records_per_class,
                    max_records_per_class_per_split=args.max_records_per_class_per_split,
                ):
                    continue

                split = str(row["split"])
                label = int(row["label"])
                source_id = str(row["source_id"])
                writers[split].writerow([row.get(field, "") for field in CSV_FIELDS])

                total_written += 1
                label_counts[label] += 1
                split_counts[split] += 1
                label_split_counts[(label, split)] += 1
                source_counts[str(row["source"])] += 1
                length_by_split[split].append(int(row["length"]))
                genomes_by_split[split].add(source_id)
                if source_id in genome_label and genome_label[source_id] != label:
                    genome_label_conflicts += 1
                else:
                    genome_label[source_id] = label

            if args.log_every > 0 and total_written and total_written % args.log_every == 0:
                log(
                    "written="
                    f"{total_written} train={split_counts['train']} dev={split_counts['dev']} "
                    f"label0={label_counts[0]} label1={label_counts[1]}"
                )

            if quotas_satisfied(
                total_written=total_written,
                label_counts=label_counts,
                label_split_counts=label_split_counts,
                max_records=args.max_records,
                max_records_per_class=args.max_records_per_class,
                max_records_per_class_per_split=args.max_records_per_class_per_split,
            ):
                log("requested record limit reached; stopping FASTA scan early")
                break

    os.replace(train_tmp, train_csv)
    os.replace(dev_tmp, dev_csv)

    genome_intersection = genomes_by_split["train"].intersection(genomes_by_split["dev"])
    if genome_intersection:
        raise RuntimeError(f"Genome leakage detected between train/dev: {len(genome_intersection)} genomes")
    if split_counts["train"] == 0 or split_counts["dev"] == 0:
        raise RuntimeError(f"Both train and dev must be non-empty; got train={split_counts['train']} dev={split_counts['dev']}")
    if label_counts[0] == 0 or label_counts[1] == 0:
        raise RuntimeError(f"Both binary labels must be present; got label0={label_counts[0]} label1={label_counts[1]}")

    def length_summary(split: str) -> dict[str, float]:
        values = length_by_split[split]
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    summary = {
        "input_fasta": str(input_fasta),
        "output_dir": str(output_dir),
        "train_csv": str(train_csv),
        "dev_csv": str(dev_csv),
        "seed": args.seed,
        "dev_fraction": args.dev_fraction,
        "num_workers": workers,
        "batch_size": args.batch_size,
        "used_pigz": bool(shutil.which("pigz") and not args.no_pigz and input_fasta.suffix == ".gz"),
        "total_seen": total_seen,
        "total_written": total_written,
        "rows_by_split": dict(split_counts),
        "rows_by_label": {str(k): v for k, v in sorted(label_counts.items())},
        "rows_by_label_split": {f"{label}:{split}": count for (label, split), count in sorted(label_split_counts.items())},
        "rows_by_source": dict(source_counts),
        "genomes_by_split": {split: len(values) for split, values in genomes_by_split.items()},
        "genome_train_dev_intersection": len(genome_intersection),
        "genome_label_conflicts": genome_label_conflicts,
        "length_by_split": {split: length_summary(split) for split in ("train", "dev")},
        "elapsed_sec": time.time() - started,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    log(
        f"done rows={total_written} train={split_counts['train']} dev={split_counts['dev']} "
        f"label0={label_counts[0]} label1={label_counts[1]} elapsed={summary['elapsed_sec']:.1f}s"
    )
    return summary


def write_fixed_split_csvs(args: argparse.Namespace) -> dict[str, object]:
    if args.input_dir:
        input_dir = Path(args.input_dir)
        train_fasta = input_dir / "train.fasta.gz"
        dev_fasta = input_dir / "dev.fasta.gz"
    else:
        train_fasta = Path(args.train_fasta)
        dev_fasta = Path(args.dev_fasta)
    output_dir = Path(args.output_dir)
    for path in (train_fasta, dev_fasta):
        if not path.is_file():
            raise FileNotFoundError(f"Input FASTA not found: {path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    workers = resolve_workers(args.num_workers)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / "train.csv"
    dev_csv = output_dir / "dev.csv"
    train_tmp = output_dir / f"train.csv.tmp.{os.getpid()}"
    dev_tmp = output_dir / f"dev.csv.tmp.{os.getpid()}"

    started = time.time()
    label_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    label_split_counts: Counter[tuple[int, str]] = Counter()
    source_counts: Counter[str] = Counter()
    length_by_split: dict[str, list[int]] = {"train": [], "dev": []}
    genomes_by_split: dict[str, set[str]] = {"train": set(), "dev": set()}
    genome_label: dict[str, int] = {}
    genome_label_conflicts = 0
    total_seen = 0
    total_written = 0

    with open(train_tmp, "w", newline="") as train_handle, open(dev_tmp, "w", newline="") as dev_handle:
        writers = {
            "train": csv.writer(train_handle, lineterminator="\n"),
            "dev": csv.writer(dev_handle, lineterminator="\n"),
        }
        for writer in writers.values():
            writer.writerow(CSV_FIELDS)

        for split, fasta_path in (("train", train_fasta), ("dev", dev_fasta)):
            records = iter_fasta_records(fasta_path, decompress_workers=workers, use_pigz=not args.no_pigz)
            for header, sequence in records:
                total_seen += 1
                row = convert_fixed_record(header, sequence, split=split)
                if not should_write_row(
                    row=row,
                    total_written=total_written,
                    label_counts=label_counts,
                    label_split_counts=label_split_counts,
                    max_records=args.max_records,
                    max_records_per_class=args.max_records_per_class,
                    max_records_per_class_per_split=args.max_records_per_class_per_split,
                ):
                    continue

                label = int(row["label"])
                source_id = str(row["source_id"])
                writers[split].writerow([row.get(field, "") for field in CSV_FIELDS])

                total_written += 1
                label_counts[label] += 1
                split_counts[split] += 1
                label_split_counts[(label, split)] += 1
                source_counts[str(row["source"])] += 1
                length_by_split[split].append(int(row["length"]))
                genomes_by_split[split].add(source_id)
                if source_id in genome_label and genome_label[source_id] != label:
                    genome_label_conflicts += 1
                else:
                    genome_label[source_id] = label

                if quotas_satisfied(
                    total_written=total_written,
                    label_counts=label_counts,
                    label_split_counts=label_split_counts,
                    max_records=args.max_records,
                    max_records_per_class=args.max_records_per_class,
                    max_records_per_class_per_split=args.max_records_per_class_per_split,
                ):
                    log("requested record limit reached; stopping FASTA scan early")
                    break

    os.replace(train_tmp, train_csv)
    os.replace(dev_tmp, dev_csv)

    genome_intersection = genomes_by_split["train"].intersection(genomes_by_split["dev"])
    if genome_intersection:
        raise RuntimeError(f"Genome leakage detected between fixed train/dev: {len(genome_intersection)} genomes")
    if split_counts["train"] == 0 or split_counts["dev"] == 0:
        raise RuntimeError(f"Both train and dev must be non-empty; got train={split_counts['train']} dev={split_counts['dev']}")
    if label_counts[0] == 0 or label_counts[1] == 0:
        raise RuntimeError(f"Both binary labels must be present; got label0={label_counts[0]} label1={label_counts[1]}")

    def length_summary(split: str) -> dict[str, float]:
        values = length_by_split[split]
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    summary = {
        "mode": "fixed_split",
        "train_fasta": str(train_fasta),
        "dev_fasta": str(dev_fasta),
        "output_dir": str(output_dir),
        "train_csv": str(train_csv),
        "dev_csv": str(dev_csv),
        "num_workers": workers,
        "batch_size": args.batch_size,
        "used_pigz": bool(shutil.which("pigz") and not args.no_pigz),
        "total_seen": total_seen,
        "total_written": total_written,
        "rows_by_split": dict(split_counts),
        "rows_by_label": {str(k): v for k, v in sorted(label_counts.items())},
        "rows_by_label_split": {f"{label}:{split}": count for (label, split), count in sorted(label_split_counts.items())},
        "rows_by_source": dict(source_counts),
        "genomes_by_split": {split: len(values) for split, values in genomes_by_split.items()},
        "genome_train_dev_intersection": len(genome_intersection),
        "genome_label_conflicts": genome_label_conflicts,
        "length_by_split": {split: length_summary(split) for split in ("train", "dev")},
        "elapsed_sec": time.time() - started,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    log(
        f"done fixed split rows={total_written} train={split_counts['train']} dev={split_counts['dev']} "
        f"label0={label_counts[0]} label1={label_counts[1]} elapsed={summary['elapsed_sec']:.1f}s"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", help="Realm-Rank train FASTA or FASTA.gz for legacy hash train/dev splitting")
    parser.add_argument("--input-dir", help="Directory containing fixed train.fasta.gz and dev.fasta.gz")
    parser.add_argument("--train-fasta", help="Fixed train FASTA or FASTA.gz")
    parser.add_argument("--dev-fasta", help="Fixed dev FASTA or FASTA.gz")
    parser.add_argument("--output-dir", required=True, help="Directory for train.csv, dev.csv, and summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-records-per-class", type=int, default=0)
    parser.add_argument("--max-records-per-class-per-split", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument("--no-pigz", action="store_true", help="Disable pigz even if it is available")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fixed_requested = bool(args.input_dir or args.train_fasta or args.dev_fasta)
    if fixed_requested:
        if args.input_fasta:
            raise SystemExit("--input-fasta cannot be combined with fixed split inputs")
        if args.input_dir and (args.train_fasta or args.dev_fasta):
            raise SystemExit("--input-dir cannot be combined with --train-fasta/--dev-fasta")
        if not args.input_dir and not (args.train_fasta and args.dev_fasta):
            raise SystemExit("fixed split mode requires --input-dir or both --train-fasta and --dev-fasta")
        write_fixed_split_csvs(args)
    else:
        if not args.input_fasta:
            raise SystemExit("legacy hash-split mode requires --input-fasta")
        write_csvs(args)


if __name__ == "__main__":
    main()
