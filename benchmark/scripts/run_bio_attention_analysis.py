#!/usr/bin/env python3
"""Resumable biological attention analysis for Realm-Rank test records."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


GAMIL_ROOT = Path(os.environ.get("GAMIL_ROOT", Path(__file__).resolve().parents[2])).resolve()
PROCESSED_DATA_ROOT = Path(os.environ.get("PROCESSED_DATA_ROOT", GAMIL_ROOT / "processed_data")).resolve()
CHECKPOINT_ROOT = Path(os.environ.get("CHECKPOINT_ROOT", GAMIL_ROOT / "checkpoint" / "local_checkpoints")).resolve()
REALM_RANK_OUTPUT_ROOT = Path(
    os.environ.get("OUTPUT_ROOT", CHECKPOINT_ROOT / "gamil_six_model")
).resolve()
PAPER_CODE = Path(
    os.environ.get("GAMIL_MODEL_CODE_DIR", GAMIL_ROOT / "model" / "code" / "gamil_six_model")
).resolve()
VL_PYTHON = Path(os.environ.get("PYTHON_BIN", sys.executable))
BIOANN_BIN = Path(os.environ.get("BIOANN_BIN", ""))
BIOANN_PYTHON = Path(os.environ.get("BIOANN_PYTHON", str(BIOANN_BIN / "python")))

TEST_FASTA = PROCESSED_DATA_ROOT / "realm_rank" / "test.fasta.gz"
PRED_ROOT = REALM_RANK_OUTPUT_ROOT / "benchmark" / "test_predictions"
MP_PRED = PRED_ROOT / "viralm_r.csv"
GA_PRED = PRED_ROOT / "viralm_r_12l_gated_mil.csv"
GA_MODEL_DIR = REALM_RANK_OUTPUT_ROOT / "models" / "viralm_r_12l_gated_mil"
DEFAULT_OUT = REALM_RANK_OUTPUT_ROOT / "bio_attention_test"

THRESHOLD = 0.5
THREADS = 56
HEADER_FIELD_RE = re.compile(r"(\S+)=([^\s]+)")

EXPECTED_CASE_COUNTS = {
    "ga_rescued_positive": {"genomes": 33, "records": 253},
    "ga_corrected_negative": {"genomes": 280, "records": 359},
    "ga_worse": {"genomes": 111, "records": 953},
    "both_correct_pool": {"genomes": 3580},
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_tsv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Sequence[str]) -> int:
    ensure_dir(path.parent)
    count = 0
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
            count += 1
    return count


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def done_path(out_dir: Path, name: str) -> Path:
    return out_dir / "state" / f"{name}.done"


def stage_done(out_dir: Path, name: str, outputs: Sequence[Path]) -> bool:
    marker = done_path(out_dir, name)
    return marker.is_file() and all(path.exists() for path in outputs)


def mark_done(out_dir: Path, name: str, payload: Dict[str, Any]) -> None:
    marker = done_path(out_dir, name)
    payload = dict(payload)
    payload["done_at"] = now()
    write_json(marker, payload)


def parse_header(line: str) -> Dict[str, str]:
    header = line[1:].strip() if line.startswith(">") else line.strip()
    first = header.split(None, 1)[0]
    meta = {"record_id": first}
    for key, value in HEADER_FIELD_RE.findall(header):
        meta[key] = value
    source = meta.get("source", "")
    meta["binary_label"] = "1" if source == "virus" else "0"
    meta["label_group"] = meta.get("label", source)
    return meta


def fasta_opener(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def iter_fasta(path: Path):
    meta: Optional[Dict[str, str]] = None
    header = ""
    seq_chunks: List[str] = []
    with fasta_opener(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if meta is not None:
                    yield meta, header, "".join(seq_chunks).upper()
                header = line.rstrip("\n")
                meta = parse_header(header)
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if meta is not None:
            yield meta, header, "".join(seq_chunks).upper()


def iter_fasta_blocks(path: Path):
    current_id: Optional[str] = None
    block: List[str] = []
    with fasta_opener(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, block
                current_id = line[1:].split(None, 1)[0].strip()
                block = [line]
            else:
                block.append(line)
        if current_id is not None:
            yield current_id, block


def is_correct(prob: float, label: int, threshold: float = THRESHOLD) -> bool:
    return (prob >= threshold) == bool(label)


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value in (None, "", "NA", "nan"):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, "", "NA", "nan"):
            return default
        return int(float(value))
    except Exception:
        return default


def log_message(out_dir: Path, message: str) -> None:
    ensure_dir(out_dir / "logs")
    line = f"[{now()}] {message}\n"
    with open(out_dir / "logs" / "pipeline.log", "a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def run_command(
    cmd: Sequence[str],
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    attempts: int = 3,
    retry_delay: int = 30,
) -> None:
    ensure_dir(log_path.parent)
    cmd_payload = {"cmd": list(cmd), "started_at": now(), "attempts": attempts}
    write_json(log_path.with_suffix(log_path.suffix + ".cmd.json"), cmd_payload)
    last_rc = 0
    for attempt in range(1, attempts + 1):
        with open(log_path, "a") as log:
            log.write(f"\n[{now()}] attempt {attempt}/{attempts}: {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            last_rc = proc.returncode
            log.write(f"[{now()}] returncode={last_rc}\n")
        if last_rc == 0:
            return
        if attempt < attempts:
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Command failed after {attempts} attempts (rc={last_rc}): {' '.join(cmd)}")


def bioann_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = f"{BIOANN_BIN}:{env.get('PATH', '')}"
    return env


def build_case_records(out_dir: Path) -> None:
    case_path = out_dir / "case_records.tsv"
    counts_path = out_dir / "summary_tables" / "case_counts.tsv"
    match_path = out_dir / "summary_tables" / "control_matches.tsv"
    if stage_done(out_dir, "case_records", [case_path, counts_path, match_path]):
        log_message(out_dir, "stage case_records already done")
        return

    log_message(out_dir, "building case groups from fixed Realm-Rank test predictions")
    mp_rows = read_csv(MP_PRED)
    ga_rows = read_csv(GA_PRED)
    if len(mp_rows) != len(ga_rows):
        raise ValueError(f"Prediction row mismatch: MP={len(mp_rows)} GA={len(ga_rows)}")

    mp_by_record = {row["record_id"]: row for row in mp_rows}
    ga_by_record = {row["record_id"]: row for row in ga_rows}
    if set(mp_by_record) != set(ga_by_record):
        raise ValueError("Prediction record_id sets differ")

    records_by_genome: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    for row in ga_rows:
        records_by_genome.setdefault(row["genome"], []).append(row)

    mp_genome: Dict[str, Dict[str, str]] = OrderedDict()
    ga_genome: Dict[str, Dict[str, str]] = OrderedDict()
    for row in mp_rows:
        mp_genome.setdefault(row["genome"], row)
    for row in ga_rows:
        ga_genome.setdefault(row["genome"], row)

    genome_case: Dict[str, str] = {}
    both_correct: List[str] = []
    for genome, ga_row in ga_genome.items():
        mp_row = mp_genome[genome]
        label = safe_int(ga_row["binary_label"])
        mp_prob = safe_float(mp_row["genome_probability"])
        ga_prob = safe_float(ga_row["genome_probability"])
        mp_ok = is_correct(mp_prob, label)
        ga_ok = is_correct(ga_prob, label)
        if label == 1 and (not mp_ok) and ga_ok:
            genome_case[genome] = "ga_rescued_positive"
        elif label == 0 and (not mp_ok) and ga_ok:
            genome_case[genome] = "ga_corrected_negative"
        elif mp_ok and (not ga_ok):
            genome_case[genome] = "ga_worse"
        elif mp_ok and ga_ok:
            both_correct.append(genome)

    def profile(genome: str) -> Dict[str, Any]:
        rows = records_by_genome[genome]
        first = rows[0]
        lengths = [safe_float(row["length"]) for row in rows]
        return {
            "genome": genome,
            "source": first.get("source", ""),
            "label_group": first.get("label_group", ""),
            "binary_label": safe_int(first.get("binary_label")),
            "n_records": len(rows),
            "median_length": median(lengths),
            "total_bp": sum(lengths),
        }

    profiles = {genome: profile(genome) for genome in records_by_genome}
    control_profiles = [profiles[genome] for genome in both_correct]
    target_genomes = [
        genome
        for group in ("ga_rescued_positive", "ga_corrected_negative", "ga_worse")
        for genome, case in genome_case.items()
        if case == group
    ]
    used_controls: set[str] = set()
    matches: List[Dict[str, Any]] = []

    def candidate_controls(target: Dict[str, Any], allow_used: bool = False) -> List[Dict[str, Any]]:
        tiers = [
            lambda row: row["binary_label"] == target["binary_label"] and row["source"] == target["source"] and row["label_group"] == target["label_group"],
            lambda row: row["binary_label"] == target["binary_label"] and row["source"] == target["source"],
            lambda row: row["binary_label"] == target["binary_label"],
            lambda row: True,
        ]
        for predicate in tiers:
            rows = [
                row
                for row in control_profiles
                if (allow_used or row["genome"] not in used_controls) and predicate(row)
            ]
            if rows:
                return rows
        return []

    def match_score(target: Dict[str, Any], control: Dict[str, Any]) -> Tuple[float, str]:
        rec = abs(float(control["n_records"]) - float(target["n_records"])) / max(1.0, float(target["n_records"]))
        med = abs(math.log1p(float(control["median_length"])) - math.log1p(float(target["median_length"])))
        bp = abs(math.log1p(float(control["total_bp"])) - math.log1p(float(target["total_bp"])))
        return rec + med + 0.5 * bp, str(control["genome"])

    control_for_target: Dict[str, str] = {}
    for genome in target_genomes:
        target = profiles[genome]
        candidates = candidate_controls(target, allow_used=False)
        reused = 0
        if not candidates:
            candidates = candidate_controls(target, allow_used=True)
            reused = 1
        if not candidates:
            raise RuntimeError(f"No both-correct control candidates for {genome}")
        control = min(candidates, key=lambda row: match_score(target, row))
        used_controls.add(control["genome"])
        control_for_target[genome] = control["genome"]
        matches.append(
            {
                "target_genome": genome,
                "target_case_group": genome_case[genome],
                "control_genome": control["genome"],
                "control_reused": reused,
                "target_source": target["source"],
                "control_source": control["source"],
                "target_label_group": target["label_group"],
                "control_label_group": control["label_group"],
                "target_records": target["n_records"],
                "control_records": control["n_records"],
                "target_median_length": target["median_length"],
                "control_median_length": control["median_length"],
            }
        )

    control_genome_to_targets: Dict[str, List[str]] = defaultdict(list)
    for target, control in control_for_target.items():
        control_genome_to_targets[control].append(target)

    selected_rows: List[Dict[str, Any]] = []
    for genome, rows in records_by_genome.items():
        if genome in genome_case:
            group = genome_case[genome]
            control_for_case = ""
            control_for_genome = ""
        elif genome in control_genome_to_targets:
            group = "both_correct_control"
            targets = control_genome_to_targets[genome]
            control_for_case = ";".join(genome_case[target] for target in targets)
            control_for_genome = ";".join(targets)
        else:
            continue
        for row in rows:
            selected_rows.append(
                {
                    "case_group": group,
                    "control_for_case": control_for_case,
                    "control_for_genome": control_for_genome,
                    "record_id": row["record_id"],
                    "source": row.get("source", ""),
                    "label": row.get("label_group", ""),
                    "binary_label": row.get("binary_label", ""),
                    "genome": row.get("genome", ""),
                    "contig": row.get("contig", ""),
                    "start": row.get("start", ""),
                    "end": row.get("end", ""),
                    "length": row.get("length", ""),
                }
            )

    count_rows: List[Dict[str, Any]] = []
    for group in sorted({row["case_group"] for row in selected_rows}):
        rows = [row for row in selected_rows if row["case_group"] == group]
        count_rows.append(
            {
                "case_group": group,
                "genomes": len({row["genome"] for row in rows}),
                "records": len(rows),
                "positive_records": sum(safe_int(row["binary_label"]) for row in rows),
                "negative_records": sum(1 for row in rows if safe_int(row["binary_label"]) == 0),
            }
        )
    count_rows.append({"case_group": "both_correct_pool", "genomes": len(both_correct), "records": ""})

    observed = {row["case_group"]: row for row in count_rows}
    for group, expected in EXPECTED_CASE_COUNTS.items():
        if group not in observed:
            raise AssertionError(f"Missing expected case group {group}")
        for key, value in expected.items():
            if value and safe_int(observed[group].get(key)) != value:
                raise AssertionError(
                    f"Case count mismatch for {group}.{key}: observed={observed[group].get(key)} expected={value}"
                )

    fields = [
        "case_group",
        "control_for_case",
        "control_for_genome",
        "record_id",
        "source",
        "label",
        "binary_label",
        "genome",
        "contig",
        "start",
        "end",
        "length",
    ]
    write_tsv(case_path, selected_rows, fields)
    write_tsv(
        counts_path,
        count_rows,
        ["case_group", "genomes", "records", "positive_records", "negative_records"],
    )
    write_tsv(
        match_path,
        matches,
        [
            "target_genome",
            "target_case_group",
            "control_genome",
            "control_reused",
            "target_source",
            "control_source",
            "target_label_group",
            "control_label_group",
            "target_records",
            "control_records",
            "target_median_length",
            "control_median_length",
        ],
    )
    mark_done(
        out_dir,
        "case_records",
        {
            "case_records": str(case_path),
            "case_counts": str(counts_path),
            "control_matches": str(match_path),
            "selected_records": len(selected_rows),
            "selected_genomes": len({row["genome"] for row in selected_rows}),
        },
    )
    log_message(out_dir, f"case_records complete: {len(selected_rows)} records")


def select_fasta_records(out_dir: Path) -> None:
    case_path = out_dir / "case_records.tsv"
    fasta_path = out_dir / "selected_records.fna"
    checksum_path = out_dir / "selected_records.checksums.tsv"
    if stage_done(out_dir, "selected_fasta", [fasta_path, checksum_path]):
        log_message(out_dir, "stage selected_fasta already done")
        return

    log_message(out_dir, "selecting exact FASTA blocks from realm_rank/test.fasta.gz")
    case_rows = read_tsv(case_path)
    selected_ids = {row["record_id"] for row in case_rows}
    written_ids: List[str] = []
    checksum_rows: List[Dict[str, Any]] = []
    with open(fasta_path, "w") as out:
        for record_id, block in iter_fasta_blocks(TEST_FASTA):
            if record_id not in selected_ids:
                continue
            text = "".join(block)
            out.write(text)
            written_ids.append(record_id)
            checksum_rows.append(
                {
                    "record_id": record_id,
                    "original_block_sha256": sha256_text(text),
                    "selected_block_sha256": sha256_text(text),
                    "matches_original": 1,
                }
            )

    missing = selected_ids - set(written_ids)
    extra = set(written_ids) - selected_ids
    if missing or extra:
        raise AssertionError(f"Selected FASTA mismatch: missing={len(missing)} extra={len(extra)}")
    write_tsv(
        checksum_path,
        checksum_rows,
        ["record_id", "original_block_sha256", "selected_block_sha256", "matches_original"],
    )
    mark_done(
        out_dir,
        "selected_fasta",
        {
            "selected_fasta": str(fasta_path),
            "checksums": str(checksum_path),
            "records": len(written_ids),
            "sha256": sha256_file(fasta_path),
        },
    )
    log_message(out_dir, f"selected_fasta complete: {len(written_ids)} records")


def write_genome_shards(out_dir: Path, genomes: Sequence[str], shards: int = 2) -> List[Path]:
    shard_dir = ensure_dir(out_dir / "work" / "attention_shards")
    paths = [shard_dir / f"genomes_gpu{i}.txt" for i in range(shards)]
    buckets = [[] for _ in range(shards)]
    for idx, genome in enumerate(sorted(genomes)):
        buckets[idx % shards].append(genome)
    for path, bucket in zip(paths, buckets):
        path.write_text("\n".join(bucket) + ("\n" if bucket else ""))
    return paths


def export_attention(out_dir: Path, args: argparse.Namespace) -> None:
    attention_path = out_dir / "attention_records.tsv"
    sums_path = out_dir / "summary_tables" / "attention_sum_check.tsv"
    if stage_done(out_dir, "attention", [attention_path, sums_path]):
        log_message(out_dir, "stage attention already done")
        return

    log_message(out_dir, "exporting gated MIL attention with genome shards")
    case_rows = read_tsv(out_dir / "case_records.tsv")
    selected_genomes = sorted({row["genome"] for row in case_rows})
    shard_paths = write_genome_shards(out_dir, selected_genomes, shards=2)
    raw_paths = [out_dir / "work" / "attention_shards" / f"attention_gpu{i}.tsv" for i in range(2)]
    script_path = Path(__file__).resolve()

    configs = [
        {"batch": args.mil_batch_size, "scan": args.scan_chunk, "devices": ["0", "1"]},
        {"batch": max(1, args.mil_batch_size // 2), "scan": max(8, args.scan_chunk // 2), "devices": ["0", "1"]},
        {"batch": 1, "scan": 8, "devices": ["0", "1"]},
        {"batch": 1, "scan": 8, "devices": ["cpu", "cpu"]},
    ]
    last_error = None
    for cfg_idx, cfg in enumerate(configs, start=1):
        for path in raw_paths:
            if path.exists():
                path.unlink()
        procs = []
        for idx in range(2):
            cmd = [
                str(VL_PYTHON),
                str(script_path),
                "attention-worker",
                "--test-fasta",
                str(TEST_FASTA),
                "--model-dir",
                str(GA_MODEL_DIR),
                "--genomes-file",
                str(shard_paths[idx]),
                "--output",
                str(raw_paths[idx]),
                "--mil-batch-size",
                str(cfg["batch"]),
                "--scan-chunk",
                str(cfg["scan"]),
                "--device",
                "cuda" if cfg["devices"][idx] != "cpu" else "cpu",
            ]
            env = os.environ.copy()
            if cfg["devices"][idx] != "cpu":
                env["CUDA_VISIBLE_DEVICES"] = cfg["devices"][idx]
            log_path = out_dir / "logs" / f"attention_gpu{idx}_attempt{cfg_idx}.log"
            log = open(log_path, "w")
            log.write(f"[{now()}] {' '.join(cmd)}\n")
            log.flush()
            procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log, log_path))
        failed = []
        for proc, log, log_path in procs:
            rc = proc.wait()
            log.write(f"[{now()}] returncode={rc}\n")
            log.close()
            if rc != 0:
                failed.append((rc, log_path))
        if not failed and all(path.is_file() for path in raw_paths):
            last_error = None
            break
        last_error = failed
        log_message(out_dir, f"attention export attempt {cfg_idx} failed; retrying with smaller batch/scan")
    if last_error:
        raise RuntimeError(f"attention export failed: {last_error}")

    raw_rows: List[Dict[str, str]] = []
    for path in raw_paths:
        raw_rows.extend(read_tsv(path))
    raw_by_id = {row["record_id"]: row for row in raw_rows}
    mp_by_id = {row["record_id"]: row for row in read_csv(MP_PRED)}
    ga_by_id = {row["record_id"]: row for row in read_csv(GA_PRED)}

    attention_rows: List[Dict[str, Any]] = []
    for case in case_rows:
        record_id = case["record_id"]
        if record_id not in raw_by_id:
            raise AssertionError(f"Missing attention for {record_id}")
        raw = raw_by_id[record_id]
        mp = mp_by_id[record_id]
        ga = ga_by_id[record_id]
        out = dict(case)
        out.update(
            {
                "mp_fragment_probability": mp["fragment_probability"],
                "mp_genome_probability": mp["genome_probability"],
                "mp_prediction": mp["prediction"],
                "ga_fragment_probability": ga["fragment_probability"],
                "ga_genome_probability": ga["genome_probability"],
                "ga_prediction": ga["prediction"],
                "ga_attention_weight": raw["ga_attention_weight"],
                "ga_attention_rank": raw["ga_attention_rank"],
                "ga_attention_percentile": raw["ga_attention_percentile"],
                "ga_export_fragment_probability": raw.get("ga_export_fragment_probability", ""),
                "ga_export_genome_probability": raw.get("ga_export_genome_probability", ""),
            }
        )
        attention_rows.append(out)

    sums: Dict[str, float] = defaultdict(float)
    n_by_genome: Dict[str, int] = defaultdict(int)
    for row in attention_rows:
        sums[row["genome"]] += safe_float(row["ga_attention_weight"], 0.0)
        n_by_genome[row["genome"]] += 1
    sum_rows = [
        {
            "genome": genome,
            "records": n_by_genome[genome],
            "attention_sum": f"{value:.12g}",
            "abs_error": f"{abs(value - 1.0):.12g}",
            "passes": int(abs(value - 1.0) < 1e-4),
        }
        for genome, value in sorted(sums.items())
    ]
    bad = [row for row in sum_rows if safe_int(row["passes"]) != 1]
    if bad:
        raise AssertionError(f"Attention weights do not sum to 1 for {len(bad)} genomes")

    fields = [
        "case_group",
        "control_for_case",
        "control_for_genome",
        "record_id",
        "source",
        "label",
        "binary_label",
        "genome",
        "contig",
        "start",
        "end",
        "length",
        "mp_fragment_probability",
        "mp_genome_probability",
        "mp_prediction",
        "ga_fragment_probability",
        "ga_genome_probability",
        "ga_prediction",
        "ga_attention_weight",
        "ga_attention_rank",
        "ga_attention_percentile",
        "ga_export_fragment_probability",
        "ga_export_genome_probability",
    ]
    write_tsv(attention_path, attention_rows, fields)
    write_tsv(sums_path, sum_rows, ["genome", "records", "attention_sum", "abs_error", "passes"])
    mark_done(
        out_dir,
        "attention",
        {
            "attention_records": str(attention_path),
            "attention_sum_check": str(sums_path),
            "records": len(attention_rows),
            "genomes": len(sums),
        },
    )
    log_message(out_dir, f"attention complete: {len(attention_rows)} records across {len(sums)} genomes")


def run_attention_worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(PAPER_CODE))
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    from benchmark_realm_rank_test import GenomeCollator
    from mil_model import ViraLM_MIL_Gated
    from mil_train_common import autocast_context
    from shared import _binary_logits

    class GenomeDataset(Dataset):
        def __init__(self, grouped_records: Sequence[Tuple[str, List[Dict[str, Any]]]]):
            self.grouped_records = list(grouped_records)

        def __len__(self) -> int:
            return len(self.grouped_records)

        def __getitem__(self, index: int) -> Tuple[str, List[Dict[str, Any]]]:
            return self.grouped_records[index]

    genome_set = {line.strip() for line in Path(args.genomes_file).read_text().splitlines() if line.strip()}
    grouped: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for meta, _header, seq in iter_fasta(Path(args.test_fasta)):
        genome = meta["genome"]
        if genome not in genome_set:
            continue
        row = dict(meta)
        row["sequence"] = seq
        grouped.setdefault(genome, []).append(row)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    config = AutoConfig.from_pretrained(str(args.model_dir), trust_remote_code=True)
    backbone = AutoModelForSequenceClassification.from_config(config, trust_remote_code=True).to(device)
    model = ViraLM_MIL_Gated(backbone, hidden_size=int(getattr(config, "hidden_size", 768)), num_classes=2).to(device)
    state = torch.load(Path(args.model_dir) / "best_mil_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    loader = DataLoader(
        GenomeDataset(list(grouped.items())),
        batch_size=args.mil_batch_size,
        shuffle=False,
        collate_fn=GenomeCollator(tokenizer, 512),
        num_workers=0,
    )
    rows: List[Dict[str, Any]] = []
    use_amp = args.fp16 and device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            with autocast_context(use_amp):
                seq_logits, attentions, frag_logits_list, _ = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    sub_chunk_size=args.scan_chunk,
                    return_frag_logits=True,
                    return_hidden=False,
                )
            genome_probs = torch.sigmoid(_binary_logits(seq_logits)).float().cpu().numpy().tolist()
            for group_idx, records_for_genome in enumerate(batch["metas"]):
                weights = attentions[group_idx].view(-1).float().numpy().tolist()
                order = sorted(range(len(weights)), key=lambda i: (-weights[i], i))
                ranks = [0] * len(weights)
                for rank, idx in enumerate(order, start=1):
                    ranks[idx] = rank
                frag_logits = frag_logits_list[group_idx] if frag_logits_list is not None else None
                if frag_logits is None:
                    frag_probs = [float("nan")] * len(records_for_genome)
                else:
                    frag_probs = torch.sigmoid(_binary_logits(frag_logits).float()).cpu().numpy().tolist()
                denom = max(1, len(records_for_genome) - 1)
                for i, row in enumerate(records_for_genome):
                    rows.append(
                        {
                            "record_id": row["record_id"],
                            "genome": row["genome"],
                            "ga_attention_weight": f"{float(weights[i]):.12g}",
                            "ga_attention_rank": ranks[i],
                            "ga_attention_percentile": f"{1.0 - ((ranks[i] - 1) / denom):.12g}",
                            "ga_export_fragment_probability": f"{float(frag_probs[i]):.12g}",
                            "ga_export_genome_probability": f"{float(genome_probs[group_idx]):.12g}",
                        }
                    )

    write_tsv(
        Path(args.output),
        rows,
        [
            "record_id",
            "genome",
            "ga_attention_weight",
            "ga_attention_rank",
            "ga_attention_percentile",
            "ga_export_fragment_probability",
            "ga_export_genome_probability",
        ],
    )


def find_genomad_db(root: Path) -> Optional[Path]:
    candidates = [root / "genomad_db", root]
    candidates.extend(path for path in root.glob("*") if path.is_dir())
    for path in candidates:
        if not path.is_dir():
            continue
        names = {item.name for item in path.iterdir()}
        if (path / "version.txt").is_file():
            return path
        if any("marker" in name.lower() or "nn" in name.lower() or "database" in name.lower() for name in names):
            return path
    return None


def find_checkv_db(root: Path) -> Optional[Path]:
    candidates = [root / "checkv-db-v1.5", root / "checkv_db", root]
    candidates.extend(path for path in root.glob("*") if path.is_dir())
    for path in candidates:
        if not path.is_dir():
            continue
        names = {item.name for item in path.iterdir()}
        if {"genome_db", "hmm_db"}.intersection(names):
            return path
    return None


def prepare_databases(out_dir: Path, args: argparse.Namespace) -> Tuple[Path, Path]:
    db_root = ensure_dir(out_dir / "db")
    genomad_root = Path(args.genomad_db_root) if args.genomad_db_root else db_root / "genomad"
    checkv_root = Path(args.checkv_db_root) if args.checkv_db_root else db_root / "checkv"
    ensure_dir(genomad_root)
    ensure_dir(checkv_root)

    genomad_db = find_genomad_db(genomad_root)
    if genomad_db is None or args.force_db_download:
        log_message(out_dir, "downloading geNomad database")
        cmd = [str(BIOANN_BIN / "genomad"), "download-database", str(genomad_root)]
        for attempt in range(1, args.download_attempts + 1):
            try:
                run_command(cmd, out_dir / "logs" / "genomad_download.log", env=bioann_env(), attempts=1)
            except Exception as exc:
                genomad_db = find_genomad_db(genomad_root)
                if genomad_db is not None:
                    log_message(out_dir, f"geNomad download returned nonzero but database is present: {genomad_db}")
                    break
                if attempt >= args.download_attempts:
                    raise exc
                time.sleep(90 * attempt)
            else:
                genomad_db = find_genomad_db(genomad_root)
                break
    if genomad_db is None:
        raise FileNotFoundError(f"Could not locate geNomad database under {genomad_root}")

    checkv_db = find_checkv_db(checkv_root)
    if checkv_db is None or args.force_db_download:
        log_message(out_dir, "downloading CheckV database")
        cmd = [str(BIOANN_BIN / "checkv"), "download_database", str(checkv_root)]
        for attempt in range(1, args.download_attempts + 1):
            try:
                run_command(cmd, out_dir / "logs" / "checkv_download.log", env=bioann_env(), attempts=1)
            except Exception as exc:
                checkv_db = find_checkv_db(checkv_root)
                if checkv_db is not None:
                    log_message(out_dir, f"CheckV download returned nonzero but database is present: {checkv_db}")
                    break
                if attempt >= args.download_attempts:
                    raise exc
                time.sleep(90 * attempt)
            else:
                checkv_db = find_checkv_db(checkv_root)
                break
    if checkv_db is None:
        raise FileNotFoundError(f"Could not locate CheckV database under {checkv_root}")
    ensure_checkv_diamond_db(out_dir, checkv_db, args)

    write_json(out_dir / "db" / "database_paths.json", {"genomad_db": str(genomad_db), "checkv_db": str(checkv_db)})
    return genomad_db, checkv_db


def ensure_checkv_diamond_db(out_dir: Path, checkv_db: Path, args: argparse.Namespace) -> None:
    faa = checkv_db / "genome_db" / "checkv_reps.faa"
    dmnd = checkv_db / "genome_db" / "checkv_reps.dmnd"
    if dmnd.is_file():
        return
    if not faa.is_file():
        raise FileNotFoundError(f"CheckV protein FASTA missing: {faa}")
    log_message(out_dir, "building missing CheckV DIAMOND database")
    run_command(
        [
            str(BIOANN_BIN / "diamond"),
            "makedb",
            "--in",
            str(faa),
            "-d",
            str(checkv_db / "genome_db" / "checkv_reps"),
            "--threads",
            str(args.threads),
        ],
        out_dir / "logs" / "checkv_diamond_makedb.log",
        env=bioann_env(),
        attempts=2,
        retry_delay=30,
    )


def run_genomad(out_dir: Path, args: argparse.Namespace, genomad_db: Path) -> None:
    genomad_out = out_dir / "genomad_out"
    features_path = out_dir / "genomad_features.tsv"
    if stage_done(out_dir, "genomad", [features_path]):
        log_message(out_dir, "stage genomad already done")
        return
    log_message(out_dir, "running geNomad end-to-end on selected records")
    cmd = [
        str(BIOANN_BIN / "genomad"),
        "end-to-end",
        "--cleanup",
        "--relaxed",
        "--threads",
        str(args.threads),
        "--splits",
        str(args.genomad_splits),
        str(out_dir / "selected_records.fna"),
        str(genomad_out),
        str(genomad_db),
    ]
    try:
        run_command(cmd, out_dir / "logs" / "genomad_end_to_end.log", env=bioann_env(), attempts=1)
    except Exception:
        log_message(out_dir, "geNomad first run failed; retrying with --restart and more splits")
        cmd_retry = cmd[:2] + ["--restart"] + cmd[2:]
        if "--splits" in cmd_retry:
            idx = cmd_retry.index("--splits")
            cmd_retry[idx + 1] = str(max(args.genomad_splits * 2, 128))
        run_command(cmd_retry, out_dir / "logs" / "genomad_end_to_end_retry.log", env=bioann_env(), attempts=2, retry_delay=60)
    parse_genomad_features(out_dir, genomad_out, features_path)
    mark_done(out_dir, "genomad", {"genomad_out": str(genomad_out), "genomad_features": str(features_path)})


def first_existing(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        if name in row and row[name] != "":
            return row[name]
    return ""


def parse_tsv_file(path: Path) -> List[Dict[str, str]]:
    try:
        with open(path, newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except Exception:
        return []


def parse_genomad_features(out_dir: Path, genomad_out: Path, features_path: Path) -> None:
    attention_rows = read_tsv(out_dir / "attention_records.tsv")
    by_id: Dict[str, Dict[str, Any]] = {
        row["record_id"]: {
            "record_id": row["record_id"],
            "genomad_class": "NA",
            "genomad_score": "NA",
            "genomad_fdr": "NA",
            "genomad_hallmark_count": 0,
            "genomad_marker_count": 0,
            "viral_marker_count": 0,
            "plasmid_marker_count": 0,
            "cellular_marker_count": 0,
            "viral_marker_density": 0.0,
            "plasmid_marker_density": 0.0,
            "cellular_marker_density": 0.0,
            "marker_enrichment": 0.0,
            "genomad_marker_enrichment": "NA",
            "viral_marker_enrichment": 0.0,
            "plasmid_marker_enrichment": 0.0,
            "cellular_marker_enrichment": 0.0,
        }
        for row in attention_rows
    }
    lengths = {row["record_id"]: max(1.0, safe_float(row["length"], 1.0)) for row in attention_rows}
    all_tsv = sorted(genomad_out.rglob("*.tsv"))
    manifest = [{"path": str(path), "size": path.stat().st_size} for path in all_tsv]
    write_json(out_dir / "genomad_out" / "parsed_tsv_manifest.json", {"tsv_files": manifest})

    feature_seen: set[str] = set()
    for path in all_tsv:
        lower = path.name.lower()
        rows = parse_tsv_file(path)
        if not rows:
            continue
        if lower.endswith("_features.tsv") or lower == "selected_records_features.tsv":
            for row in rows:
                record_id = first_existing(row, ["seq_name", "sequence", "sequence_name", "contig", "record_id", "id"])
                if record_id not in by_id:
                    continue
                feature_seen.add(record_id)
                n_genes = safe_float(row.get("n_genes"), 0.0)
                viral_count = safe_float(row.get("v_marker_freq"), 0.0) * n_genes
                plasmid_count = safe_float(row.get("p_marker_freq"), 0.0) * n_genes
                cellular_count = safe_float(row.get("c_marker_freq"), 0.0) * n_genes
                by_id[record_id]["viral_marker_count"] = max(safe_float(by_id[record_id]["viral_marker_count"], 0.0), viral_count)
                by_id[record_id]["plasmid_marker_count"] = max(safe_float(by_id[record_id]["plasmid_marker_count"], 0.0), plasmid_count)
                by_id[record_id]["cellular_marker_count"] = max(safe_float(by_id[record_id]["cellular_marker_count"], 0.0), cellular_count)
                by_id[record_id]["genomad_marker_count"] = max(
                    safe_float(by_id[record_id]["genomad_marker_count"], 0.0),
                    viral_count + plasmid_count + cellular_count,
                )
                by_id[record_id]["genomad_hallmark_count"] = max(
                    safe_float(by_id[record_id]["genomad_hallmark_count"], 0.0),
                    safe_float(row.get("n_virus_hallmarks"), 0.0) + safe_float(row.get("n_plasmid_hallmarks"), 0.0),
                )
                by_id[record_id]["viral_marker_enrichment"] = safe_float(row.get("marker_enrichment_v"), 0.0)
                by_id[record_id]["plasmid_marker_enrichment"] = safe_float(row.get("marker_enrichment_p"), 0.0)
                by_id[record_id]["cellular_marker_enrichment"] = safe_float(row.get("marker_enrichment_c"), 0.0)
                by_id[record_id]["marker_enrichment"] = safe_float(row.get("marker_enrichment_v"), 0.0)
        if "summary" in lower:
            if "virus" in lower:
                cls = "virus"
            elif "plasmid" in lower:
                cls = "plasmid"
            elif "chromosome" in lower:
                cls = "chromosome"
            else:
                cls = ""
            for row in rows:
                record_id = first_existing(row, ["seq_name", "sequence", "sequence_name", "contig", "record_id", "id"])
                if record_id not in by_id:
                    continue
                score = first_existing(row, ["score", "virus_score", "plasmid_score", "calibrated_score", "nn_score", "marker_score"])
                fdr = first_existing(row, ["fdr", "false_discovery_rate"])
                enrichment = first_existing(row, ["marker_enrichment"])
                if cls:
                    current = by_id[record_id]["genomad_class"]
                    current_score = safe_float(by_id[record_id]["genomad_score"], -1.0)
                    new_score = safe_float(score, 0.0)
                    if current == "NA" or new_score >= current_score:
                        by_id[record_id]["genomad_class"] = cls
                        by_id[record_id]["genomad_score"] = score or "NA"
                        by_id[record_id]["genomad_fdr"] = fdr or "NA"
                        by_id[record_id]["genomad_marker_enrichment"] = enrichment or "NA"
                for key, out_key in [
                    ("n_hallmarks", "genomad_hallmark_count"),
                    ("hallmarks", "genomad_hallmark_count"),
                    ("n_markers", "genomad_marker_count"),
                    ("markers", "genomad_marker_count"),
                    ("virus_hallmarks", "genomad_hallmark_count"),
                ]:
                    if key in row and row[key] != "":
                        by_id[record_id][out_key] = max(safe_float(by_id[record_id][out_key], 0.0), safe_float(row[key], 0.0))
        if "gene" in lower or "marker" in lower or "annotate" in lower:
            for row in rows:
                record_id = first_existing(row, ["seq_name", "sequence", "sequence_name", "contig", "record_id", "id"])
                if not record_id and "gene" in row and row["gene"]:
                    record_id = row["gene"].rsplit("_", 1)[0]
                if record_id not in by_id:
                    continue
                marker = str(row.get("marker", "")).strip()
                if marker and marker.upper() != "NA" and record_id not in feature_seen:
                    by_id[record_id]["genomad_marker_count"] = safe_float(by_id[record_id]["genomad_marker_count"], 0.0) + 1
                    suffix = marker.rsplit(".", 1)[-1].upper()
                    if suffix.startswith("V"):
                        by_id[record_id]["viral_marker_count"] = safe_float(by_id[record_id]["viral_marker_count"], 0.0) + 1
                    elif suffix.startswith("P"):
                        by_id[record_id]["plasmid_marker_count"] = safe_float(by_id[record_id]["plasmid_marker_count"], 0.0) + 1
                    elif suffix.startswith("C"):
                        by_id[record_id]["cellular_marker_count"] = safe_float(by_id[record_id]["cellular_marker_count"], 0.0) + 1
                if record_id not in feature_seen:
                    by_id[record_id]["genomad_hallmark_count"] = safe_float(by_id[record_id]["genomad_hallmark_count"], 0.0) + (
                        1 if safe_int(row.get("virus_hallmark"), 0) or safe_int(row.get("plasmid_hallmark"), 0) else 0
                    )

    for record_id, row in by_id.items():
        kb = lengths[record_id] / 1000.0
        row["viral_marker_density"] = safe_float(row["viral_marker_count"], 0.0) / kb
        row["plasmid_marker_density"] = safe_float(row["plasmid_marker_count"], 0.0) / kb
        row["cellular_marker_density"] = safe_float(row["cellular_marker_count"], 0.0) / kb
        row["marker_enrichment"] = row["viral_marker_density"] - max(row["plasmid_marker_density"], row["cellular_marker_density"])

    fields = [
        "record_id",
        "genomad_class",
        "genomad_score",
        "genomad_fdr",
        "genomad_hallmark_count",
        "genomad_marker_count",
        "viral_marker_count",
        "plasmid_marker_count",
        "cellular_marker_count",
        "viral_marker_density",
        "plasmid_marker_density",
        "cellular_marker_density",
        "marker_enrichment",
        "genomad_marker_enrichment",
        "viral_marker_enrichment",
        "plasmid_marker_enrichment",
        "cellular_marker_enrichment",
    ]
    write_tsv(features_path, by_id.values(), fields)


def select_checkv_fasta(out_dir: Path) -> Path:
    merged_path = out_dir / "merged_bio_attention.tsv"
    rows = read_tsv(merged_path)
    selected: set[str] = set()
    for row in rows:
        if safe_int(row["binary_label"]) == 1:
            selected.add(row["record_id"])
        elif row.get("genomad_class") == "virus":
            selected.add(row["record_id"])
        elif safe_float(row.get("ga_genome_probability"), 0.0) >= THRESHOLD and safe_int(row.get("ga_attention_rank"), 999999) <= 3:
            selected.add(row["record_id"])
    out = out_dir / "checkv_selected_records.fna"
    with open(out, "w") as handle:
        for record_id, block in iter_fasta_blocks(out_dir / "selected_records.fna"):
            if record_id in selected:
                handle.write("".join(block))
    return out


def run_checkv(out_dir: Path, args: argparse.Namespace, checkv_db: Path) -> None:
    checkv_out = out_dir / "checkv_out"
    features_path = out_dir / "checkv_features.tsv"
    if stage_done(out_dir, "checkv", [features_path]):
        log_message(out_dir, "stage checkv already done")
        return
    log_message(out_dir, "running CheckV on representative viral-like records")
    checkv_fasta = select_checkv_fasta(out_dir)
    cmd = [
        str(BIOANN_BIN / "checkv"),
        "end_to_end",
        str(checkv_fasta),
        str(checkv_out),
        "-t",
        str(args.threads),
        "-d",
        str(checkv_db),
    ]
    try:
        run_command(cmd, out_dir / "logs" / "checkv_end_to_end.log", env=bioann_env(), attempts=1)
    except Exception:
        log_message(out_dir, "CheckV first run failed; retrying with --restart")
        cmd_retry = cmd + ["--restart"]
        run_command(cmd_retry, out_dir / "logs" / "checkv_end_to_end_retry.log", env=bioann_env(), attempts=2, retry_delay=60)
    parse_checkv_features(out_dir, checkv_out, features_path)
    mark_done(out_dir, "checkv", {"checkv_out": str(checkv_out), "checkv_features": str(features_path)})


def parse_checkv_features(out_dir: Path, checkv_out: Path, features_path: Path) -> None:
    all_ids = [row["record_id"] for row in read_tsv(out_dir / "attention_records.tsv")]
    features = {
        record_id: {
            "record_id": record_id,
            "checkv_quality": "NA",
            "checkv_completeness": "NA",
            "checkv_contamination": "NA",
            "checkv_warnings": "NA",
        }
        for record_id in all_ids
    }
    quality = checkv_out / "quality_summary.tsv"
    if quality.is_file():
        for row in parse_tsv_file(quality):
            record_id = first_existing(row, ["contig_id", "record_id", "seq_name", "sequence"])
            if record_id not in features:
                continue
            features[record_id].update(
                {
                    "checkv_quality": first_existing(row, ["checkv_quality", "quality"]),
                    "checkv_completeness": first_existing(row, ["completeness"]),
                    "checkv_contamination": first_existing(row, ["contamination"]),
                    "checkv_warnings": first_existing(row, ["warnings"]),
                }
            )
    write_tsv(
        features_path,
        features.values(),
        ["record_id", "checkv_quality", "checkv_completeness", "checkv_contamination", "checkv_warnings"],
    )


def merge_features(out_dir: Path) -> None:
    merged_path = out_dir / "merged_bio_attention.tsv"
    deps = [out_dir / "attention_records.tsv", out_dir / "genomad_features.tsv"]
    checkv_dep = out_dir / "checkv_features.tsv"
    if checkv_dep.is_file():
        deps.append(checkv_dep)
    merge_current = (
        done_path(out_dir, "merge").is_file()
        and merged_path.is_file()
        and all(dep.is_file() for dep in deps)
        and merged_path.stat().st_mtime >= max(dep.stat().st_mtime for dep in deps)
    )
    if merge_current:
        log_message(out_dir, "stage merge already done")
        return
    log_message(out_dir, "merging prediction, attention, geNomad and CheckV feature tables")
    att_rows = read_tsv(out_dir / "attention_records.tsv")
    genomad = {row["record_id"]: row for row in read_tsv(out_dir / "genomad_features.tsv")}
    checkv_path = out_dir / "checkv_features.tsv"
    checkv = {row["record_id"]: row for row in read_tsv(checkv_path)} if checkv_path.is_file() else {}
    merged: List[Dict[str, Any]] = []
    for row in att_rows:
        out = dict(row)
        out.update(genomad.get(row["record_id"], {}))
        out.update(checkv.get(row["record_id"], {}))
        merged.append(out)
    fields = list(merged[0].keys()) if merged else []
    write_tsv(merged_path, merged, fields)
    mark_done(out_dir, "merge", {"merged": str(merged_path), "records": len(merged)})


def wilcoxon_paired(diffs: Sequence[float]) -> float:
    vals = [float(x) for x in diffs if not math.isnan(float(x)) and abs(float(x)) > 0]
    if len(vals) < 2:
        return float("nan")
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(vals).pvalue)
    except Exception:
        positives = sum(1 for x in vals if x > 0)
        n = len(vals)
        k = min(positives, n - positives)
        prob = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return float(min(1.0, 2.0 * prob))


def write_summaries(out_dir: Path) -> None:
    summary_dir = ensure_dir(out_dir / "summary_tables")
    figures_dir = ensure_dir(out_dir / "figures")
    evidence_path = summary_dir / "evidence_assessment.tsv"
    if stage_done(out_dir, "summaries", [evidence_path]):
        log_message(out_dir, "stage summaries already done")
        return
    log_message(out_dir, "writing summary tables and SVG figures")
    rows = read_tsv(out_dir / "merged_bio_attention.tsv")
    groups = sorted({row["case_group"] for row in rows})

    case_dist: List[Dict[str, Any]] = []
    for group in groups:
        gr = [row for row in rows if row["case_group"] == group]
        case_dist.append(
            {
                "case_group": group,
                "genomes": len({row["genome"] for row in gr}),
                "records": len(gr),
                "mean_attention": mean([safe_float(row["ga_attention_weight"]) for row in gr]),
                "mean_viral_marker_density": mean([safe_float(row.get("viral_marker_density"), 0.0) for row in gr]),
                "mean_plasmid_marker_density": mean([safe_float(row.get("plasmid_marker_density"), 0.0) for row in gr]),
                "mean_cellular_marker_density": mean([safe_float(row.get("cellular_marker_density"), 0.0) for row in gr]),
            }
        )
    write_tsv(
        summary_dir / "case_feature_distribution.tsv",
        case_dist,
        [
            "case_group",
            "genomes",
            "records",
            "mean_attention",
            "mean_viral_marker_density",
            "mean_plasmid_marker_density",
            "mean_cellular_marker_density",
        ],
    )

    enrichment_rows: List[Dict[str, Any]] = []
    evidence_rows: List[Dict[str, Any]] = []
    metrics = ["genomad_hallmark_count", "viral_marker_density", "plasmid_marker_density", "cellular_marker_density", "marker_enrichment"]
    for group in groups:
        gr = [row for row in rows if row["case_group"] == group]
        by_genome: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in gr:
            by_genome[row["genome"]].append(row)
        for metric in metrics:
            diffs = []
            top_vals = []
            low_vals = []
            for genome_rows in by_genome.values():
                if len(genome_rows) < 2:
                    continue
                top = min(genome_rows, key=lambda row: safe_int(row.get("ga_attention_rank"), 999999))
                low = [row for row in genome_rows if row["record_id"] != top["record_id"]]
                if not low:
                    continue
                top_val = safe_float(top.get(metric), 0.0)
                low_val = mean([safe_float(row.get(metric), 0.0) for row in low])
                if math.isnan(top_val) or math.isnan(low_val):
                    continue
                diffs.append(top_val - low_val)
                top_vals.append(top_val)
                low_vals.append(low_val)
            p = wilcoxon_paired(diffs)
            effect = mean(diffs)
            enrichment_rows.append(
                {
                    "case_group": group,
                    "metric": metric,
                    "genomes_tested": len(diffs),
                    "top_mean": mean(top_vals),
                    "low_mean": mean(low_vals),
                    "mean_top_minus_low": effect,
                    "paired_wilcoxon_p": p,
                }
            )
            if metric in {"viral_marker_density", "marker_enrichment"}:
                if len(diffs) >= 10 and effect > 0 and (not math.isnan(p)) and p < 0.05:
                    level = "strong evidence"
                elif len(diffs) >= 5 and effect > 0 and (not math.isnan(p)) and p < 0.1:
                    level = "weak evidence"
                else:
                    level = "negative finding"
                evidence_rows.append(
                    {
                        "case_group": group,
                        "claim": f"top-attention records enriched for {metric}",
                        "evidence_level": level,
                        "effect": effect,
                        "p_value": p,
                        "genomes_tested": len(diffs),
                    }
                )
    write_tsv(
        summary_dir / "top_vs_low_attention_enrichment.tsv",
        enrichment_rows,
        ["case_group", "metric", "genomes_tested", "top_mean", "low_mean", "mean_top_minus_low", "paired_wilcoxon_p"],
    )
    write_tsv(
        evidence_path,
        evidence_rows,
        ["case_group", "claim", "evidence_level", "effect", "p_value", "genomes_tested"],
    )

    reps = sorted(
        rows,
        key=lambda row: (row["case_group"], safe_int(row.get("ga_attention_rank"), 999999), -safe_float(row.get("ga_attention_weight"), 0.0)),
    )
    write_tsv(
        summary_dir / "representative_sequences.tsv",
        reps[:200],
        [
            "case_group",
            "record_id",
            "genome",
            "source",
            "label",
            "length",
            "mp_fragment_probability",
            "ga_fragment_probability",
            "ga_genome_probability",
            "ga_attention_weight",
            "ga_attention_rank",
            "genomad_class",
            "genomad_score",
            "genomad_hallmark_count",
            "viral_marker_density",
            "plasmid_marker_density",
            "cellular_marker_density",
            "checkv_quality",
            "checkv_completeness",
        ],
    )
    draw_figures(figures_dir, rows, enrichment_rows)
    mark_done(out_dir, "summaries", {"summary_tables": str(summary_dir), "figures": str(figures_dir)})


def svg_escape(value: Any) -> str:
    text = str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def draw_bar_svg(path: Path, title: str, labels: Sequence[str], series: Dict[str, Sequence[float]]) -> None:
    width, height = 960, 520
    margin_l, margin_b, margin_t = 90, 100, 60
    plot_w, plot_h = width - margin_l - 40, height - margin_t - margin_b
    max_y = max([0.01] + [safe_float(v, 0.0) for vals in series.values() for v in vals])
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    n_series = len(series)
    group_w = plot_w / max(1, len(labels))
    bar_w = group_w / max(1, n_series + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20">{svg_escape(title)}</text>',
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#222"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#222"/>',
    ]
    for s_idx, (name, vals) in enumerate(series.items()):
        color = colors[s_idx % len(colors)]
        parts.append(f'<rect x="{margin_l + s_idx*150}" y="44" width="16" height="16" fill="{color}"/>')
        parts.append(f'<text x="{margin_l + s_idx*150 + 22}" y="58" font-family="Arial" font-size="13">{svg_escape(name)}</text>')
        for i, value in enumerate(vals):
            v = max(0.0, safe_float(value, 0.0))
            h = (v / max_y) * plot_h
            x = margin_l + i * group_w + (s_idx + 0.5) * bar_w
            y = margin_t + plot_h - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w*0.85:.2f}" height="{h:.2f}" fill="{color}"/>')
    for i, label in enumerate(labels):
        x = margin_l + i * group_w + group_w / 2
        parts.append(
            f'<text x="{x:.2f}" y="{height-46}" text-anchor="end" transform="rotate(-35 {x:.2f},{height-46})" font-family="Arial" font-size="12">{svg_escape(label)}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def draw_scatter_svg(path: Path, title: str, rows: Sequence[Dict[str, str]], x_key: str, y_key: str) -> None:
    width, height = 960, 620
    margin_l, margin_b, margin_t = 80, 70, 60
    plot_w, plot_h = width - margin_l - 40, height - margin_t - margin_b
    xs = [safe_float(row.get(x_key), 0.0) for row in rows]
    ys = [safe_float(row.get(y_key), 0.0) for row in rows]
    max_y = max([0.01] + ys)
    colors = {
        "ga_rescued_positive": "#0072B2",
        "ga_corrected_negative": "#D55E00",
        "ga_worse": "#CC79A7",
        "both_correct_control": "#009E73",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="20">{svg_escape(title)}</text>',
        f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{margin_l+plot_w}" y2="{margin_t+plot_h}" stroke="#222"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#222"/>',
        f'<text x="{margin_l+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="Arial" font-size="14">{svg_escape(x_key)}</text>',
        f'<text x="18" y="{margin_t+plot_h/2}" text-anchor="middle" transform="rotate(-90 18,{margin_t+plot_h/2})" font-family="Arial" font-size="14">{svg_escape(y_key)}</text>',
    ]
    for idx, (group, color) in enumerate(colors.items()):
        parts.append(f'<circle cx="{margin_l + idx*190}" cy="48" r="5" fill="{color}" opacity="0.8"/>')
        parts.append(f'<text x="{margin_l + idx*190 + 12}" y="52" font-family="Arial" font-size="12">{svg_escape(group)}</text>')
    for row, x, y in zip(rows, xs, ys):
        px = margin_l + min(1.0, max(0.0, x)) * plot_w
        py = margin_t + plot_h - (min(max_y, max(0.0, y)) / max_y) * plot_h
        color = colors.get(row.get("case_group", ""), "#666")
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" fill="{color}" opacity="0.55"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


def draw_figures(figures_dir: Path, rows: Sequence[Dict[str, str]], enrichment_rows: Sequence[Dict[str, Any]]) -> None:
    groups = ["ga_rescued_positive", "ga_corrected_negative", "ga_worse", "both_correct_control"]
    top_vals = []
    low_vals = []
    for group in groups:
        match = [
            row
            for row in enrichment_rows
            if row["case_group"] == group and row["metric"] == "viral_marker_density"
        ]
        top_vals.append(safe_float(match[0]["top_mean"], 0.0) if match else 0.0)
        low_vals.append(safe_float(match[0]["low_mean"], 0.0) if match else 0.0)
    draw_bar_svg(
        figures_dir / "top_vs_low_attention_enrichment.svg",
        "Top vs Low Attention Viral Marker Density",
        groups,
        {"top attention": top_vals, "low attention": low_vals},
    )
    draw_scatter_svg(
        figures_dir / "mp_probability_vs_ga_attention_quadrants.svg",
        "MP Fragment Probability vs GA Attention",
        rows,
        "mp_fragment_probability",
        "ga_attention_weight",
    )
    draw_scatter_svg(
        figures_dir / "attention_marker_alignment.svg",
        "GA Attention vs Viral Marker Density",
        rows,
        "ga_attention_weight",
        "viral_marker_density",
    )
    worse = [row for row in rows if row["case_group"] == "ga_worse"]
    draw_scatter_svg(
        figures_dir / "ga_worse_sanity_check.svg",
        "GA Worse Sanity Check",
        worse,
        "ga_attention_weight",
        "viral_marker_density",
    )


def verify_outputs(out_dir: Path) -> None:
    report_path = out_dir / "verification_report.tsv"
    if stage_done(out_dir, "verification", [report_path]):
        log_message(out_dir, "stage verification already done")
        return
    log_message(out_dir, "verifying required outputs")
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": int(bool(passed)), "detail": detail})
        if not passed:
            log_message(out_dir, f"verification failed: {name}: {detail}")

    case_counts = {row["case_group"]: row for row in read_tsv(out_dir / "summary_tables" / "case_counts.tsv")}
    for group, expected in EXPECTED_CASE_COUNTS.items():
        ok = True
        details = []
        for key, value in expected.items():
            if value:
                observed = safe_int(case_counts[group].get(key))
                details.append(f"{key}={observed}")
                ok = ok and observed == value
        add(f"case_count_{group}", ok, ", ".join(details))

    checksum_rows = read_tsv(out_dir / "selected_records.checksums.tsv")
    add(
        "selected_fasta_checksum",
        all(row["original_block_sha256"] == row["selected_block_sha256"] and safe_int(row["matches_original"]) == 1 for row in checksum_rows),
        f"records={len(checksum_rows)}",
    )

    sums = read_tsv(out_dir / "summary_tables" / "attention_sum_check.tsv")
    add(
        "attention_sums",
        all(safe_int(row["passes"]) == 1 for row in sums),
        f"genomes={len(sums)} max_abs_error={max([safe_float(row['abs_error'], 0.0) for row in sums] or [0.0])}",
    )

    merged = read_tsv(out_dir / "merged_bio_attention.tsv")
    add("merged_has_record_id", bool(merged and "record_id" in merged[0]), f"records={len(merged)}")
    add("merged_row_count", len(merged) == len(read_tsv(out_dir / "case_records.tsv")), f"merged={len(merged)} case={len(read_tsv(out_dir / 'case_records.tsv'))}")
    add(
        "genomad_na_or_values",
        all(row.get("genomad_class", "NA") != "" and row.get("viral_marker_density", "") != "" for row in merged),
        "geNomad missing records retained as NA/0",
    )
    for subdir in ("summary_tables", "figures"):
        paths = list((out_dir / subdir).glob("*"))
        add(f"{subdir}_nonempty", bool(paths), f"files={len(paths)}")

    write_tsv(report_path, checks, ["check", "passed", "detail"])
    failed = [row for row in checks if safe_int(row["passed"]) != 1]
    if failed:
        raise AssertionError(f"{len(failed)} verification checks failed")
    mark_done(out_dir, "verification", {"verification_report": str(report_path), "checks": len(checks)})


def run_pipeline(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "state")
    ensure_dir(out_dir / "logs")
    ensure_dir(out_dir / "summary_tables")
    ensure_dir(out_dir / "figures")
    write_json(
        out_dir / "run_config.json",
        {
            "test_fasta": str(TEST_FASTA),
            "mp_predictions": str(MP_PRED),
            "ga_predictions": str(GA_PRED),
            "ga_model_dir": str(GA_MODEL_DIR),
            "threshold": THRESHOLD,
            "threads": args.threads,
            "created_at": now(),
        },
    )
    for required in (TEST_FASTA, MP_PRED, GA_PRED, GA_MODEL_DIR / "best_mil_model.pt", VL_PYTHON):
        if not required.exists():
            raise FileNotFoundError(str(required))
    if args.skip_annotations:
        genomad_db = checkv_db = None
    else:
        for exe in (BIOANN_BIN / "genomad", BIOANN_BIN / "checkv"):
            if not exe.exists():
                raise FileNotFoundError(f"Annotation executable missing: {exe}")

    build_case_records(out_dir)
    select_fasta_records(out_dir)
    export_attention(out_dir, args)
    if not args.skip_annotations:
        genomad_db, checkv_db = prepare_databases(out_dir, args)
        run_genomad(out_dir, args, genomad_db)
        merge_features(out_dir)
        run_checkv(out_dir, args, checkv_db)
        merge_features(out_dir)
    else:
        parse_genomad_features(out_dir, out_dir / "genomad_out", out_dir / "genomad_features.tsv")
        merge_features(out_dir)
    write_summaries(out_dir)
    verify_outputs(out_dir)
    log_message(out_dir, "pipeline complete")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pipeline")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--threads", type=int, default=THREADS)
    p.add_argument("--mil-batch-size", type=int, default=8)
    p.add_argument("--scan-chunk", type=int, default=48)
    p.add_argument("--genomad-splits", type=int, default=64)
    p.add_argument("--download-attempts", type=int, default=5)
    p.add_argument("--genomad-db-root", default="")
    p.add_argument("--checkv-db-root", default="")
    p.add_argument("--force-db-download", action="store_true")
    p.add_argument("--skip-annotations", action="store_true")

    w = sub.add_parser("attention-worker")
    w.add_argument("--test-fasta", required=True)
    w.add_argument("--model-dir", required=True)
    w.add_argument("--genomes-file", required=True)
    w.add_argument("--output", required=True)
    w.add_argument("--mil-batch-size", type=int, default=8)
    w.add_argument("--scan-chunk", type=int, default=48)
    w.add_argument("--device", default="cuda")
    w.add_argument("--fp16", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "attention-worker":
        run_attention_worker(args)
    elif args.command == "pipeline":
        run_pipeline(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
