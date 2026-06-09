#!/usr/bin/env python3
"""Build Realm-Rank v4 train/dev/test FASTA files.

Realm-Rank v4 keeps the v3 source policy and decontamination flow, but fixes
the model-selection split at dataset-build time:

* nonviral candidates are decontaminated against viral sequences before split
  balancing;
* selected genomes/source records are split three ways as train/dev/test;
* dev is BLAST-filtered against train, and test against final train+dev;
* final held-out splits are rebalanced after BLAST removal without deleting
  train fragments to compensate for held-out leakage.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from realm_rank_dataset_common import (  # noqa: E402
    DEFAULT_SEED,
    GENUS_CAPS,
    LARGE_DOWNSAMPLE_SOURCES,
    NONVIRAL_SOURCES,
    REALM_GROUPS,
    GenomeRow,
    assign_viral_realms,
    build_decontam_intervals,
    build_virus_fasta,
    create_fragment_indexes,
    decontaminate_nonviral_fasta,
    discover_files,
    discover_nonviral_genomes,
    dump_final_fragments,
    init_fragment_db,
    insert_fragments_for_fasta,
    interval_coverage,
    iter_fasta,
    iter_fragment_coords,
    load_contig_info,
    load_genome_split,
    make_blast_db,
    max_distribution_delta,
    merge_intervals,
    quantiles_from_length_counts,
    read_tsv,
    resolve_threads,
    resolve_tool,
    run_blastn,
    selected_bp_by_split,
    stable_hash_hex,
    stable_int,
    update_unselected,
    validate_input_files,
    write_fasta,
    write_genome_split,
    write_nonviral_metadata,
    write_qc,
    write_selected_nonviral_fasta,
    write_tsv,
    write_virus_metadata,
)


GAMIL_ROOT = Path(os.environ.get("GAMIL_ROOT", SCRIPT_DIR.parents[1])).resolve()
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", GAMIL_ROOT / "raw_data" / "local_sources")).resolve()
PROCESSED_DATA_ROOT = Path(os.environ.get("PROCESSED_DATA_ROOT", GAMIL_ROOT / "processed_data")).resolve()
SPLITS = ("train", "dev", "test")
HELDOUT_SPLITS = ("dev", "test")
FASTA_WIDTH = 80


def log(message: str) -> None:
    print(f"[realm-rank-v4] {message}", flush=True)


def ensure_dirs(output_dir: Path, force: bool) -> tuple[Path, Path, Path, Path]:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any((output_dir / name).exists() for name in ("train.fasta.gz", "dev.fasta.gz", "test.fasta.gz")) and not force:
        raise RuntimeError(f"{output_dir} already contains v4 output; use --force to rebuild")
    metadata_dir = output_dir / "metadata"
    qc_dir = output_dir / "qc"
    work_dir = output_dir / "work"
    blast_dir = output_dir / "blast"
    for path in (metadata_dir, qc_dir, work_dir, blast_dir):
        path.mkdir(parents=True, exist_ok=True)
    return metadata_dir, qc_dir, work_dir, blast_dir


def split_count(n: int, fraction: float, min_train: int, already_reserved: int = 0) -> int:
    if n <= min_train + already_reserved:
        return 0
    count = int(round(n * fraction))
    if fraction > 0 and n >= 3:
        count = max(1, count)
    return min(count, n - min_train - already_reserved)


def split_genomes_three_way(
    virus_rows: list[GenomeRow],
    nonviral_rows: list[GenomeRow],
    seed: int,
    dev_fraction: float,
    test_fraction: float,
) -> None:
    selected = [row for row in virus_rows] + [row for row in nonviral_rows if row.selected]
    by_label: dict[str, list[GenomeRow]] = defaultdict(list)
    for row in selected:
        by_label[row.label].append(row)

    for label, rows in sorted(by_label.items()):
        ordered = sorted(rows, key=lambda row: row.genome_id)
        rng = random.Random(stable_int(seed, "split-v4", label))
        rng.shuffle(ordered)

        n = len(ordered)
        test_n = split_count(n, test_fraction, min_train=1)
        dev_n = split_count(n, dev_fraction, min_train=1, already_reserved=test_n)
        test_ids = {row.genome_id for row in ordered[:test_n]}
        dev_ids = {row.genome_id for row in ordered[test_n : test_n + dev_n]}

        for row in rows:
            if row.genome_id in test_ids:
                row.split = "test"
            elif row.genome_id in dev_ids:
                row.split = "dev"
            else:
                row.split = "train"

    for row in nonviral_rows:
        if not row.selected:
            row.split = "excluded"


def write_skipped_viral_taxonomy(virus_rows: list[GenomeRow], metadata_dir: Path) -> None:
    for row in virus_rows:
        row.label = "SmallRealm"
        row.realm = ""
        row.taxid = ""
        row.scientific_name = ""
        row.query_status = "skipped_taxonomy"
    write_virus_metadata(metadata_dir / "virus_accession_realm.tsv", virus_rows)


def stable_row_order(seed: int, fragment_id: str, purpose: str) -> int:
    return stable_int(seed, purpose, fragment_id)


def length_hist(conn: sqlite3.Connection, split: str, supergroup: str) -> dict[int, int]:
    return {
        int(bin_start): int(count)
        for bin_start, count in conn.execute(
            """
            SELECT length_bin, COUNT(*)
            FROM fragments
            WHERE selected=1 AND split=? AND supergroup=?
            GROUP BY length_bin
            """,
            (split, supergroup),
        )
    }


def bp_by_length_bin(conn: sqlite3.Connection, split: str, supergroup: str) -> dict[int, int]:
    return {
        int(bin_start): int(total_bp or 0)
        for bin_start, total_bp in conn.execute(
            """
            SELECT length_bin, SUM(length)
            FROM fragments
            WHERE selected=1 AND split=? AND supergroup=?
            GROUP BY length_bin
            """,
            (split, supergroup),
        )
    }


def allocate_equal_source_bp_targets(available_by_source: dict[str, int], target_bp: int) -> dict[str, int]:
    if target_bp <= 0:
        return {source: 0 for source in NONVIRAL_SOURCES}
    active = {source for source in NONVIRAL_SOURCES if available_by_source.get(source, 0) > 0}
    targets = {source: 0 for source in NONVIRAL_SOURCES}
    remaining = target_bp
    while active and remaining > 0:
        share = remaining / len(active)
        capped = [source for source in active if available_by_source.get(source, 0) <= share]
        if not capped:
            base = remaining // len(active)
            extra = remaining % len(active)
            for idx, source in enumerate(sorted(active)):
                targets[source] = min(available_by_source.get(source, 0), base + (1 if idx < extra else 0))
            break
        for source in capped:
            amount = available_by_source.get(source, 0)
            targets[source] = amount
            remaining -= amount
            active.remove(source)
    return targets


def downsample_large_sources_to_virus_bp_v4(
    conn: sqlite3.Connection,
    seed: int,
    qc_dir: Path,
    splits: Iterable[str] = SPLITS,
) -> None:
    virus_bp = selected_bp_by_split(conn, "supergroup='virus'")
    rows_out: list[tuple[object, ...]] = []
    for source in LARGE_DOWNSAMPLE_SOURCES:
        for split in splits:
            target_bp = virus_bp.get(split, 0)
            rows = list(
                conn.execute(
                    """
                    SELECT fragment_id, length
                    FROM fragments
                    WHERE selected=1 AND source=? AND split=?
                    """,
                    (source, split),
                )
            )
            before_bp = sum(int(length) for _fragment_id, length in rows)
            if before_bp <= target_bp or target_bp <= 0:
                rows_out.append((source, split, before_bp, target_bp, before_bp, 0, "kept_all"))
                continue
            rows.sort(key=lambda row: stable_row_order(seed, str(row[0]), f"large-source-{source}-{split}"))
            keep: set[str] = set()
            kept_bp = 0
            for fragment_id, length in rows:
                length = int(length)
                if kept_bp + length <= target_bp:
                    keep.add(str(fragment_id))
                    kept_bp += length
            drop = [str(fragment_id) for fragment_id, _length in rows if str(fragment_id) not in keep]
            update_unselected(conn, drop, f"{source}_bp_downsample")
            rows_out.append((source, split, before_bp, target_bp, kept_bp, len(drop), "downsampled"))
    write_tsv(
        qc_dir / "large_source_downsample.tsv",
        ["source", "split", "source_bp_before", "virus_bp_target", "source_bp_after", "fragments_dropped", "status"],
        rows_out,
    )


def balance_nonvirus_length_distribution_v4(
    conn: sqlite3.Connection,
    seed: int,
    qc_dir: Path,
    threshold: float,
    splits: Iterable[str] = SPLITS,
) -> None:
    actions: list[tuple[object, ...]] = []
    for split in splits:
        virus = length_hist(conn, split, "virus")
        nonvirus = length_hist(conn, split, "nonvirus")
        delta = max_distribution_delta(virus, nonvirus)
        if delta <= threshold or not virus or not nonvirus:
            actions.append((split, f"{delta:.8f}", 0, "not_needed"))
            continue
        virus_total = sum(virus.values())
        proportions = {bin_start: count / virus_total for bin_start, count in virus.items() if count > 0}
        feasible_totals = [
            math.floor(nonvirus.get(bin_start, 0) / proportion)
            for bin_start, proportion in proportions.items()
            if proportion > 0
        ]
        target_total = min([sum(nonvirus.values())] + feasible_totals) if feasible_totals else 0
        if target_total <= 0:
            actions.append((split, f"{delta:.8f}", 0, "failed_no_feasible_nonvirus_target"))
            continue
        targets = {
            bin_start: min(nonvirus.get(bin_start, 0), int(round(target_total * prop)))
            for bin_start, prop in proportions.items()
        }
        target_sum = sum(targets.values())
        while target_sum > target_total:
            bin_start = max(targets, key=lambda b: targets[b] - target_total * proportions[b])
            if targets[bin_start] == 0:
                break
            targets[bin_start] -= 1
            target_sum -= 1
        while target_sum < target_total:
            candidates = [b for b in proportions if targets.get(b, 0) < nonvirus.get(b, 0)]
            if not candidates:
                break
            bin_start = min(candidates, key=lambda b: targets.get(b, 0) - target_total * proportions[b])
            targets[bin_start] = targets.get(bin_start, 0) + 1
            target_sum += 1
        dropped = 0
        for bin_start, available in nonvirus.items():
            target = targets.get(bin_start, 0)
            if available <= target:
                continue
            rows = [
                str(row[0])
                for row in conn.execute(
                    """
                    SELECT fragment_id
                    FROM fragments
                    WHERE selected=1 AND split=? AND supergroup='nonvirus' AND length_bin=?
                    """,
                    (split, bin_start),
                )
            ]
            rows.sort(key=lambda fragment_id: stable_row_order(seed, fragment_id, f"length-balance-{split}-{bin_start}"))
            drop = rows[target:]
            update_unselected(conn, drop, "length_bin_downsample")
            dropped += len(drop)
        new_delta = max_distribution_delta(length_hist(conn, split, "virus"), length_hist(conn, split, "nonvirus"))
        actions.append((split, f"{delta:.8f}", dropped, f"downsampled_new_delta={new_delta:.8f}"))
    write_tsv(
        qc_dir / "length_balance_actions.tsv",
        ["split", "initial_max_abs_bin_delta", "fragments_dropped", "status"],
        actions,
    )


def selected_fragment_ids(conn: sqlite3.Connection) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {split: set() for split in SPLITS}
    for split, fragment_id in conn.execute("SELECT split, fragment_id FROM fragments WHERE selected=1"):
        if split in result:
            result[str(split)].add(str(fragment_id))
    return result


def keep_ids_by_bp(rows: list[tuple[str, int]], target_bp: int, seed: int, purpose: str) -> set[str]:
    if target_bp <= 0:
        return set()
    ordered = sorted(rows, key=lambda row: stable_row_order(seed, row[0], purpose))
    keep: set[str] = set()
    kept_bp = 0
    for fragment_id, length in ordered:
        length = int(length)
        if kept_bp + length <= target_bp:
            keep.add(fragment_id)
            kept_bp += length
    return keep


def downsample_binary_to_common_bp_by_length_bin(
    conn: sqlite3.Connection,
    seed: int,
    qc_dir: Path,
    splits: Iterable[str],
    reason: str,
) -> None:
    """Downsample the majority binary class in each split/bin.

    The function never creates new fragments and never moves fragments between
    splits. For nonviral rows, per-bin target bp is distributed across source
    categories so source composition remains as even as the available data
    permits.
    """

    summary_rows: list[tuple[object, ...]] = []
    bin_rows: list[tuple[object, ...]] = []
    for split in splits:
        before = {
            str(supergroup): (int(count), int(total_bp or 0))
            for supergroup, count, total_bp in conn.execute(
                """
                SELECT supergroup, COUNT(*), SUM(length)
                FROM fragments
                WHERE selected=1 AND split=?
                GROUP BY supergroup
                """,
                (split,),
            )
        }
        all_bins = sorted(
            {
                int(bin_start)
                for (bin_start,) in conn.execute(
                    """
                    SELECT DISTINCT length_bin
                    FROM fragments
                    WHERE selected=1 AND split=?
                    """,
                    (split,),
                )
            }
        )
        dropped_ids: list[str] = []
        for bin_start in all_bins:
            rows_by_super: dict[str, list[tuple[str, int, str]]] = {"virus": [], "nonvirus": []}
            for fragment_id, supergroup, source, length in conn.execute(
                """
                SELECT fragment_id, supergroup, source, length
                FROM fragments
                WHERE selected=1 AND split=? AND length_bin=?
                """,
                (split, bin_start),
            ):
                rows_by_super[str(supergroup)].append((str(fragment_id), int(length), str(source)))

            virus_bp = sum(length for _fid, length, _source in rows_by_super["virus"])
            nonvirus_bp = sum(length for _fid, length, _source in rows_by_super["nonvirus"])
            target_bp = min(virus_bp, nonvirus_bp)

            keep_by_super: dict[str, set[str]] = {"virus": set(), "nonvirus": set()}
            if virus_bp <= target_bp:
                keep_by_super["virus"] = {fid for fid, _length, _source in rows_by_super["virus"]}
            else:
                keep_by_super["virus"] = keep_ids_by_bp(
                    [(fid, length) for fid, length, _source in rows_by_super["virus"]],
                    target_bp,
                    seed,
                    f"{reason}-{split}-{bin_start}-virus",
                )

            if nonvirus_bp <= target_bp:
                keep_by_super["nonvirus"] = {fid for fid, _length, _source in rows_by_super["nonvirus"]}
            else:
                available_by_source: dict[str, int] = defaultdict(int)
                rows_by_source: dict[str, list[tuple[str, int]]] = defaultdict(list)
                for fid, length, source in rows_by_super["nonvirus"]:
                    available_by_source[source] += length
                    rows_by_source[source].append((fid, length))
                min_nonvirus_len = min(length for _fid, length, _source in rows_by_super["nonvirus"])
                if target_bp < min_nonvirus_len * max(1, len(rows_by_source)):
                    keep = keep_ids_by_bp(
                        [(fid, length) for fid, length, _source in rows_by_super["nonvirus"]],
                        target_bp,
                        seed,
                        f"{reason}-{split}-{bin_start}-nonvirus",
                    )
                else:
                    source_targets = allocate_equal_source_bp_targets(dict(available_by_source), target_bp)
                    keep = set()
                    for source, rows in rows_by_source.items():
                        keep.update(
                            keep_ids_by_bp(
                                rows,
                                source_targets.get(source, 0),
                                seed,
                                f"{reason}-{split}-{bin_start}-nonvirus-{source}",
                            )
                        )
                keep_by_super["nonvirus"] = keep

            for supergroup in ("virus", "nonvirus"):
                before_count = len(rows_by_super[supergroup])
                before_bp = sum(length for _fid, length, _source in rows_by_super[supergroup])
                keep = keep_by_super[supergroup]
                after_count = len(keep)
                after_bp = sum(length for fid, length, _source in rows_by_super[supergroup] if fid in keep)
                dropped_ids.extend(fid for fid, _length, _source in rows_by_super[supergroup] if fid not in keep)
                bin_rows.append(
                    (
                        reason,
                        split,
                        bin_start,
                        supergroup,
                        before_count,
                        before_bp,
                        target_bp,
                        after_count,
                        after_bp,
                        before_count - after_count,
                    )
                )

        update_unselected(conn, dropped_ids, reason)
        after = {
            str(supergroup): (int(count), int(total_bp or 0))
            for supergroup, count, total_bp in conn.execute(
                """
                SELECT supergroup, COUNT(*), SUM(length)
                FROM fragments
                WHERE selected=1 AND split=?
                GROUP BY supergroup
                """,
                (split,),
            )
        }
        for supergroup in ("virus", "nonvirus"):
            before_count, before_bp = before.get(supergroup, (0, 0))
            after_count, after_bp = after.get(supergroup, (0, 0))
            summary_rows.append(
                (
                    reason,
                    split,
                    supergroup,
                    before_count,
                    before_bp,
                    after_count,
                    after_bp,
                    before_count - after_count,
                    before_bp - after_bp,
                )
            )

    suffix = reason.replace(" ", "_")
    write_tsv(
        qc_dir / f"{suffix}.tsv",
        [
            "reason",
            "split",
            "supergroup",
            "fragments_before",
            "bp_before",
            "fragments_after",
            "bp_after",
            "fragments_dropped",
            "bp_dropped",
        ],
        summary_rows,
    )
    write_tsv(
        qc_dir / f"{suffix}_by_bin.tsv",
        [
            "reason",
            "split",
            "length_bin",
            "supergroup",
            "fragments_before",
            "bp_before",
            "common_bp_target",
            "fragments_after",
            "bp_after",
            "fragments_dropped",
        ],
        bin_rows,
    )


def write_selected_fragment_fastas_v4(
    conn: sqlite3.Connection,
    fasta_infos: list[tuple[Path, dict[str, dict[str, str]]]],
    work_dir: Path,
    min_len: int,
    max_len: int,
    seed: int,
) -> dict[str, Path]:
    selected = selected_fragment_ids(conn)
    paths = {split: work_dir / f"{split}.prefilter.fasta" for split in SPLITS}
    handles = {split: open(path, "wt") for split, path in paths.items()}
    try:
        for fasta_path, contig_info in fasta_infos:
            for contig_id, _header, seq in iter_fasta(fasta_path):
                info = contig_info[contig_id]
                split = info["split"]
                if split not in handles:
                    continue
                wanted = selected[split]
                for start, end, frag_len in iter_fragment_coords(len(seq), min_len, max_len, seed, contig_id):
                    fragment_id = "frag_" + stable_hash_hex(contig_id, start, end, frag_len)
                    if fragment_id not in wanted:
                        continue
                    subseq = seq[start - 1 : end]
                    desc = (
                        f"label={info['label']} source={info['source']} split={split} "
                        f"genome={info['genome_id']} contig={contig_id} start={start} end={end} length={frag_len}"
                    )
                    write_fasta(handles[split], fragment_id, subseq, desc)
    finally:
        for handle in handles.values():
            handle.close()
    return paths


def write_plain_filtered_fasta(prefilter: Path, output_path: Path, keep_ids: set[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(prefilter, "rt") as src, open(tmp_path, "wt") as dst:
        keep = True
        for line in src:
            if line.startswith(">"):
                fragment_id = line[1:].split(None, 1)[0]
                keep = fragment_id in keep_ids
            if keep:
                dst.write(line)
    tmp_path.replace(output_path)


def write_gzip_filtered_fasta(prefilter: Path, output_path: Path, keep_ids: set[str], threads: int, compression_level: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pigz = shutil.which("pigz")
    if pigz:
        with open(prefilter, "rb") as src, open(tmp_path, "wb") as dst:
            proc = subprocess.Popen(
                [pigz, "-p", str(max(1, threads)), f"-{compression_level}", "-c"],
                stdin=subprocess.PIPE,
                stdout=dst,
            )
            assert proc.stdin is not None
            keep = True
            try:
                for line in src:
                    if line.startswith(b">"):
                        fragment_id = line[1:].split(None, 1)[0].decode("ascii")
                        keep = fragment_id in keep_ids
                    if keep:
                        proc.stdin.write(line)
            finally:
                proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, proc.args)
    else:
        with open(prefilter, "rt") as src, gzip.open(tmp_path, "wt", compresslevel=compression_level) as dst:
            keep = True
            for line in src:
                if line.startswith(">"):
                    fragment_id = line[1:].split(None, 1)[0]
                    keep = fragment_id in keep_ids
                if keep:
                    dst.write(line)
    tmp_path.replace(output_path)


def concat_plain_fastas(paths: Iterable[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(tmp_path, "wb") as dst:
        for path in paths:
            with open(path, "rb") as src:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp_path.replace(output_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def reuse_selected_nonviral_outputs(
    reuse_dir: Path,
    nonviral_rows: list[GenomeRow],
    work_dir: Path,
    metadata_dir: Path,
) -> tuple[Path, Path]:
    reuse_work = reuse_dir / "work"
    selected_fasta = reuse_work / "nonviral_selected.fasta"
    contigs_tsv = reuse_work / "nonviral_contigs.tsv"
    missing = [str(path) for path in (selected_fasta, contigs_tsv) if not path.exists()]
    if missing:
        raise RuntimeError(f"--reuse-decontam-from is missing selected nonviral files: {missing}")
    write_nonviral_metadata(metadata_dir, nonviral_rows)
    out_selected = work_dir / "nonviral_selected.fasta"
    out_contigs = work_dir / "nonviral_contigs.tsv"
    link_or_copy(selected_fasta, out_selected)
    link_or_copy(contigs_tsv, out_contigs)
    return out_selected, out_contigs


def reuse_decontamination_outputs(
    reuse_dir: Path,
    current_virus_fasta: Path,
    work_dir: Path,
    metadata_dir: Path,
    blast_dir: Path,
) -> Path:
    reuse_work = reuse_dir / "work"
    reuse_metadata = reuse_dir / "metadata"
    required = [
        reuse_work / "virus_all.fasta",
        reuse_work / "nonviral_selected.fasta",
        reuse_work / "nonviral_contigs.tsv",
        reuse_work / "nonviral_vs_virus.tsv",
        reuse_work / "nonviral_virus_intervals.tsv",
        reuse_work / "nonviral_cleaned.fasta",
        reuse_metadata / "nonviral_decontamination.tsv",
        reuse_metadata / "nonviral_decontamination_contigs.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"--reuse-decontam-from is missing required files: {missing}")

    current_virus_sha = sha256_file(current_virus_fasta)
    reuse_virus_sha = sha256_file(reuse_work / "virus_all.fasta")
    if current_virus_sha != reuse_virus_sha:
        raise RuntimeError(
            "Cannot reuse decontamination outputs: virus_all.fasta differs "
            f"(current={current_virus_sha}, reuse={reuse_virus_sha})"
        )

    link_or_copy(reuse_work / "nonviral_vs_virus.tsv", blast_dir / "nonviral_vs_virus.tsv")
    link_or_copy(reuse_work / "nonviral_virus_intervals.tsv", blast_dir / "nonviral_virus_intervals.tsv")
    cleaned_nonviral = work_dir / "nonviral_cleaned.fasta"
    link_or_copy(reuse_work / "nonviral_cleaned.fasta", cleaned_nonviral)
    link_or_copy(reuse_metadata / "nonviral_decontamination.tsv", metadata_dir / "nonviral_decontamination.tsv")
    link_or_copy(
        reuse_metadata / "nonviral_decontamination_contigs.tsv",
        metadata_dir / "nonviral_decontamination_contigs.tsv",
    )
    with open(metadata_dir / "decontamination_reuse.tsv", "wt") as handle:
        handle.write("source_dir\tvirus_sha256\treused_files\n")
        handle.write(
            f"{reuse_dir}\t{current_virus_sha}\t"
            "nonviral_vs_virus.tsv,nonviral_virus_intervals.tsv,nonviral_cleaned.fasta,"
            "nonviral_decontamination.tsv,nonviral_decontamination_contigs.tsv\n"
        )
    return cleaned_nonviral


def parse_blast_coverage(
    blast_tsv: Path,
    removed_tsv: Path,
    pident_min: float,
    coverage_threshold: float,
    reason: str,
) -> dict[str, float]:
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    qlens: dict[str, int] = {}
    if blast_tsv.exists():
        with open(blast_tsv, "rt") as handle:
            for line in handle:
                if not line.strip():
                    continue
                qseqid, qstart, qend, pident, _length, qlen, evalue, _bitscore = line.rstrip("\n").split("\t")
                if float(pident) < pident_min or float(evalue) > 1e-10:
                    continue
                qlens[qseqid] = int(qlen)
                raw[qseqid].append((int(qstart), int(qend)))
    removed: dict[str, float] = {}
    rows: list[tuple[object, ...]] = []
    for qseqid, intervals in sorted(raw.items()):
        qlen = qlens.get(qseqid, 0)
        merged = merge_intervals(intervals, qlen)
        covered = interval_coverage(merged)
        coverage = covered / qlen if qlen else 0.0
        if coverage > coverage_threshold:
            removed[qseqid] = coverage
            rows.append((qseqid, qlen, covered, f"{coverage:.8f}", len(merged), reason))
    write_tsv(
        removed_tsv,
        ["fragment_id", "query_length", "covered_bp", "coverage", "merged_interval_count", "reason"],
        rows,
    )
    return removed


def update_removed_by_blast(conn: sqlite3.Connection, removed: dict[str, float], split: str, reason: str) -> None:
    if not removed:
        return
    with conn:
        conn.executemany(
            """
            UPDATE fragments
            SET selected=0,
                filter_reason=?,
                removed_by_train_blast=1,
                train_coverage=?
            WHERE fragment_id=? AND split=?
            """,
            [(reason, coverage, fragment_id, split) for fragment_id, coverage in removed.items()],
        )


def write_split_balance_json(
    conn: sqlite3.Connection,
    qc_dir: Path,
    dev_removed: dict[str, float],
    test_removed: dict[str, float],
    dev_fraction: float,
    test_fraction: float,
) -> None:
    splits: dict[str, dict[str, object]] = {}
    total_fragments = 0
    total_bp = 0
    for split in SPLITS:
        rows = conn.execute(
            """
            SELECT supergroup, COUNT(*), COALESCE(SUM(length), 0)
            FROM fragments
            WHERE selected=1 AND split=?
            GROUP BY supergroup
            """,
            (split,),
        ).fetchall()
        by_super = {str(super): {"fragments": int(count), "bp": int(bp)} for super, count, bp in rows}
        split_fragments = sum(item["fragments"] for item in by_super.values())
        split_bp = sum(item["bp"] for item in by_super.values())
        total_fragments += split_fragments
        total_bp += split_bp
        length_bins = {
            str(bin_start): {
                "virus": int(virus_count or 0),
                "nonvirus": int(nonvirus_count or 0),
            }
            for bin_start, virus_count, nonvirus_count in conn.execute(
                """
                SELECT length_bin,
                       SUM(CASE WHEN supergroup='virus' THEN 1 ELSE 0 END),
                       SUM(CASE WHEN supergroup='nonvirus' THEN 1 ELSE 0 END)
                FROM fragments
                WHERE selected=1 AND split=?
                GROUP BY length_bin
                ORDER BY length_bin
                """,
                (split,),
            )
        }
        splits[split] = {
            "supergroups": by_super,
            "fragments": split_fragments,
            "bp": split_bp,
            "length_bins": length_bins,
        }
    for split in SPLITS:
        splits[split]["fragment_fraction"] = (splits[split]["fragments"] / total_fragments) if total_fragments else 0.0
        splits[split]["bp_fraction"] = (splits[split]["bp"] / total_bp) if total_bp else 0.0
    payload = {
        "requested_dev_fraction": dev_fraction,
        "requested_test_fraction": test_fraction,
        "actual_total_fragments": total_fragments,
        "actual_total_bp": total_bp,
        "splits": splits,
        "blast_removed": {
            "dev_vs_train": len(dev_removed),
            "test_vs_train_dev": len(test_removed),
        },
    }
    with open(qc_dir / "split_balance.json", "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_verification_v4(
    conn: sqlite3.Connection,
    input_root: Path,
    metadata_dir: Path,
    qc_dir: Path,
    min_len: int,
    max_len: int,
    dev_removed: dict[str, float],
    test_removed: dict[str, float],
    coverage_threshold: float,
) -> None:
    checks: list[tuple[str, str, str]] = []
    virus_realms = {row["realm_group"] for row in read_tsv(metadata_dir / "virus_accession_realm.tsv")}
    checks.append(("viral_realms_in_target_set", "pass" if virus_realms <= set(REALM_GROUPS) else "fail", ",".join(sorted(virus_realms))))

    nonviral_rows = read_tsv(metadata_dir / "nonviral_genomes.tsv")
    bacteria_counts: dict[str, int] = defaultdict(int)
    insect_counts: dict[str, int] = defaultdict(int)
    for row in nonviral_rows:
        if row["selected"] != "1":
            continue
        if row["source"] == "bacteria":
            bacteria_counts[row["genus"]] += 1
        if row["source"] == "insect":
            insect_counts[row["genus"]] += 1
    bacteria_over = {genus: count for genus, count in bacteria_counts.items() if count > GENUS_CAPS["bacteria"]}
    insect_over = {genus: count for genus, count in insect_counts.items() if count > GENUS_CAPS["insect"]}
    checks.append(("bacteria_selected_per_genus_le_2", "pass" if not bacteria_over else "fail", str(bacteria_over)))
    checks.append(("insect_selected_per_genus_le_1", "pass" if not insect_over else "fail", str(insect_over)))

    split_by_genome: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(metadata_dir / "genome_split.tsv"):
        split_by_genome[row["genome_id"]].add(row["split"])
    cross_split = [genome for genome, splits in split_by_genome.items() if len(splits) > 1]
    checks.append(("genome_not_cross_split", "pass" if not cross_split else "fail", str(cross_split[:10])))

    min_observed, max_observed = conn.execute(
        "SELECT MIN(length), MAX(length) FROM fragments WHERE selected=1"
    ).fetchone()
    length_pass = (min_observed is None) or (int(min_observed) >= min_len and int(max_observed) <= max_len)
    checks.append(("final_fragment_lengths_in_range", "pass" if length_pass else "fail", f"{min_observed}-{max_observed}"))

    for split in SPLITS:
        groups = {
            str(supergroup): (int(count), int(total_bp or 0))
            for supergroup, count, total_bp in conn.execute(
                """
                SELECT supergroup, COUNT(*), SUM(length)
                FROM fragments
                WHERE selected=1 AND split=?
                GROUP BY supergroup
                """,
                (split,),
            )
        }
        missing = [name for name in ("virus", "nonvirus") if groups.get(name, (0, 0))[0] == 0]
        checks.append((f"{split}_has_both_binary_classes", "pass" if not missing else "fail", ",".join(missing)))
        virus_count, virus_bp = groups.get("virus", (0, 0))
        nonvirus_count, nonvirus_bp = groups.get("nonvirus", (0, 0))
        count_ratio = min(virus_count, nonvirus_count) / max(virus_count, nonvirus_count) if max(virus_count, nonvirus_count) else 1.0
        bp_ratio = min(virus_bp, nonvirus_bp) / max(virus_bp, nonvirus_bp) if max(virus_bp, nonvirus_bp) else 1.0
        checks.append((f"{split}_binary_fragment_count_ratio_ge_0.90", "pass" if count_ratio >= 0.90 else "warn", f"{count_ratio:.6f}"))
        checks.append((f"{split}_binary_bp_ratio_ge_0.90", "pass" if bp_ratio >= 0.90 else "warn", f"{bp_ratio:.6f}"))

    still_selected_dev = conn.execute(
        "SELECT COUNT(*) FROM fragments WHERE selected=1 AND filter_reason='dev_train_blast_coverage'"
    ).fetchone()[0]
    still_selected_test = conn.execute(
        "SELECT COUNT(*) FROM fragments WHERE selected=1 AND filter_reason='test_train_dev_blast_coverage'"
    ).fetchone()[0]
    checks.append(("removed_dev_fragments_not_selected", "pass" if still_selected_dev == 0 else "fail", str(still_selected_dev)))
    checks.append(("removed_test_fragments_not_selected", "pass" if still_selected_test == 0 else "fail", str(still_selected_test)))
    checks.append(
        (
            "dev_train_coverage_filter_applied",
            "pass",
            f"removed={len(dev_removed)} max_removed_coverage={max(dev_removed.values()) if dev_removed else 0:.6g} threshold={coverage_threshold}",
        )
    )
    checks.append(
        (
            "test_train_dev_coverage_filter_applied",
            "pass",
            f"removed={len(test_removed)} max_removed_coverage={max(test_removed.values()) if test_removed else 0:.6g} threshold={coverage_threshold}",
        )
    )
    insect_txt = input_root / "insect/insect.txt"
    checks.append(("insect_manifest_present", "pass" if insect_txt.exists() else "fail", str(insect_txt)))
    write_tsv(qc_dir / "verification.tsv", ["check", "status", "detail"], checks)


def write_processing_summary_v4(
    output_dir: Path,
    input_root: Path,
    seed: int,
    threads: int,
    dev_fraction: float,
    test_fraction: float,
    min_len: int,
    max_len: int,
    bin_width: int,
    coverage_threshold: float,
) -> None:
    summary_rows = read_tsv(output_dir / "qc/summary.tsv")
    verification_rows = read_tsv(output_dir / "qc/verification.tsv")
    split_rows = read_tsv(output_dir / "metadata/genome_split.tsv")
    failed = [row for row in verification_rows if row["status"] == "fail"]
    warnings = [row for row in verification_rows if row["status"] == "warn"]

    split_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in split_rows:
        split_counts[(row["label"], row["split"])] += 1

    lines = [
        "# Realm-Rank v4 Processing Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Input root: `{input_root}`",
        f"Output directory: `{output_dir}`",
        "",
        "## Parameters",
        "",
        f"- Seed: `{seed}`",
        f"- Threads: `{threads}`",
        f"- Dev fraction: `{dev_fraction}`",
        f"- Test fraction: `{test_fraction}`",
        f"- Fragment length: `{min_len}-{max_len}`",
        f"- Length bin width: `{bin_width}`",
        f"- Dev-vs-train and test-vs-train+dev removal threshold: `>{coverage_threshold}`",
        f"- Nonviral sources: `{', '.join(NONVIRAL_SOURCES)}`",
        f"- Genus caps: bacteria <= {GENUS_CAPS['bacteria']}; insect <= {GENUS_CAPS['insect']}",
        "",
        "## Genome Split",
        "",
    ]
    for label in sorted({label for label, _split in split_counts}):
        lines.append(
            f"- {label}: train={split_counts.get((label, 'train'), 0)} "
            f"dev={split_counts.get((label, 'dev'), 0)} test={split_counts.get((label, 'test'), 0)}"
        )
    lines.extend(["", "## Final Fragments", ""])
    for row in summary_rows:
        lines.append(
            f"- {row['split']} {row['label']} ({row['source']}): "
            f"{row['fragment_count']} fragments, {row['total_bp']} bp, "
            f"length {row['min_length']}-{row['max_length']}"
        )
    lines.extend(["", "## Verification", ""])
    if failed:
        lines.append(f"- Failed checks: {len(failed)}")
        for row in failed:
            lines.append(f"- {row['check']}: {row['detail']}")
    else:
        lines.append("- No failed checks in `qc/verification.tsv`.")
    if warnings:
        lines.append(f"- Warnings: {len(warnings)}")
        for row in warnings:
            lines.append(f"- {row['check']}: {row['detail']}")
    lines.extend(
        [
            "",
            "## Key Outputs",
            "",
            "- `train.fasta.gz`, `dev.fasta.gz`, and `test.fasta.gz`: final compressed FASTA files.",
            "- `metadata/fragments.tsv`: selected fragment coordinates.",
            "- `metadata/dev_removed_by_train_blast.tsv`: dev fragments removed by train BLAST coverage.",
            "- `metadata/test_removed_by_train_dev_blast.tsv`: test fragments removed by train+dev BLAST coverage.",
            "- `qc/split_balance.json`: requested and actual split proportions plus binary balance details.",
            "- `blast/*.tsv`: raw BLAST tables used for decontamination and leakage checks.",
        ]
    )
    (output_dir / "processing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=RAW_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_ROOT / "realm_rank_v4")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=0, help="Default: 85%% of available CPUs")
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--min-fragment-length", type=int, default=300)
    parser.add_argument("--max-fragment-length", type=int, default=2000)
    parser.add_argument("--length-bin-width", type=int, default=100)
    parser.add_argument("--length-balance-threshold", type=float, default=0.05)
    parser.add_argument("--decontam-pident", type=float, default=90.0)
    parser.add_argument("--decontam-hsp-qcov", type=float, default=0.80)
    parser.add_argument("--leakage-pident", type=float, default=90.0)
    parser.add_argument("--leakage-coverage", type=float, default=0.50)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--ncbi-email", default=os.environ.get("NCBI_EMAIL", "realm-rank-builder@example.com"))
    parser.add_argument("--ncbi-api-key", default=os.environ.get("NCBI_API_KEY", ""))
    parser.add_argument("--ncbi-batch-size", type=int, default=200)
    parser.add_argument("--ncbi-delay", type=float, default=0.34)
    parser.add_argument("--skip-ncbi-taxonomy", action="store_true", help="Smoke-test helper: mark viral records as SmallRealm without Entrez calls")
    parser.add_argument(
        "--reuse-decontam-from",
        type=Path,
        help="Reuse split-independent v3/v4 nonviral-vs-virus BLAST decontamination outputs from this Realm-Rank directory.",
    )
    parser.add_argument("--blastn", default=None)
    parser.add_argument("--makeblastdb", default=None)
    parser.add_argument("--no-validate-gzip", action="store_true", help="Skip gzip integrity preflight")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the output directory")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.dev_fraction < 1.0):
        raise ValueError("--dev-fraction must be in [0, 1)")
    if not (0.0 <= args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be in [0, 1)")
    if args.dev_fraction + args.test_fraction >= 1.0:
        raise ValueError("--dev-fraction + --test-fraction must be < 1")
    if args.min_fragment_length <= 0 or args.max_fragment_length < args.min_fragment_length:
        raise ValueError("Invalid fragment length bounds")
    if not (0.0 <= args.length_balance_threshold <= 1.0):
        raise ValueError("--length-balance-threshold must be in [0, 1]")
    if not (0.0 <= args.leakage_coverage <= 1.0):
        raise ValueError("--leakage-coverage must be in [0, 1]")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    validate_args(args)
    start = time.time()
    threads = resolve_threads(args.threads)
    blastn = resolve_tool("blastn", args.blastn)
    makeblastdb = resolve_tool("makeblastdb", args.makeblastdb)
    metadata_dir, qc_dir, work_dir, blast_dir = ensure_dirs(args.output_dir, args.force)

    with open(qc_dir / "build_config.json", "w") as handle:
        json.dump({key: str(value) for key, value in vars(args).items()}, handle, indent=2, sort_keys=True)
        handle.write("\n")

    log(f"using {threads} BLAST threads")
    files = discover_files(args.input_root)
    if not args.no_validate_gzip:
        files = validate_input_files(files, metadata_dir, threads)

    log("building viral FASTA and accession list")
    virus_rows, virus_fasta, virus_contigs = build_virus_fasta(files["virus"], work_dir)
    if args.skip_ncbi_taxonomy:
        log("skipping NCBI viral taxonomy; marking viral records SmallRealm")
        write_skipped_viral_taxonomy(virus_rows, metadata_dir)
    else:
        assign_viral_realms(
            virus_rows,
            metadata_dir,
            work_dir,
            args.ncbi_email,
            args.ncbi_api_key,
            args.ncbi_batch_size,
            args.ncbi_delay,
        )

    log("discovering nonviral genomes and applying bacteria/insect genus caps")
    nonviral_rows = discover_nonviral_genomes(files, args.seed)
    split_genomes_three_way(virus_rows, nonviral_rows, args.seed, args.dev_fraction, args.test_fraction)
    write_genome_split(metadata_dir, virus_rows, nonviral_rows)

    if args.reuse_decontam_from:
        log(f"reusing selected nonviral FASTA and contigs from {args.reuse_decontam_from}")
        nonviral_fasta, nonviral_contigs = reuse_selected_nonviral_outputs(
            args.reuse_decontam_from,
            nonviral_rows,
            work_dir,
            metadata_dir,
        )
    else:
        log("writing selected nonviral FASTA")
        nonviral_fasta, nonviral_contigs = write_selected_nonviral_fasta(files, nonviral_rows, work_dir, metadata_dir)

    if args.reuse_decontam_from:
        log(f"reusing split-independent decontamination outputs from {args.reuse_decontam_from}")
        cleaned_nonviral = reuse_decontamination_outputs(
            args.reuse_decontam_from,
            virus_fasta,
            work_dir,
            metadata_dir,
            blast_dir,
        )
    else:
        log("building viral BLAST database")
        virus_db = work_dir / "blast/virus_db/virus"
        make_blast_db(makeblastdb, virus_fasta, virus_db)
        log("running nonviral-vs-virus BLAST for contamination intervals")
        nonviral_vs_virus = blast_dir / "nonviral_vs_virus.tsv"
        run_blastn(blastn, nonviral_fasta, virus_db, nonviral_vs_virus, threads)
        log("merging viral-like intervals and cleaning nonviral sequences")
        intervals_tsv = blast_dir / "nonviral_virus_intervals.tsv"
        intervals = build_decontam_intervals(nonviral_vs_virus, intervals_tsv, args.decontam_pident, args.decontam_hsp_qcov)
        cleaned_nonviral = decontaminate_nonviral_fasta(nonviral_fasta, nonviral_contigs, intervals, work_dir, metadata_dir)

    log("generating fragment metadata")
    genome_split = load_genome_split(metadata_dir)
    virus_info = load_contig_info(virus_contigs, genome_split, viral=True)
    nonviral_info = load_contig_info(nonviral_contigs, genome_split, viral=False)
    fragment_db = work_dir / "fragments.sqlite"
    conn = init_fragment_db(fragment_db)
    virus_fragment_count = insert_fragments_for_fasta(
        conn,
        virus_fasta,
        virus_info,
        args.min_fragment_length,
        args.max_fragment_length,
        args.length_bin_width,
        args.seed,
    )
    nonvirus_fragment_count = insert_fragments_for_fasta(
        conn,
        cleaned_nonviral,
        nonviral_info,
        args.min_fragment_length,
        args.max_fragment_length,
        args.length_bin_width,
        args.seed,
    )
    create_fragment_indexes(conn)
    log(f"generated {virus_fragment_count:,} viral and {nonvirus_fragment_count:,} nonviral fragments")

    log("downsampling large bacteria/insect pools against viral bp targets")
    downsample_large_sources_to_virus_bp_v4(conn, args.seed, qc_dir)
    log("checking and balancing nonviral length-bin distributions")
    balance_nonvirus_length_distribution_v4(conn, args.seed, qc_dir, args.length_balance_threshold)
    log("performing initial binary bp/bin balance across train/dev/test")
    downsample_binary_to_common_bp_by_length_bin(conn, args.seed, qc_dir, SPLITS, "initial_binary_bp_balance")

    log("writing selected prefilter train/dev/test FASTA")
    prefilters = write_selected_fragment_fastas_v4(
        conn,
        [(virus_fasta, virus_info), (cleaned_nonviral, nonviral_info)],
        work_dir,
        args.min_fragment_length,
        args.max_fragment_length,
        args.seed,
    )

    log("building final train-fragment BLAST database")
    selected = selected_fragment_ids(conn)
    train_for_db = work_dir / "blast/train.final.fasta"
    write_plain_filtered_fasta(prefilters["train"], train_for_db, selected["train"])
    train_db = work_dir / "blast/train_db/train"
    make_blast_db(makeblastdb, train_for_db, train_db)

    log("running dev-vs-train BLAST leakage check")
    dev_vs_train = blast_dir / "dev_vs_train.tsv"
    run_blastn(blastn, prefilters["dev"], train_db, dev_vs_train, threads)
    dev_removed = parse_blast_coverage(
        dev_vs_train,
        metadata_dir / "dev_removed_by_train_blast.tsv",
        args.leakage_pident,
        args.leakage_coverage,
        "dev_train_blast_coverage",
    )
    update_removed_by_blast(conn, dev_removed, "dev", "dev_train_blast_coverage")
    log(f"removed dev fragments by train coverage > {args.leakage_coverage}: {len(dev_removed):,}")
    downsample_binary_to_common_bp_by_length_bin(conn, args.seed, qc_dir, ("dev",), "dev_post_blast_binary_bp_balance")

    log("building final train+dev BLAST database")
    selected = selected_fragment_ids(conn)
    dev_for_db = work_dir / "blast/dev.final.fasta"
    train_dev_for_db = work_dir / "blast/train_dev.final.fasta"
    write_plain_filtered_fasta(prefilters["dev"], dev_for_db, selected["dev"])
    concat_plain_fastas((train_for_db, dev_for_db), train_dev_for_db)
    train_dev_db = work_dir / "blast/train_dev_db/train_dev"
    make_blast_db(makeblastdb, train_dev_for_db, train_dev_db)

    log("running test-vs-train+dev BLAST leakage check")
    test_vs_train_dev = blast_dir / "test_vs_train_dev.tsv"
    run_blastn(blastn, prefilters["test"], train_dev_db, test_vs_train_dev, threads)
    test_removed = parse_blast_coverage(
        test_vs_train_dev,
        metadata_dir / "test_removed_by_train_dev_blast.tsv",
        args.leakage_pident,
        args.leakage_coverage,
        "test_train_dev_blast_coverage",
    )
    update_removed_by_blast(conn, test_removed, "test", "test_train_dev_blast_coverage")
    log(f"removed test fragments by train+dev coverage > {args.leakage_coverage}: {len(test_removed):,}")
    downsample_binary_to_common_bp_by_length_bin(conn, args.seed, qc_dir, ("test",), "test_post_blast_binary_bp_balance")

    log("writing final compressed FASTA and metadata")
    selected = selected_fragment_ids(conn)
    for split in SPLITS:
        write_gzip_filtered_fasta(
            prefilters[split],
            args.output_dir / f"{split}.fasta.gz",
            selected[split],
            threads=threads,
            compression_level=args.compression_level,
        )
    dump_final_fragments(conn, metadata_dir)
    write_qc(conn, qc_dir, args.min_fragment_length, args.max_fragment_length, args.length_bin_width)
    write_split_balance_json(conn, qc_dir, dev_removed, test_removed, args.dev_fraction, args.test_fraction)
    write_verification_v4(
        conn,
        args.input_root,
        metadata_dir,
        qc_dir,
        args.min_fragment_length,
        args.max_fragment_length,
        dev_removed,
        test_removed,
        args.leakage_coverage,
    )
    write_processing_summary_v4(
        args.output_dir,
        args.input_root,
        args.seed,
        threads,
        args.dev_fraction,
        args.test_fraction,
        args.min_fragment_length,
        args.max_fragment_length,
        args.length_bin_width,
        args.leakage_coverage,
    )
    conn.close()
    log(f"done: {args.output_dir} elapsed={time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
