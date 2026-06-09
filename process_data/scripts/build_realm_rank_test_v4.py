#!/usr/bin/env python3
"""Build fixed-length Realm-Rank benchmark test v4 files.

This script keeps the existing Realm-Rank v4 train/dev/test outputs intact and
creates a separate fixed-length benchmark suite under
``processed_data/realm_rank_test_v4``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlencode
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from realm_rank_dataset_common import (  # noqa: E402
    build_decontam_intervals,
    clean_sequence,
    iter_fasta,
    make_blast_db,
    merge_intervals,
    interval_coverage,
    read_tsv,
    resolve_tool,
    run_command,
    safe_id,
    stable_hash_hex,
    stable_int,
    write_fasta,
    write_tsv,
)


DEFAULT_LENGTHS = (500, 1000, 2000, 10000, 20000)
DEFAULT_SEED = 1729
DEFAULT_THREADS = 54
DEFAULT_BLASTN = os.environ.get("BLASTN_BIN", "blastn")
DEFAULT_MAKEBLASTDB = os.environ.get("MAKEBLASTDB_BIN", "makeblastdb")
FASTA_WIDTH = 80
GAMIL_ROOT = Path(os.environ.get("GAMIL_ROOT", SCRIPT_DIR.parents[1])).resolve()
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", GAMIL_ROOT / "raw_data" / "local_sources")).resolve()
PROCESSED_DATA_ROOT = Path(os.environ.get("PROCESSED_DATA_ROOT", GAMIL_ROOT / "processed_data")).resolve()
BENCHMARKS = {
    "bench-pro": ("bacteria", "archaea", "plasmid"),
    "bench-euk": ("fungi", "protozoa", "insect", "human", "bat"),
}
NATIVE_EUK_SOURCES = ("fungi", "protozoa", "insect")
REFERENCE_EUK_SOURCES = ("fungi", "protozoa", "insect", "human", "bat")
REFERENCE_SPECS = (
    ("human", "GCF_000001405.40", "Homo sapiens", "human_GRCh38.p14"),
    ("bat", "GCF_004115265.2", "Rhinolophus ferrumequinum", "bat_Rhinolophus_ferrumequinum"),
)


@dataclass(frozen=True)
class ReferenceRecord:
    source: str
    accession: str
    organism: str
    output_subdir: str
    assembly_name: str
    ftp_path: str
    fasta_path: Path
    assembly_report_path: Path
    fasta_sha256: str
    assembly_report_sha256: str
    fasta_bytes: int
    assembly_report_bytes: int
    gzip_valid: bool


@dataclass(frozen=True)
class SelectionRow:
    benchmark: str
    output_id: str
    candidate_id: str
    length: int
    class_label: str
    label: str
    realm: str
    source: str
    supergroup: str
    genome_id: str
    contig_id: str
    start: int
    end: int


def log(message: str) -> None:
    print(f"[realm-rank-test-v4] {message}", flush=True)


def ensure_dirs(output_dir: Path, force: bool) -> tuple[Path, Path, Path]:
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    metadata_dir = output_dir / "metadata"
    qc_dir = output_dir / "qc"
    work_dir = output_dir / "work"
    for path in (metadata_dir, qc_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)
    return metadata_dir, qc_dir, work_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return True
    except Exception:  # noqa: BLE001 - validation should return a boolean
        return False


def download_url(urls: str | list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if isinstance(urls, str):
        urls = [urls]
    last_error: subprocess.CalledProcessError | None = None
    for url in urls:
        cmd = [
            "curl",
            "--silent",
            "--show-error",
            "-L",
            "--fail",
            "--retry",
            "8",
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--speed-limit",
            "200000",
            "--speed-time",
            "60",
            "-C",
            "-",
            "-o",
            str(tmp_path),
            url,
        ]
        try:
            run_command(cmd)
            tmp_path.replace(output_path)
            return
        except subprocess.CalledProcessError as exc:
            last_error = exc
            log(f"download failed, trying fallback if available: {url}")
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No download URLs provided for {output_path}")


def reference_download_base(ftp_path: str, organism: str) -> list[str]:
    https_path = ftp_path.replace("ftp://", "https://", 1).rstrip("/")
    basename = https_path.split("/")[-1]
    organism_dir = organism.replace(" ", "_")
    latest = (
        "https://ftp.ncbi.nlm.nih.gov/genomes/refseq/vertebrate_mammalian/"
        f"{organism_dir}/latest_assembly_versions/{basename}"
    )
    return [latest, https_path]


def ncbi_assembly_summary(accession: str) -> tuple[str, str]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urlencode(
        {
            "db": "assembly",
            "term": f"{accession}[Assembly Accession]",
            "retmode": "json",
        }
    )
    with urlopen(search_url, timeout=60) as handle:
        search = json.load(handle)
    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise RuntimeError(f"NCBI assembly accession not found: {accession}")

    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urlencode(
        {
            "db": "assembly",
            "id": ",".join(ids),
            "retmode": "json",
        }
    )
    with urlopen(summary_url, timeout=60) as handle:
        summary = json.load(handle)
    uid = summary["result"]["uids"][0]
    record = summary["result"][uid]
    assembly_name = record.get("assemblyname") or accession
    ftp_path = record.get("ftppath_refseq") or record.get("ftppath_genbank")
    if not ftp_path:
        raise RuntimeError(f"NCBI summary for {accession} has no FTP path")
    return assembly_name, ftp_path


def download_references(reference_dir: Path, metadata_dir: Path) -> list[ReferenceRecord]:
    records: list[ReferenceRecord] = []
    rows: list[tuple[object, ...]] = []
    for source, accession, organism, subdir in REFERENCE_SPECS:
        assembly_name, ftp_path = ncbi_assembly_summary(accession)
        basename = ftp_path.rstrip("/").split("/")[-1]
        download_bases = reference_download_base(ftp_path, organism)
        dest_dir = reference_dir / subdir
        fasta_path = dest_dir / f"{basename}_genomic.fna.gz"
        report_path = dest_dir / f"{basename}_assembly_report.txt"

        if not fasta_path.exists():
            log(f"downloading {source} FASTA: {accession} {assembly_name}")
            download_url([f"{base}/{basename}_genomic.fna.gz" for base in download_bases], fasta_path)
        if not report_path.exists():
            log(f"downloading {source} assembly report: {accession} {assembly_name}")
            download_url([f"{base}/{basename}_assembly_report.txt" for base in download_bases], report_path)

        gzip_valid = validate_gzip(fasta_path)
        if not gzip_valid:
            raise RuntimeError(f"Downloaded FASTA is not a valid gzip file: {fasta_path}")

        fasta_sha = sha256_file(fasta_path)
        report_sha = sha256_file(report_path)
        record = ReferenceRecord(
            source=source,
            accession=accession,
            organism=organism,
            output_subdir=subdir,
            assembly_name=assembly_name,
            ftp_path=ftp_path,
            fasta_path=fasta_path,
            assembly_report_path=report_path,
            fasta_sha256=fasta_sha,
            assembly_report_sha256=report_sha,
            fasta_bytes=fasta_path.stat().st_size,
            assembly_report_bytes=report_path.stat().st_size,
            gzip_valid=gzip_valid,
        )
        records.append(record)
        rows.append(
            (
                source,
                accession,
                organism,
                assembly_name,
                ftp_path,
                fasta_path,
                fasta_path.stat().st_size,
                fasta_sha,
                "pass" if gzip_valid else "fail",
                report_path,
                report_path.stat().st_size,
                report_sha,
            )
        )

    write_tsv(
        metadata_dir / "downloaded_references.tsv",
        [
            "source",
            "accession",
            "organism",
            "assembly_name",
            "ftp_path",
            "genomic_fasta",
            "genomic_fasta_bytes",
            "genomic_fasta_sha256",
            "genomic_fasta_gzip_check",
            "assembly_report",
            "assembly_report_bytes",
            "assembly_report_sha256",
        ],
        rows,
    )
    return records


def load_genome_split(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in read_tsv(path):
        result[row["genome_id"]] = row
    return result


def load_native_contig_info(realm_rank_dir: Path) -> dict[str, dict[str, str]]:
    genome_split = load_genome_split(realm_rank_dir / "metadata/genome_split.tsv")
    info: dict[str, dict[str, str]] = {}

    for row in read_tsv(realm_rank_dir / "work/virus_contigs.tsv"):
        split_row = genome_split.get(row["genome_id"])
        if not split_row or split_row["split"] != "test":
            continue
        label = split_row["label"]
        info[row["contig_id"]] = {
            "class_label": "positive",
            "label": label,
            "realm": label,
            "source": "virus",
            "supergroup": "virus",
            "genome_id": row["genome_id"],
            "contig_id": row["contig_id"],
        }

    for row in read_tsv(realm_rank_dir / "work/nonviral_contigs.tsv"):
        split_row = genome_split.get(row["genome_id"])
        if not split_row or split_row["split"] != "test":
            continue
        source = split_row["source"]
        info[row["contig_id"]] = {
            "class_label": "negative",
            "label": source,
            "realm": "",
            "source": source,
            "supergroup": "nonvirus",
            "genome_id": row["genome_id"],
            "contig_id": row["contig_id"],
        }

    return info


def write_reference_raw_fasta(records: list[ReferenceRecord], work_dir: Path) -> tuple[Path, Path]:
    raw_fasta = work_dir / "reference_raw.fasta"
    contigs_tsv = work_dir / "reference_contigs.tsv"
    if raw_fasta.exists() and contigs_tsv.exists():
        return raw_fasta, contigs_tsv

    seen: Counter[str] = Counter()
    with open(raw_fasta, "wt") as fasta, open(contigs_tsv, "wt") as contigs:
        contigs.write(
            "contig_id\tgenome_id\tsource\tpath\toriginal_id\tdescription\toriginal_length\t"
            "accession\tassembly_name\torganism\n"
        )
        for record in records:
            genome_id = f"{record.source}__{record.accession}"
            for original_id, header, seq in iter_fasta(record.fasta_path):
                base = f"{genome_id}__{safe_id(original_id)}"
                count = seen[base]
                seen[base] += 1
                contig_id = base if count == 0 else f"{base}__dup{count}"
                description = header[len(original_id) :].strip()
                write_fasta(fasta, contig_id, seq)
                contigs.write(
                    f"{contig_id}\t{genome_id}\t{record.source}\t{record.fasta_path}\t"
                    f"{original_id}\t{description}\t{len(seq)}\t{record.accession}\t"
                    f"{record.assembly_name}\t{record.organism}\n"
                )
    return raw_fasta, contigs_tsv


def run_blastn_decontam(blastn: str, query: Path, db_prefix: Path, out_tsv: Path, threads: int) -> None:
    outfmt = "6 qseqid qstart qend pident length qlen evalue bitscore"
    cmd = [
        blastn,
        "-task",
        "megablast",
        "-query",
        str(query),
        "-db",
        str(db_prefix),
        "-out",
        str(out_tsv),
        "-outfmt",
        outfmt,
        "-evalue",
        "1e-10",
        "-num_threads",
        str(threads),
    ]
    run_command(cmd)


def decontaminate_reference_fasta(
    raw_fasta: Path,
    contigs_tsv: Path,
    intervals: dict[str, list[tuple[int, int]]],
    metadata_dir: Path,
    work_dir: Path,
) -> Path:
    cleaned_fasta = work_dir / "reference_cleaned.fasta"
    if cleaned_fasta.exists():
        return cleaned_fasta

    contig_meta = {row["contig_id"]: row for row in read_tsv(contigs_tsv)}
    genome_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {
            "source": "",
            "contig_count": 0,
            "original_bp": 0,
            "deleted_bp": 0,
            "cleaned_bp": 0,
        }
    )
    contig_rows: list[tuple[object, ...]] = []
    with open(cleaned_fasta, "wt") as out:
        for contig_id, _header, seq in iter_fasta(raw_fasta):
            meta = contig_meta[contig_id]
            cleaned, deleted = clean_sequence(seq, intervals.get(contig_id, []))
            if cleaned:
                write_fasta(out, contig_id, cleaned)
            stats = genome_stats[meta["genome_id"]]
            stats["source"] = meta["source"]
            stats["contig_count"] = int(stats["contig_count"]) + 1
            stats["original_bp"] = int(stats["original_bp"]) + len(seq)
            stats["deleted_bp"] = int(stats["deleted_bp"]) + deleted
            stats["cleaned_bp"] = int(stats["cleaned_bp"]) + len(cleaned)
            contig_rows.append(
                (
                    contig_id,
                    meta["genome_id"],
                    meta["source"],
                    len(seq),
                    deleted,
                    len(cleaned),
                    deleted / len(seq) if seq else 0,
                )
            )

    write_tsv(
        metadata_dir / "reference_decontamination_contigs.tsv",
        ["contig_id", "genome_id", "source", "original_bp", "deleted_bp", "cleaned_bp", "deleted_fraction"],
        contig_rows,
    )
    genome_rows: list[tuple[object, ...]] = []
    for genome_id, stats in sorted(genome_stats.items()):
        original_bp = int(stats["original_bp"])
        deleted_bp = int(stats["deleted_bp"])
        cleaned_bp = int(stats["cleaned_bp"])
        genome_rows.append(
            (
                genome_id,
                stats["source"],
                stats["contig_count"],
                original_bp,
                deleted_bp,
                cleaned_bp,
                deleted_bp / original_bp if original_bp else 0,
            )
        )
    write_tsv(
        metadata_dir / "reference_decontamination.tsv",
        ["genome_id", "source", "contig_count", "original_bp", "deleted_bp", "cleaned_bp", "deleted_fraction"],
        genome_rows,
    )
    return cleaned_fasta


def prepare_references(
    records: list[ReferenceRecord],
    realm_rank_dir: Path,
    metadata_dir: Path,
    work_dir: Path,
    blastn: str,
    makeblastdb: str,
    threads: int,
) -> tuple[Path, Path]:
    raw_fasta, contigs_tsv = write_reference_raw_fasta(records, work_dir)
    blast_dir = work_dir / "blast"
    virus_db = blast_dir / "virus_db/virus"
    if not (virus_db.with_suffix(".nhr").exists() or (virus_db.parent / "virus.nal").exists()):
        log("building v4 virus BLAST database for reference decontamination")
        make_blast_db(makeblastdb, realm_rank_dir / "work/virus_all.fasta", virus_db)

    reference_vs_virus = work_dir / "reference_vs_virus.tsv"
    if not reference_vs_virus.exists():
        log("running human/bat decontamination BLAST against virus DB")
        run_blastn_decontam(blastn, raw_fasta, virus_db, reference_vs_virus, threads)

    intervals_tsv = work_dir / "reference_virus_intervals.tsv"
    if intervals_tsv.exists():
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in read_tsv(intervals_tsv):
            intervals[row["contig_id"]].append((int(row["start"]), int(row["end"])))
        merged = {contig: merge_intervals(vals) for contig, vals in intervals.items()}
    else:
        intervals = build_decontam_intervals(
            reference_vs_virus,
            intervals_tsv,
            pident_min=90.0,
            hsp_qcov_min=0.80,
        )
        merged = intervals

    cleaned_fasta = decontaminate_reference_fasta(raw_fasta, contigs_tsv, merged, metadata_dir, work_dir)
    return cleaned_fasta, contigs_tsv


def init_candidate_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        """
        CREATE TABLE candidates (
            candidate_id TEXT PRIMARY KEY,
            length INTEGER NOT NULL,
            class_label TEXT NOT NULL,
            label TEXT NOT NULL,
            realm TEXT NOT NULL,
            source TEXT NOT NULL,
            supergroup TEXT NOT NULL,
            genome_id TEXT NOT NULL,
            contig_id TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            removed_by_train_blast INTEGER NOT NULL DEFAULT 0,
            train_coverage REAL,
            matched_train_ids TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE selected (
            benchmark TEXT NOT NULL,
            output_id TEXT NOT NULL PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            length INTEGER NOT NULL,
            class_label TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        )
        """
    )
    conn.execute("CREATE INDEX idx_candidates_available ON candidates(length, class_label, source, removed_by_train_blast)")
    conn.execute("CREATE INDEX idx_candidates_class ON candidates(class_label, length, removed_by_train_blast)")
    conn.execute("CREATE INDEX idx_selected_candidate ON selected(candidate_id)")
    conn.commit()
    return conn


def open_candidate_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    return conn


def load_reference_contig_info(contigs_tsv: Path) -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    for row in read_tsv(contigs_tsv):
        source = row["source"]
        info[row["contig_id"]] = {
            "class_label": "negative",
            "label": source,
            "realm": "",
            "source": source,
            "supergroup": "nonvirus",
            "genome_id": row["genome_id"],
            "contig_id": row["contig_id"],
        }
    return info


def iter_fixed_windows(seq: str, length: int) -> Iterator[tuple[int, int, str]]:
    seq_len = len(seq)
    for start0 in range(0, seq_len - length + 1, length):
        subseq = seq[start0 : start0 + length]
        if "N" in subseq:
            continue
        yield start0 + 1, start0 + length, subseq


def insert_candidate_batch(conn: sqlite3.Connection, batch: list[tuple[object, ...]]) -> None:
    conn.executemany(
        """
        INSERT INTO candidates (
            candidate_id, length, class_label, label, realm, source, supergroup,
            genome_id, contig_id, start, end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        batch,
    )


def generate_candidates_for_fasta(
    conn: sqlite3.Connection,
    fasta_path: Path,
    contig_info: dict[str, dict[str, str]],
    lengths: tuple[int, ...],
    out_handle,
) -> int:
    batch: list[tuple[object, ...]] = []
    total = 0
    for contig_id, _header, seq in iter_fasta(fasta_path):
        info = contig_info.get(contig_id)
        if info is None:
            continue
        for length in lengths:
            for start, end, subseq in iter_fixed_windows(seq, length):
                candidate_id = "rrv4_" + stable_hash_hex(
                    "test_v4",
                    length,
                    info["class_label"],
                    info["label"],
                    info["source"],
                    info["genome_id"],
                    contig_id,
                    start,
                    end,
                    n=24,
                )
                batch.append(
                    (
                        candidate_id,
                        length,
                        info["class_label"],
                        info["label"],
                        info["realm"],
                        info["source"],
                        info["supergroup"],
                        info["genome_id"],
                        contig_id,
                        start,
                        end,
                    )
                )
                write_fasta(out_handle, candidate_id, subseq)
                total += 1
                if len(batch) >= 20000:
                    insert_candidate_batch(conn, batch)
                    conn.commit()
                    batch = []
        if total and total % 500000 == 0:
            log(f"candidate windows written so far: {total:,}")
    if batch:
        insert_candidate_batch(conn, batch)
        conn.commit()
    return total


def generate_candidate_pool(
    db_path: Path,
    candidate_fasta: Path,
    realm_rank_dir: Path,
    reference_cleaned_fasta: Path | None,
    reference_contigs_tsv: Path | None,
    lengths: tuple[int, ...],
) -> sqlite3.Connection:
    if db_path.exists() and candidate_fasta.exists():
        conn = open_candidate_db(db_path)
        count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        if count > 0:
            log(f"reusing candidate pool: {count:,} candidates")
            return conn
        conn.close()

    if candidate_fasta.exists():
        candidate_fasta.unlink()
    conn = init_candidate_db(db_path)
    native_info = load_native_contig_info(realm_rank_dir)
    reference_info = load_reference_contig_info(reference_contigs_tsv) if reference_contigs_tsv else {}

    log("generating fixed-length candidate windows from test virus genomes")
    with open(candidate_fasta, "wt") as out:
        virus_count = generate_candidates_for_fasta(
            conn,
            realm_rank_dir / "work/virus_all.fasta",
            native_info,
            lengths,
            out,
        )
        log(f"virus candidate windows: {virus_count:,}")

        log("generating fixed-length candidate windows from native nonviral test genomes")
        nonviral_count = generate_candidates_for_fasta(
            conn,
            realm_rank_dir / "work/nonviral_cleaned.fasta",
            native_info,
            lengths,
            out,
        )
        log(f"native nonviral candidate windows: {nonviral_count:,}")

        if reference_cleaned_fasta is not None:
            log("generating fixed-length candidate windows from human/bat references")
            reference_count = generate_candidates_for_fasta(conn, reference_cleaned_fasta, reference_info, lengths, out)
            log(f"human/bat candidate windows: {reference_count:,}")
        else:
            log("skipping human/bat reference candidate windows")

    total = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    log(f"candidate pool complete: {total:,} windows")
    return conn


def decompress_final_fasta(input_gz: Path, output_fasta: Path, threads: int) -> None:
    if output_fasta.exists():
        return
    tmp_path = output_fasta.with_suffix(output_fasta.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pigz = shutil.which("pigz")
    if pigz:
        cmd = [pigz, "-dc", "-p", str(max(1, threads)), str(input_gz)]
        log(f"decompressing final FASTA with pigz: {input_gz}")
        with open(tmp_path, "wb") as out:
            subprocess.run(cmd, stdout=out, check=True)
    else:
        log(f"decompressing final FASTA with python gzip: {input_gz}")
        with gzip.open(input_gz, "rb") as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    tmp_path.replace(output_fasta)


def db_files_exist(prefix: Path) -> bool:
    return prefix.with_suffix(".nhr").exists() or (prefix.parent / f"{prefix.name}.nal").exists()


def prepare_train_dev_blast_db(realm_rank_dir: Path, work_dir: Path, makeblastdb: str, threads: int) -> Path:
    train_fasta = work_dir / "train.final.fasta"
    dev_fasta = work_dir / "dev.final.fasta"
    train_dev_fasta = work_dir / "train_dev.final.fasta"
    decompress_final_fasta(realm_rank_dir / "train.fasta.gz", train_fasta, threads)
    decompress_final_fasta(realm_rank_dir / "dev.fasta.gz", dev_fasta, threads)
    if not train_dev_fasta.exists():
        tmp_path = train_dev_fasta.with_suffix(train_dev_fasta.suffix + ".tmp")
        with open(tmp_path, "wb") as dst:
            for path in (train_fasta, dev_fasta):
                with open(path, "rb") as src:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
        tmp_path.replace(train_dev_fasta)
    train_dev_db = work_dir / "blast/train_dev_db/train_dev"
    if not db_files_exist(train_dev_db):
        log("building BLAST DB from final train.fasta.gz + dev.fasta.gz")
        make_blast_db(makeblastdb, train_dev_fasta, train_dev_db)
    return train_dev_db


def run_blastn_candidates(blastn: str, query: Path, db_prefix: Path, out_tsv: Path, threads: int) -> None:
    outfmt = "6 qseqid sseqid qstart qend pident length qlen evalue bitscore"
    cmd = [
        blastn,
        "-task",
        "megablast",
        "-query",
        str(query),
        "-db",
        str(db_prefix),
        "-out",
        str(out_tsv),
        "-outfmt",
        outfmt,
        "-evalue",
        "1e-10",
        "-num_threads",
        str(threads),
    ]
    run_command(cmd)


def candidate_brief(conn: sqlite3.Connection, candidate_id: str) -> tuple[int, str, str]:
    row = conn.execute(
        "SELECT length, label, source FROM candidates WHERE candidate_id=?",
        (candidate_id,),
    ).fetchone()
    if row is None:
        return 0, "", ""
    return int(row[0]), str(row[1]), str(row[2])


def summarize_subjects(subjects: set[str], max_ids: int) -> str:
    ordered = sorted(subjects)
    if len(ordered) <= max_ids:
        return ",".join(ordered)
    shown = ordered[:max_ids]
    shown.append(f"...(+{len(ordered) - max_ids}_more)")
    return ",".join(shown)


def parse_train_blast_streaming(
    conn: sqlite3.Connection,
    blast_tsv: Path,
    removed_tsv: Path,
    pident_min: float,
    coverage_threshold: float,
    max_matched_train_ids: int,
) -> int:
    update_batch: list[tuple[object, ...]] = []
    removed_count = 0

    def finalize(
        qid: str | None,
        qlen: int,
        intervals: list[tuple[int, int]],
        subjects: set[str],
        removed_handle,
    ) -> None:
        nonlocal removed_count
        if qid is None or not intervals or qlen <= 0:
            return
        merged = merge_intervals(intervals, qlen)
        coverage = interval_coverage(merged) / qlen
        if coverage <= coverage_threshold:
            return
        length, label, source = candidate_brief(conn, qid)
        matched = summarize_subjects(subjects, max_matched_train_ids)
        removed_handle.write(
            "\t".join(
                str(value)
                for value in (
                    qid,
                    length or qlen,
                    label,
                    source,
                    matched,
                    f"{coverage:.8f}",
                    "benchmark_train_dev_blast_coverage",
                )
            )
            + "\n"
        )
        removed_count += 1
        update_batch.append((coverage, matched, qid))
        if len(update_batch) >= 10000:
            conn.executemany(
                """
                UPDATE candidates
                SET removed_by_train_blast=1, train_coverage=?, matched_train_ids=?
                WHERE candidate_id=?
                """,
                update_batch,
            )
            conn.commit()
            update_batch.clear()

    current_qid: str | None = None
    current_qlen = 0
    current_intervals: list[tuple[int, int]] = []
    current_subjects: set[str] = set()

    with open(blast_tsv, "rt") as handle, open(removed_tsv, "wt") as removed_handle:
        removed_handle.write(
            "\t".join(["query_id", "length", "label", "source", "matched_train_ids", "merged_query_coverage", "reason"])
            + "\n"
        )
        for line in handle:
            if not line.strip():
                continue
            qseqid, sseqid, qstart, qend, pident, _length, qlen, evalue, _bitscore = line.rstrip("\n").split("\t")
            if qseqid != current_qid:
                finalize(current_qid, current_qlen, current_intervals, current_subjects, removed_handle)
                current_qid = qseqid
                current_qlen = int(qlen)
                current_intervals = []
                current_subjects = set()
            if float(pident) < pident_min or float(evalue) > 1e-10:
                continue
            current_qlen = int(qlen)
            current_intervals.append((int(qstart), int(qend)))
            current_subjects.add(sseqid)
        finalize(current_qid, current_qlen, current_intervals, current_subjects, removed_handle)

    if update_batch:
        conn.executemany(
            """
            UPDATE candidates
            SET removed_by_train_blast=1, train_coverage=?, matched_train_ids=?
            WHERE candidate_id=?
            """,
            update_batch,
        )
        conn.commit()

    return removed_count


def apply_leakage_filter(
    conn: sqlite3.Connection,
    candidate_fasta: Path,
    realm_rank_dir: Path,
    metadata_dir: Path,
    work_dir: Path,
    blastn: str,
    makeblastdb: str,
    threads: int,
    coverage_threshold: float,
    pident_min: float,
    max_matched_train_ids: int,
) -> None:
    removed_tsv = metadata_dir / "test_v4_removed_by_train_dev_blast.tsv"
    already_removed = conn.execute("SELECT COUNT(*) FROM candidates WHERE removed_by_train_blast=1").fetchone()[0]
    if removed_tsv.exists() and already_removed > 0:
        log(f"reusing leakage filter results: {already_removed:,} removed candidates")
        return

    train_db = prepare_train_dev_blast_db(realm_rank_dir, work_dir, makeblastdb, threads)
    blast_tsv = work_dir / "candidates_vs_train_dev.tsv"
    if not blast_tsv.exists():
        log("running candidate-vs-final-train+dev BLAST leakage check")
        run_blastn_candidates(blastn, candidate_fasta, train_db, blast_tsv, threads)
    log("parsing candidate-vs-train+dev BLAST coverage")
    removed_count = parse_train_blast_streaming(
        conn,
        blast_tsv,
        removed_tsv,
        pident_min=pident_min,
        coverage_threshold=coverage_threshold,
        max_matched_train_ids=max_matched_train_ids,
    )
    log(f"removed by train+dev BLAST coverage > {coverage_threshold}: {removed_count:,}")


def write_empty_removed_tsv(metadata_dir: Path) -> None:
    write_tsv(
        metadata_dir / "test_v4_removed_by_train_dev_blast.tsv",
        ["query_id", "length", "label", "source", "matched_train_ids", "merged_query_coverage", "reason"],
        [],
    )


def reset_train_blast_marks(conn: sqlite3.Connection, metadata_dir: Path) -> None:
    marked = conn.execute(
        """
        SELECT COUNT(*)
        FROM candidates
        WHERE removed_by_train_blast != 0
           OR train_coverage IS NOT NULL
           OR matched_train_ids != ''
        """
    ).fetchone()[0]
    if marked:
        log(f"resetting stale candidate-vs-train+dev BLAST marks for {marked:,} candidates")
        conn.execute(
            """
            UPDATE candidates
            SET removed_by_train_blast=0,
                train_coverage=NULL,
                matched_train_ids=''
            WHERE removed_by_train_blast != 0
               OR train_coverage IS NOT NULL
               OR matched_train_ids != ''
            """
        )
        conn.commit()
    write_empty_removed_tsv(metadata_dir)


def available_counts(conn: sqlite3.Connection, length: int, sources: Iterable[str]) -> dict[str, int]:
    wanted = tuple(sources)
    placeholders = ",".join("?" for _ in wanted)
    rows = conn.execute(
        f"""
        SELECT source, COUNT(*)
        FROM candidates
        WHERE length=? AND class_label='negative' AND removed_by_train_blast=0
          AND source IN ({placeholders})
        GROUP BY source
        """,
        (length, *wanted),
    ).fetchall()
    counts = {source: 0 for source in wanted}
    counts.update({str(source): int(count) for source, count in rows})
    return counts


def source_balanced_capacity(counts: dict[str, int], sources: tuple[str, ...]) -> int:
    if not sources:
        return 0
    base = min(counts.get(source, 0) for source in sources)
    extra = sum(1 for source in sources if counts.get(source, 0) > base)
    return base * len(sources) + extra


def balanced_quotas(total: int, counts: dict[str, int], sources: tuple[str, ...]) -> dict[str, int]:
    if total <= 0:
        return {source: 0 for source in sources}
    base = total // len(sources)
    remainder = total - base * len(sources)
    quotas = {source: base for source in sources}
    eligible = [source for source in sources if counts.get(source, 0) > base]
    eligible.sort(key=lambda source: (-counts.get(source, 0), source))
    for source in eligible[:remainder]:
        quotas[source] += 1
    if sum(quotas.values()) != total or any(quotas[source] > counts.get(source, 0) for source in sources):
        raise RuntimeError(f"Could not allocate balanced quotas total={total} counts={counts} quotas={quotas}")
    return quotas


def stable_sample_ids(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple[object, ...],
    n: int,
    seed_parts: tuple[object, ...],
) -> list[str]:
    if n <= 0:
        return []
    heap: list[tuple[int, str]] = []
    for (candidate_id,) in conn.execute(sql, params):
        key = stable_int(*seed_parts, candidate_id)
        item = (-key, candidate_id)
        if len(heap) < n:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) < n:
        raise RuntimeError(f"Requested {n} candidates but only sampled {len(heap)} from query: {sql}")
    return [candidate_id for _neg_key, candidate_id in sorted(heap, reverse=True)]


def select_benchmarks(
    conn: sqlite3.Connection,
    lengths: tuple[int, ...],
    seed: int,
) -> dict[str, int]:
    conn.execute("DELETE FROM selected")
    family_n: dict[str, int] = {}
    for benchmark, sources in BENCHMARKS.items():
        feasible_pairs: list[int] = []
        for length in lengths:
            pos_count = conn.execute(
                """
                SELECT COUNT(*)
                FROM candidates
                WHERE length=? AND class_label='positive' AND removed_by_train_blast=0
                """,
                (length,),
            ).fetchone()[0]
            neg_counts = available_counts(conn, length, sources)
            neg_capacity = source_balanced_capacity(neg_counts, sources)
            feasible_pairs.append(min(int(pos_count), neg_capacity))
        n_family = min(feasible_pairs) if feasible_pairs else 0
        if n_family <= 0:
            raise RuntimeError(f"No feasible balanced benchmark size for {benchmark}; per-length pairs={feasible_pairs}")
        family_n[benchmark] = n_family
        log(f"{benchmark}: selecting {n_family:,} positives and {n_family:,} negatives per length")

        selected_rows: list[tuple[object, ...]] = []
        for length in lengths:
            pos_ids = stable_sample_ids(
                conn,
                """
                SELECT candidate_id
                FROM candidates
                WHERE length=? AND class_label='positive' AND removed_by_train_blast=0
                """,
                (length,),
                n_family,
                (seed, benchmark, length, "positive"),
            )
            for candidate_id in pos_ids:
                output_id = f"{benchmark}__{candidate_id}"
                selected_rows.append((benchmark, output_id, candidate_id, length, "positive"))

            neg_counts = available_counts(conn, length, sources)
            quotas = balanced_quotas(n_family, neg_counts, sources)
            for source, quota in quotas.items():
                neg_ids = stable_sample_ids(
                    conn,
                    """
                    SELECT candidate_id
                    FROM candidates
                    WHERE length=? AND class_label='negative' AND source=? AND removed_by_train_blast=0
                    """,
                    (length, source),
                    quota,
                    (seed, benchmark, length, "negative", source),
                )
                for candidate_id in neg_ids:
                    output_id = f"{benchmark}__{candidate_id}"
                    selected_rows.append((benchmark, output_id, candidate_id, length, "negative"))
        conn.executemany(
            """
            INSERT INTO selected (benchmark, output_id, candidate_id, length, class_label)
            VALUES (?, ?, ?, ?, ?)
            """,
            selected_rows,
        )
        conn.commit()
    return family_n


def load_selection_rows(conn: sqlite3.Connection) -> list[SelectionRow]:
    rows: list[SelectionRow] = []
    for row in conn.execute(
        """
        SELECT s.benchmark, s.output_id, c.candidate_id, c.length, c.class_label,
               c.label, c.realm, c.source, c.supergroup, c.genome_id, c.contig_id,
               c.start, c.end
        FROM selected s
        JOIN candidates c ON c.candidate_id=s.candidate_id
        ORDER BY s.benchmark, c.length, c.class_label, c.source, s.output_id
        """
    ):
        rows.append(
            SelectionRow(
                benchmark=str(row[0]),
                output_id=str(row[1]),
                candidate_id=str(row[2]),
                length=int(row[3]),
                class_label=str(row[4]),
                label=str(row[5]),
                realm=str(row[6]),
                source=str(row[7]),
                supergroup=str(row[8]),
                genome_id=str(row[9]),
                contig_id=str(row[10]),
                start=int(row[11]),
                end=int(row[12]),
            )
        )
    return rows


def write_benchmark_metadata(metadata_dir: Path, rows: list[SelectionRow]) -> None:
    write_tsv(
        metadata_dir / "benchmark_fragments.tsv",
        [
            "query_id",
            "candidate_id",
            "benchmark",
            "length",
            "class_label",
            "label",
            "realm",
            "source",
            "supergroup",
            "genome_id",
            "contig_id",
            "start",
            "end",
        ],
        (
            (
                row.output_id,
                row.candidate_id,
                row.benchmark,
                row.length,
                row.class_label,
                row.label,
                row.realm,
                row.source,
                row.supergroup,
                row.genome_id,
                row.contig_id,
                row.start,
                row.end,
            )
            for row in rows
        ),
    )


def selection_description(row: SelectionRow) -> str:
    attrs = [
        f"label={row.label}",
        f"source={row.source}",
        f"class_label={row.class_label}",
        f"benchmark={row.benchmark}",
        f"length={row.length}",
        f"genome={row.genome_id}",
        f"contig={row.contig_id}",
        f"start={row.start}",
        f"end={row.end}",
        f"candidate={row.candidate_id}",
    ]
    if row.realm:
        attrs.insert(2, f"realm={row.realm}")
    return " ".join(attrs)


def write_selected_fastas(
    candidate_fasta: Path,
    output_dir: Path,
    selection_rows: list[SelectionRow],
    lengths: tuple[int, ...],
    compression_level: int,
) -> None:
    for benchmark in BENCHMARKS:
        (output_dir / benchmark).mkdir(parents=True, exist_ok=True)

    by_candidate: dict[str, list[SelectionRow]] = defaultdict(list)
    for row in selection_rows:
        by_candidate[row.candidate_id].append(row)

    handles = {}
    try:
        for benchmark in BENCHMARKS:
            for length in lengths:
                path = output_dir / benchmark / f"{benchmark}-{length}.fasta.gz"
                handles[(benchmark, length)] = gzip.open(path, "wt", compresslevel=compression_level)

        written: Counter[tuple[str, int]] = Counter()
        for candidate_id, _header, seq in iter_fasta(candidate_fasta):
            rows = by_candidate.get(candidate_id)
            if not rows:
                continue
            for row in rows:
                handle = handles[(row.benchmark, row.length)]
                write_fasta(handle, row.output_id, seq, selection_description(row))
                written[(row.benchmark, row.length)] += 1
    finally:
        for handle in handles.values():
            handle.close()

    expected = Counter((row.benchmark, row.length) for row in selection_rows)
    missing = {key: expected[key] - written[key] for key in expected if written[key] != expected[key]}
    if missing:
        raise RuntimeError(f"Selected FASTA writing missed records: {missing}")

    for benchmark in BENCHMARKS:
        mixed_path = output_dir / benchmark / f"{benchmark}-mixed.fasta.gz"
        with open(mixed_path, "wb") as mixed:
            for length in lengths:
                fixed_path = output_dir / benchmark / f"{benchmark}-{length}.fasta.gz"
                with open(fixed_path, "rb") as src:
                    shutil.copyfileobj(src, mixed, length=1024 * 1024)


def iter_fasta_lengths(path: Path) -> Iterator[tuple[str, int, bool]]:
    for record_id, _header, seq in iter_fasta(path):
        yield record_id, len(seq), ("N" not in seq)


def write_counts_qc(conn: sqlite3.Connection, qc_dir: Path, lengths: tuple[int, ...]) -> None:
    rows: list[tuple[object, ...]] = []
    for benchmark in BENCHMARKS:
        for length in lengths:
            pos = conn.execute(
                "SELECT COUNT(*) FROM selected WHERE benchmark=? AND length=? AND class_label='positive'",
                (benchmark, length),
            ).fetchone()[0]
            neg = conn.execute(
                "SELECT COUNT(*) FROM selected WHERE benchmark=? AND length=? AND class_label='negative'",
                (benchmark, length),
            ).fetchone()[0]
            rows.append((benchmark, f"{benchmark}-{length}.fasta.gz", length, pos, neg, pos + neg))
        total = conn.execute("SELECT COUNT(*) FROM selected WHERE benchmark=?", (benchmark,)).fetchone()[0]
        rows.append((benchmark, f"{benchmark}-mixed.fasta.gz", "mixed", total // 2, total // 2, total))
    write_tsv(
        qc_dir / "benchmark_counts.tsv",
        ["benchmark", "file", "length", "positive_count", "negative_count", "total_count"],
        rows,
    )


def write_source_balance_qc(conn: sqlite3.Connection, qc_dir: Path, lengths: tuple[int, ...]) -> None:
    rows: list[tuple[object, ...]] = []
    for benchmark, sources in BENCHMARKS.items():
        for length in lengths:
            neg_counts = {
                source: int(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM selected s
                        JOIN candidates c ON c.candidate_id=s.candidate_id
                        WHERE s.benchmark=? AND s.length=? AND s.class_label='negative' AND c.source=?
                        """,
                        (benchmark, length, source),
                    ).fetchone()[0]
                )
                for source in sources
            }
            min_neg = min(neg_counts.values()) if neg_counts else 0
            max_neg = max(neg_counts.values()) if neg_counts else 0
            status = "pass" if max_neg - min_neg <= 1 else "fail"
            for source, count in sorted(neg_counts.items()):
                rows.append((benchmark, length, "negative_source", source, count, min_neg, max_neg, status))

            realm_counts = conn.execute(
                """
                SELECT c.realm, COUNT(*)
                FROM selected s
                JOIN candidates c ON c.candidate_id=s.candidate_id
                WHERE s.benchmark=? AND s.length=? AND s.class_label='positive'
                GROUP BY c.realm
                ORDER BY c.realm
                """,
                (benchmark, length),
            ).fetchall()
            for realm, count in realm_counts:
                rows.append((benchmark, length, "positive_realm", realm, count, "", "", "info"))
    write_tsv(
        qc_dir / "benchmark_source_balance.tsv",
        ["benchmark", "length", "category", "source_or_realm", "count", "min_count", "max_count", "status"],
        rows,
    )


def write_length_check_qc(output_dir: Path, qc_dir: Path, lengths: tuple[int, ...]) -> dict[str, tuple[str, str]]:
    rows: list[tuple[object, ...]] = []
    status_by_file: dict[str, tuple[str, str]] = {}
    for benchmark in BENCHMARKS:
        fixed_composition: Counter[int] = Counter()
        for length in lengths:
            path = output_dir / benchmark / f"{benchmark}-{length}.fasta.gz"
            observed_lengths: Counter[int] = Counter()
            acgt_only = True
            ids: set[str] = set()
            duplicate_ids = 0
            for record_id, seq_len, is_acgt in iter_fasta_lengths(path):
                observed_lengths[seq_len] += 1
                fixed_composition[seq_len] += 1
                acgt_only = acgt_only and is_acgt
                if record_id in ids:
                    duplicate_ids += 1
                ids.add(record_id)
            total = sum(observed_lengths.values())
            min_len = min(observed_lengths) if observed_lengths else 0
            max_len = max(observed_lengths) if observed_lengths else 0
            status = (
                "pass"
                if total > 0 and min_len == length and max_len == length and acgt_only and duplicate_ids == 0
                else "fail"
            )
            details = ",".join(f"{seq_len}:{count}" for seq_len, count in sorted(observed_lengths.items()))
            file_name = path.relative_to(output_dir)
            status_by_file[str(file_name)] = (status, details)
            rows.append((benchmark, file_name, length, total, min_len, max_len, acgt_only, duplicate_ids, details, status))

        mixed_path = output_dir / benchmark / f"{benchmark}-mixed.fasta.gz"
        mixed_lengths: Counter[int] = Counter()
        mixed_acgt_only = True
        for _record_id, seq_len, is_acgt in iter_fasta_lengths(mixed_path):
            mixed_lengths[seq_len] += 1
            mixed_acgt_only = mixed_acgt_only and is_acgt
        mixed_status = "pass" if mixed_lengths == fixed_composition and mixed_acgt_only else "fail"
        details = ",".join(f"{seq_len}:{count}" for seq_len, count in sorted(mixed_lengths.items()))
        file_name = mixed_path.relative_to(output_dir)
        status_by_file[str(file_name)] = (mixed_status, details)
        rows.append(
            (
                benchmark,
                file_name,
                "mixed",
                sum(mixed_lengths.values()),
                min(mixed_lengths) if mixed_lengths else 0,
                max(mixed_lengths) if mixed_lengths else 0,
                mixed_acgt_only,
                "",
                details,
                mixed_status,
            )
        )

    write_tsv(
        qc_dir / "benchmark_length_check.tsv",
        [
            "benchmark",
            "file",
            "expected_length",
            "record_count",
            "min_length",
            "max_length",
            "acgt_only",
            "duplicate_ids_within_file",
            "length_composition",
            "status",
        ],
        rows,
    )
    return status_by_file


def write_verification_qc(
    conn: sqlite3.Connection,
    output_dir: Path,
    realm_rank_dir: Path,
    metadata_dir: Path,
    qc_dir: Path,
    lengths: tuple[int, ...],
    family_n: dict[str, int],
    include_references: bool,
    run_train_blast: bool,
    coverage_threshold: float,
) -> None:
    checks: list[tuple[str, str, str]] = []
    required_inputs = [
        realm_rank_dir / "train.fasta.gz",
        realm_rank_dir / "dev.fasta.gz",
        realm_rank_dir / "test.fasta.gz",
        realm_rank_dir / "work/virus_all.fasta",
        realm_rank_dir / "work/nonviral_cleaned.fasta",
        realm_rank_dir / "work/nonviral_contigs.tsv",
        realm_rank_dir / "metadata/genome_split.tsv",
    ]
    missing = [str(path) for path in required_inputs if not path.exists()]
    checks.append(("required_realm_rank_inputs_exist", "pass" if not missing else "fail", ",".join(missing)))

    downloaded_path = metadata_dir / "downloaded_references.tsv"
    downloaded = read_tsv(downloaded_path) if downloaded_path.exists() else []
    ref_status = (not include_references) or all(row["genomic_fasta_gzip_check"] == "pass" for row in downloaded)
    checks.append(
        (
            "downloaded_references_valid",
            "pass" if ref_status else "fail",
            "skipped" if not include_references else str(len(downloaded)),
        )
    )

    if run_train_blast:
        removed_selected = conn.execute(
            """
            SELECT COUNT(*)
            FROM selected s
            JOIN candidates c ON c.candidate_id=s.candidate_id
            WHERE c.removed_by_train_blast=1
            """
        ).fetchone()[0]
        checks.append(("selected_not_removed_by_train_dev_blast", "pass" if removed_selected == 0 else "fail", str(removed_selected)))
        selected_over_threshold = conn.execute(
            """
            SELECT COUNT(*)
            FROM selected s
            JOIN candidates c ON c.candidate_id=s.candidate_id
            WHERE c.train_coverage > ?
            """,
            (coverage_threshold,),
        ).fetchone()[0]
        checks.append(
            (
                "selected_train_dev_coverage_le_threshold",
                "pass" if selected_over_threshold == 0 else "fail",
                f"{selected_over_threshold} selected candidates over {coverage_threshold}",
            )
        )
    else:
        checks.append(("selected_not_removed_by_train_dev_blast", "skipped", "candidate-vs-train+dev BLAST not requested"))
        checks.append(("selected_train_dev_coverage_le_threshold", "skipped", "candidate-vs-train+dev BLAST not requested"))

    old_train_exists = (realm_rank_dir / "train.fasta.gz").exists()
    old_dev_exists = (realm_rank_dir / "dev.fasta.gz").exists()
    old_test_exists = (realm_rank_dir / "test.fasta.gz").exists()
    checks.append(
        (
            "v4_train_dev_test_present",
            "pass" if old_train_exists and old_dev_exists and old_test_exists else "fail",
            f"train={old_train_exists} dev={old_dev_exists} test={old_test_exists}",
        )
    )

    if include_references:
        human_bat_outside_euk = conn.execute(
            """
            SELECT COUNT(*)
            FROM selected s
            JOIN candidates c ON c.candidate_id=s.candidate_id
            WHERE c.source IN ('human', 'bat') AND s.benchmark!='bench-euk'
            """
        ).fetchone()[0]
        checks.append(("human_bat_only_in_bench_euk", "pass" if human_bat_outside_euk == 0 else "fail", str(human_bat_outside_euk)))
    else:
        checks.append(("human_bat_only_in_bench_euk", "skipped", "human/bat references not included"))

    if min(lengths) >= 500:
        no_short = conn.execute("SELECT COUNT(*) FROM selected WHERE length < 500").fetchone()[0]
        checks.append(("no_300_499_bp_benchmark_records", "pass" if no_short == 0 else "fail", str(no_short)))
    else:
        checks.append(("no_300_499_bp_benchmark_records", "skipped", f"requested smoke/test lengths include {min(lengths)}"))

    status_by_file = write_length_check_qc(output_dir, qc_dir, lengths)
    failed_length = [file_name for file_name, (status, _details) in status_by_file.items() if status != "pass"]
    checks.append(("fasta_length_and_acgt_checks", "pass" if not failed_length else "fail", ",".join(failed_length)))

    requested_long_lengths = [length for length in (10000, 20000) if length in lengths]
    if requested_long_lengths:
        nonempty_long = []
        for benchmark in BENCHMARKS:
            for length in requested_long_lengths:
                count = conn.execute(
                    "SELECT COUNT(*) FROM selected WHERE benchmark=? AND length=?",
                    (benchmark, length),
                ).fetchone()[0]
                nonempty_long.append(count > 0)
        checks.append(("ten_k_and_twenty_k_files_nonempty", "pass" if all(nonempty_long) else "fail", str(nonempty_long)))
    else:
        checks.append(("ten_k_and_twenty_k_files_nonempty", "skipped", "10000/20000 not requested"))

    balance_failures: list[str] = []
    for benchmark, sources in BENCHMARKS.items():
        for length in lengths:
            expected_total = family_n[benchmark] * 2
            pos = conn.execute(
                "SELECT COUNT(*) FROM selected WHERE benchmark=? AND length=? AND class_label='positive'",
                (benchmark, length),
            ).fetchone()[0]
            neg = conn.execute(
                "SELECT COUNT(*) FROM selected WHERE benchmark=? AND length=? AND class_label='negative'",
                (benchmark, length),
            ).fetchone()[0]
            if pos != neg:
                balance_failures.append(f"{benchmark}:{length}:pos={pos}:neg={neg}")
            source_counts = []
            for source in sources:
                source_counts.append(
                    conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM selected s
                        JOIN candidates c ON c.candidate_id=s.candidate_id
                        WHERE s.benchmark=? AND s.length=? AND s.class_label='negative' AND c.source=?
                        """,
                        (benchmark, length, source),
                    ).fetchone()[0]
                )
            if max(source_counts) - min(source_counts) > 1:
                balance_failures.append(f"{benchmark}:{length}:sources={source_counts}")
            if pos + neg != expected_total:
                balance_failures.append(f"{benchmark}:{length}:total={pos + neg}:expected={expected_total}")
    checks.append(("benchmark_balance_checks", "pass" if not balance_failures else "fail", ";".join(balance_failures)))

    euk_sources = BENCHMARKS.get("bench-euk", ())
    euk_exact_failures: list[str] = []
    if not include_references:
        euk_exact_failures.append("skipped_no_references")
    elif euk_sources:
        for length in lengths:
            counts_by_source = []
            for source in euk_sources:
                count = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM selected s
                    JOIN candidates c ON c.candidate_id=s.candidate_id
                    WHERE s.benchmark='bench-euk' AND s.length=? AND s.class_label='negative' AND c.source=?
                    """,
                    (length, source),
                ).fetchone()[0]
                counts_by_source.append(int(count))
            if max(counts_by_source) - min(counts_by_source) > 1:
                euk_exact_failures.append(f"{length}:sources={counts_by_source}")
    else:
        euk_exact_failures.append("bench-euk has no sources")
    checks.append(
        (
            "bench_euk_negative_sources_balanced",
            "pass" if not euk_exact_failures else ("skipped" if euk_exact_failures == ["skipped_no_references"] else "fail"),
            ";".join(euk_exact_failures),
        )
    )

    removed_path = metadata_dir / "test_v4_removed_by_train_dev_blast.tsv"
    removed_rows = sum(1 for _ in open(removed_path, "rt")) - 1 if removed_path.exists() else -1
    checks.append(("removed_by_train_dev_blast_recorded", "pass" if removed_rows >= 0 else "fail", str(removed_rows)))

    write_tsv(qc_dir / "benchmark_verification.tsv", ["check", "status", "details"], checks)


def write_summary(
    output_dir: Path,
    metadata_dir: Path,
    lengths: tuple[int, ...],
    family_n: dict[str, int],
    coverage_threshold: float,
    pident_min: float,
    threads: int,
    include_references: bool,
    run_train_blast: bool,
) -> None:
    counts = read_tsv(output_dir / "qc/benchmark_counts.tsv")
    refs_path = metadata_dir / "downloaded_references.tsv"
    refs = read_tsv(refs_path) if refs_path.exists() else []
    lines = [
        "# Realm-Rank Benchmark Test v4 Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "The existing `processed_data/realm_rank_v4/test.fasta.gz` is retained as the 300-2000 bp held-out test set. "
        "This v4 output is a separate fixed-length benchmark suite.",
        "",
        "## References",
        "",
    ]
    if include_references:
        for row in refs:
            lines.append(
                f"- {row['source']}: {row['accession']} ({row['assembly_name']}), "
                f"{row['organism']}; FASTA sha256 `{row['genomic_fasta_sha256']}`"
            )
        lines.extend(
            [
                "",
                "Human and bat references are used only as `bench-euk` test negatives and are not added to training. "
                "Insect negatives come from the v4 test split.",
            ]
        )
    else:
        lines.append("- Human/bat references were not included in this run; `bench-euk` negatives use fungi/protozoa/insect test genomes.")
    lines.extend(
        [
            "",
            "## BLAST Filters",
            "",
        ]
    )
    if include_references:
        lines.append(
            f"- Human/bat decontamination: `blastn -task megablast -evalue 1e-10 -num_threads {threads}`; "
            "keeps intervals for removal at `pident >= 90`, `length/qlen >= 0.80`, `evalue <= 1e-10`."
        )
    else:
        lines.append("- Human/bat decontamination BLAST was skipped because no new human/bat references were included.")
    if run_train_blast:
        lines.append(
            f"- Benchmark-vs-train+dev leakage: final `processed_data/realm_rank_v4/train.fasta.gz` and `dev.fasta.gz` were decompressed and indexed; "
            f"candidate HSPs use `pident >= {pident_min:g}`, `evalue <= 1e-10`, and remove candidates with merged query "
            f"coverage `> {coverage_threshold:g}`."
        )
    else:
        lines.append(
            "- Candidate-vs-train+dev leakage BLAST was skipped; this is only intended for parser/smoke checks."
        )
    lines.extend(["", "## Sampling", ""])
    for benchmark, n_family in family_n.items():
        sources = ", ".join(BENCHMARKS[benchmark])
        lines.append(
            f"- {benchmark}: {n_family} positive and {n_family} source-balanced negative records per length; negatives: {sources}."
        )
    lines.extend(
        [
            "",
            "Positive viral records preserve the natural realm distribution of the test virus genome pool; realm counts are in "
            "`qc/benchmark_source_balance.tsv`.",
            "",
            "## Outputs",
            "",
        ]
    )
    for row in counts:
        lines.append(
            f"- `{row['benchmark']}/{row['file']}`: length={row['length']}, "
            f"positive={row['positive_count']}, negative={row['negative_count']}, total={row['total_count']}"
        )
    lines.extend(
        [
            "",
            "Mixed FASTA files are bytewise gzip concatenations of the five fixed-length FASTA files in this order: "
            + ", ".join(str(length) for length in lengths)
            + ". No mixed records were re-sliced.",
            "",
            "## Metadata and QC",
            "",
            "- `metadata/benchmark_fragments.tsv` records selected benchmark fragments and genomic coordinates.",
        ]
    )
    if include_references:
        lines.append("- `metadata/downloaded_references.tsv` records reference FTP paths, gzip checks, and sha256 values.")
    else:
        lines.append("- `metadata/downloaded_references.tsv` is header-only because human/bat references were not included.")
    if run_train_blast:
        lines.append("- `metadata/test_v4_removed_by_train_dev_blast.tsv` records candidates removed by final-train+dev BLAST coverage.")
    else:
        lines.append("- `metadata/test_v4_removed_by_train_dev_blast.tsv` is header-only because candidate-vs-train+dev BLAST was skipped.")
    lines.append(
        "- `qc/benchmark_counts.tsv`, `qc/benchmark_length_check.tsv`, `qc/benchmark_source_balance.tsv`, and "
        "`qc/benchmark_verification.tsv` record balance and integrity checks."
    )
    (output_dir / "data_process_test_v4_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_required_inputs(realm_rank_dir: Path) -> None:
    required = [
        realm_rank_dir / "train.fasta.gz",
        realm_rank_dir / "dev.fasta.gz",
        realm_rank_dir / "test.fasta.gz",
        realm_rank_dir / "work/virus_all.fasta",
        realm_rank_dir / "work/nonviral_cleaned.fasta",
        realm_rank_dir / "work/nonviral_contigs.tsv",
        realm_rank_dir / "work/virus_contigs.tsv",
        realm_rank_dir / "metadata/genome_split.tsv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("Missing required inputs: " + ", ".join(missing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realm-rank-dir", type=Path, default=PROCESSED_DATA_ROOT / "realm_rank_v4")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_ROOT / "realm_rank_test_v4")
    parser.add_argument("--reference-dir", type=Path, default=RAW_DATA_ROOT / "reference")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--blastn", default=DEFAULT_BLASTN)
    parser.add_argument("--makeblastdb", default=DEFAULT_MAKEBLASTDB)
    parser.add_argument("--compression-level", type=int, default=1)
    parser.add_argument("--test-train-coverage", type=float, default=0.50)
    parser.add_argument("--pident-min", type=float, default=90.0)
    parser.add_argument("--max-matched-train-ids", type=int, default=100)
    parser.add_argument(
        "--include-references",
        dest="include_references",
        action="store_true",
        default=True,
        help="Include fixed human/bat references as bench-euk negatives and run their virus decontamination BLAST. Default: enabled.",
    )
    parser.add_argument(
        "--no-include-references",
        dest="include_references",
        action="store_false",
        help="Disable human/bat references; bench-euk then uses only native fungi/protozoa/insect negatives.",
    )
    parser.add_argument(
        "--run-train-dev-blast",
        dest="run_train_blast",
        action="store_true",
        default=True,
        help="Run candidate-vs-final-train+dev BLAST leakage filtering. Default: enabled.",
    )
    parser.add_argument(
        "--run-train-blast",
        dest="run_train_blast",
        action="store_true",
        help="Backward-compatible alias for --run-train-dev-blast.",
    )
    parser.add_argument(
        "--skip-train-dev-blast",
        dest="run_train_blast",
        action="store_false",
        help="Skip candidate-vs-final-train+dev BLAST leakage filtering.",
    )
    parser.add_argument(
        "--skip-train-blast",
        dest="run_train_blast",
        action="store_false",
        help="Backward-compatible alias for --skip-train-dev-blast.",
    )
    parser.add_argument("--force", action="store_true", help="Remove and rebuild the v4 output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.time()
    lengths = tuple(int(length) for length in args.lengths)
    if any(length <= 0 for length in lengths):
        raise RuntimeError(f"All lengths must be positive: {lengths}")
    if len(set(lengths)) != len(lengths):
        raise RuntimeError(f"Lengths must be unique: {lengths}")

    check_required_inputs(args.realm_rank_dir)
    metadata_dir, qc_dir, work_dir = ensure_dirs(args.output_dir, args.force)
    blastn = resolve_tool("blastn", args.blastn)
    makeblastdb = resolve_tool("makeblastdb", args.makeblastdb)

    if args.include_references:
        BENCHMARKS["bench-euk"] = REFERENCE_EUK_SOURCES
        log("downloading or validating fixed human/bat references")
        reference_records = download_references(args.reference_dir, metadata_dir)
        reference_cleaned_fasta, reference_contigs_tsv = prepare_references(
            reference_records,
            args.realm_rank_dir,
            metadata_dir,
            work_dir,
            blastn,
            makeblastdb,
            args.threads,
        )
    else:
        BENCHMARKS["bench-euk"] = NATIVE_EUK_SOURCES
        log("skipping human/bat references; bench-euk uses existing fungi/protozoa/insect test genomes")
        write_tsv(
            metadata_dir / "downloaded_references.tsv",
            [
                "source",
                "accession",
                "organism",
                "assembly_name",
                "ftp_path",
                "genomic_fasta",
                "genomic_fasta_bytes",
                "genomic_fasta_sha256",
                "genomic_fasta_gzip_check",
                "assembly_report",
                "assembly_report_bytes",
                "assembly_report_sha256",
            ],
            [],
        )
        reference_cleaned_fasta = None
        reference_contigs_tsv = None

    candidate_db = work_dir / "candidates.sqlite"
    candidate_fasta = work_dir / "candidates.prefilter.fasta"
    conn = generate_candidate_pool(
        candidate_db,
        candidate_fasta,
        args.realm_rank_dir,
        reference_cleaned_fasta,
        reference_contigs_tsv,
        lengths,
    )
    if args.run_train_blast:
        apply_leakage_filter(
            conn,
            candidate_fasta,
            args.realm_rank_dir,
            metadata_dir,
            work_dir,
            blastn,
            makeblastdb,
            args.threads,
            coverage_threshold=args.test_train_coverage,
            pident_min=args.pident_min,
            max_matched_train_ids=args.max_matched_train_ids,
        )
    else:
        log("skipping fixed candidate-vs-final-train+dev BLAST leakage filter")
        reset_train_blast_marks(conn, metadata_dir)

    family_n = select_benchmarks(conn, lengths, args.seed)
    selection_rows = load_selection_rows(conn)
    write_benchmark_metadata(metadata_dir, selection_rows)
    write_selected_fastas(candidate_fasta, args.output_dir, selection_rows, lengths, args.compression_level)
    write_counts_qc(conn, qc_dir, lengths)
    write_source_balance_qc(conn, qc_dir, lengths)
    write_verification_qc(
        conn,
        args.output_dir,
        args.realm_rank_dir,
        metadata_dir,
        qc_dir,
        lengths,
        family_n,
        include_references=args.include_references,
        run_train_blast=args.run_train_blast,
        coverage_threshold=args.test_train_coverage,
    )
    write_summary(
        args.output_dir,
        metadata_dir,
        lengths,
        family_n,
        args.test_train_coverage,
        args.pident_min,
        args.threads,
        include_references=args.include_references,
        run_train_blast=args.run_train_blast,
    )
    conn.close()
    log(f"done: {args.output_dir} elapsed={time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
