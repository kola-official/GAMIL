#!/usr/bin/env python3
"""Prepare strict Insecta assembly inputs for Realm-Rank v3.

The script queries NCBI Assembly for Insecta chromosome/complete assemblies,
selects one assembly per genus, downloads the genomic FASTA plus assembly
report, and writes chromosome-only FASTA files under ``data/insect``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
GAMIL_ROOT = Path(os.environ.get("GAMIL_ROOT", SCRIPT_DIR.parents[1])).resolve()
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", GAMIL_ROOT / "raw_data" / "local_sources")).resolve()
DEFAULT_SEED = 1729
DEFAULT_THREADS = 54
INSECTA_TAXID = 50557
FASTA_WIDTH = 80
DOWNLOAD_SPEED_LIMIT = "50000"
DOWNLOAD_SPEED_TIME = "180"
ALLOWED_ASSEMBLY_LEVELS = {"complete genome", "chromosome"}
ALLOWED_MOLECULE_TYPES = {"chromosome"}
NON_DNA_RE = re.compile(r"[^ACGTN]")


@dataclass(frozen=True)
class AssemblyCandidate:
    uid: str
    accession: str
    source: str
    ftp_path: str
    fasta_url: str
    report_url: str
    assembly_name: str
    organism: str
    species_name: str
    taxid: str
    genus: str
    assembly_status: str
    seq_release_date: str
    last_update_date: str
    date_sort: int
    refseq_category: str
    anomalous: str
    excluded_from_refseq: str
    is_latest: bool
    contig_n50: int
    scaffold_n50: int
    total_length: int
    property_list: str


@dataclass(frozen=True)
class ChromosomeRecord:
    sequence_name: str
    role: str
    assigned_molecule: str
    molecule_type: str
    genbank_accession: str
    refseq_accession: str
    assembly_unit: str
    length: int
    ucsc_name: str


@dataclass(frozen=True)
class PreparedAssembly:
    candidate: AssemblyCandidate
    raw_fasta: Path
    assembly_report: Path
    filtered_fasta: Path
    retained_records: list[ChromosomeRecord]
    retained_bp: int
    status: str
    notes: str


def log(message: str) -> None:
    print(f"[prepare-insect-v3] {message}", flush=True)


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("_")
    return safe or "unknown"


def stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def normalize_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return NON_DNA_RE.sub("N", seq)


def accession_from_header_id(record_id: str) -> str:
    token = record_id.split()[0]
    if "|" in token:
        parts = [part for part in token.split("|") if part]
        for part in reversed(parts):
            if re.match(r"^[A-Z]{1,4}_[A-Z0-9]+(\.\d+)?$", part) or re.match(
                r"^[A-Z]{1,4}\d+(\.\d+)?$", part
            ):
                return part
    return token


def genus_from_name(species_name: str, organism: str) -> str:
    text = species_name or organism
    text = re.sub(r"\s*\(.*?\)\s*", " ", text).strip()
    words = text.split()
    if not words:
        return "unknown"
    if words[0].lower() in {"candidatus", "candidate"} and len(words) > 1:
        return safe_id(f"{words[0]}_{words[1]}")
    return safe_id(words[0])


def as_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(str(value)))
    except ValueError:
        return default


def parse_ncbi_date(value: str) -> int:
    value = (value or "").strip()
    for fmt in ("%Y/%m/%d %H:%M", "%m/%d/%y %H:%M", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return int(datetime.strptime(value, fmt).timestamp())
        except ValueError:
            continue
    return 0


def stats_total_length(meta: str) -> int:
    match = re.search(r'<Stat category="total_length" sequence_tag="all">(\d+)</Stat>', meta or "")
    return int(match.group(1)) if match else 0


def write_tsv(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wt", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


def eutils_json(
    endpoint: str,
    params: dict[str, object],
    email: str,
    api_key: str,
    delay: float,
    retries: int = 5,
) -> dict[str, object]:
    merged = dict(params)
    merged["retmode"] = "json"
    merged["tool"] = "realm-rank-insect-v3"
    if email:
        merged["email"] = email
    if api_key:
        merged["api_key"] = api_key
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{endpoint}.fcgi?" + urlencode(merged)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            try:
                with urlopen(url, timeout=120) as handle:
                    payload = json.load(handle)
            except Exception:
                curl = shutil.which("curl")
                if not curl:
                    raise
                proc = subprocess.run(
                    [
                        curl,
                        "--silent",
                        "--show-error",
                        "-L",
                        "--fail",
                        "--retry",
                        "3",
                        "--retry-delay",
                        "2",
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = json.loads(proc.stdout)
            if delay > 0:
                time.sleep(delay)
            return payload
        except Exception as exc:  # noqa: BLE001 - report transient NCBI failures with retry context
            last_error = exc
            sleep_for = min(60, delay + attempt * 2)
            log(f"NCBI {endpoint} attempt {attempt}/{retries} failed: {exc}; sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"NCBI {endpoint} failed after {retries} attempts: {last_error}")


def query_assembly_uids(email: str, api_key: str, delay: float, retmax: int) -> list[str]:
    term = f'txid{INSECTA_TAXID}[Organism:exp] AND ("complete genome"[Assembly Level] OR chromosome[Assembly Level])'
    payload = eutils_json(
        "esearch",
        {"db": "assembly", "term": term, "retmax": retmax},
        email,
        api_key,
        delay,
    )
    result = payload.get("esearchresult", {})
    count = int(result.get("count", "0"))
    uids = [str(uid) for uid in result.get("idlist", [])]
    if count > len(uids):
        raise RuntimeError(f"NCBI returned {len(uids)} UIDs but reports {count}; increase --retmax")
    log(f"NCBI Assembly query returned {len(uids):,} Insecta complete/chromosome assemblies")
    return uids


def url_base(ftp_path: str) -> str:
    return ftp_path.replace("ftp://", "https://", 1).rstrip("/")


def candidate_from_summary(uid: str, record: dict[str, object]) -> AssemblyCandidate | None:
    assembly_status = str(record.get("assemblystatus") or "")
    if assembly_status.lower() not in ALLOWED_ASSEMBLY_LEVELS:
        return None

    synonym = record.get("synonym") or {}
    if not isinstance(synonym, dict):
        synonym = {}
    refseq_acc = str(synonym.get("refseq") or "")
    genbank_acc = str(synonym.get("genbank") or "")
    assembly_acc = str(record.get("assemblyaccession") or "")
    if not refseq_acc and assembly_acc.startswith("GCF_"):
        refseq_acc = assembly_acc
    if not genbank_acc and assembly_acc.startswith("GCA_"):
        genbank_acc = assembly_acc

    refseq_ftp = str(record.get("ftppath_refseq") or "")
    genbank_ftp = str(record.get("ftppath_genbank") or "")
    if refseq_ftp and refseq_acc:
        source = "RefSeq"
        accession = refseq_acc
        ftp_path = refseq_ftp
    elif genbank_ftp:
        source = "GenBank"
        accession = genbank_acc or assembly_acc
        ftp_path = genbank_ftp
    else:
        return None

    base = url_base(ftp_path)
    basename = base.rstrip("/").split("/")[-1]
    fasta_url = f"{base}/{basename}_genomic.fna.gz"
    report_url = f"{base}/{basename}_assembly_report.txt"

    property_list = record.get("propertylist") or []
    if not isinstance(property_list, list):
        property_list = [str(property_list)]
    anomalous = record.get("anomalouslist") or []
    if not isinstance(anomalous, list):
        anomalous = [str(anomalous)]
    excluded = record.get("exclfromrefseq") or []
    if not isinstance(excluded, list):
        excluded = [str(excluded)]

    source_date = (
        str(record.get("asmreleasedate_refseq") or "")
        if source == "RefSeq"
        else str(record.get("asmreleasedate_genbank") or "")
    )
    seq_date = source_date or str(record.get("seqreleasedate") or "")
    last_update = str(record.get("lastupdatedate") or record.get("asmupdatedate") or "")
    species_name = str(record.get("speciesname") or "")
    organism = str(record.get("organism") or "")
    genus = genus_from_name(species_name, organism)

    return AssemblyCandidate(
        uid=uid,
        accession=accession,
        source=source,
        ftp_path=ftp_path,
        fasta_url=fasta_url,
        report_url=report_url,
        assembly_name=str(record.get("assemblyname") or accession),
        organism=organism,
        species_name=species_name,
        taxid=str(record.get("taxid") or ""),
        genus=genus,
        assembly_status=assembly_status,
        seq_release_date=seq_date,
        last_update_date=last_update,
        date_sort=max(parse_ncbi_date(seq_date), parse_ncbi_date(last_update)),
        refseq_category=str(record.get("refseq_category") or ""),
        anomalous=";".join(str(item) for item in anomalous),
        excluded_from_refseq=";".join(str(item) for item in excluded),
        is_latest="latest" in {str(item).lower() for item in property_list},
        contig_n50=as_int(record.get("contign50")),
        scaffold_n50=as_int(record.get("scaffoldn50")),
        total_length=stats_total_length(str(record.get("meta") or "")),
        property_list=";".join(str(item) for item in property_list),
    )


def fetch_assembly_candidates(
    uids: list[str],
    email: str,
    api_key: str,
    batch_size: int,
    delay: float,
) -> list[AssemblyCandidate]:
    candidates: list[AssemblyCandidate] = []
    for start in range(0, len(uids), batch_size):
        batch = uids[start : start + batch_size]
        payload = eutils_json(
            "esummary",
            {"db": "assembly", "id": ",".join(batch)},
            email,
            api_key,
            delay,
        )
        result = payload.get("result", {})
        result_uids = [str(uid) for uid in result.get("uids", [])]
        for uid in result_uids:
            record = result.get(uid)
            if not isinstance(record, dict):
                continue
            candidate = candidate_from_summary(uid, record)
            if candidate is not None:
                candidates.append(candidate)
        log(f"fetched assembly summaries {min(start + batch_size, len(uids)):,}/{len(uids):,}")
    log(f"usable strict Insecta candidates: {len(candidates):,}")
    return candidates


def candidate_sort_key(candidate: AssemblyCandidate, seed: int) -> tuple[object, ...]:
    source_rank = 0 if candidate.source == "RefSeq" else 1
    level_rank = 0 if candidate.assembly_status.lower() == "complete genome" else 1
    latest_rank = 0 if candidate.is_latest else 1
    anomaly_rank = 1 if candidate.anomalous or candidate.excluded_from_refseq else 0
    return (
        source_rank,
        level_rank,
        latest_rank,
        anomaly_rank,
        -candidate.date_sort,
        -candidate.scaffold_n50,
        -candidate.contig_n50,
        -candidate.total_length,
        stable_int(seed, candidate.genus, candidate.accession),
        candidate.accession,
    )


def rank_candidates_by_genus(candidates: list[AssemblyCandidate], seed: int) -> dict[str, list[AssemblyCandidate]]:
    by_genus: dict[str, list[AssemblyCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_genus[candidate.genus].append(candidate)
    return {
        genus: sorted(rows, key=lambda candidate: candidate_sort_key(candidate, seed))
        for genus, rows in sorted(by_genus.items())
    }


def run_command(cmd: list[str]) -> None:
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def download_url(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
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
        DOWNLOAD_SPEED_LIMIT,
        "--speed-time",
        DOWNLOAD_SPEED_TIME,
        "-C",
        "-",
        "-o",
        str(tmp_path),
        url,
    ]
    run_command(cmd)
    tmp_path.replace(output_path)


def validate_gzip(path: Path) -> tuple[Path, bool, str]:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return path, True, ""
    except Exception as exc:  # noqa: BLE001 - validation should report file-level errors
        return path, False, str(exc)


def parse_assembly_report(report_path: Path) -> list[ChromosomeRecord]:
    header: list[str] | None = None
    records: list[ChromosomeRecord] = []
    with open(report_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith("# Sequence-Name"):
                header = line.lstrip("# ").split("\t")
                continue
            if line.startswith("#") or header is None:
                continue
            parts = line.split("\t")
            if len(parts) < len(header):
                continue
            row = dict(zip(header, parts))
            role = row.get("Sequence-Role", "")
            molecule_type = row.get("Assigned-Molecule-Location/Type", "")
            if role.lower() != "assembled-molecule":
                continue
            if molecule_type.lower() not in ALLOWED_MOLECULE_TYPES:
                continue
            records.append(
                ChromosomeRecord(
                    sequence_name=row.get("Sequence-Name", ""),
                    role=role,
                    assigned_molecule=row.get("Assigned-Molecule", ""),
                    molecule_type=molecule_type,
                    genbank_accession=row.get("GenBank-Accn", ""),
                    refseq_accession=row.get("RefSeq-Accn", ""),
                    assembly_unit=row.get("Assembly-Unit", ""),
                    length=as_int(row.get("Sequence-Length")),
                    ucsc_name=row.get("UCSC-style-name", ""),
                )
            )
    return records


def accepted_accessions(records: list[ChromosomeRecord]) -> set[str]:
    accessions: set[str] = set()
    for record in records:
        for accession in (record.genbank_accession, record.refseq_accession):
            if accession and accession.lower() != "na":
                accessions.add(accession)
    return accessions


def write_filtered_fasta(raw_fasta: Path, filtered_fasta: Path, wanted: set[str], force: bool) -> tuple[int, int, set[str]]:
    if filtered_fasta.exists() and not force:
        record_count = 0
        total_bp = 0
        kept_accessions: set[str] = set()
        with gzip.open(filtered_fasta, "rt") as handle:
            for line in handle:
                if line.startswith(">"):
                    record_count += 1
                    kept_accessions.add(accession_from_header_id(line[1:].split()[0]))
                else:
                    total_bp += len(line.strip())
        return record_count, total_bp, kept_accessions

    filtered_fasta.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filtered_fasta.with_suffix(filtered_fasta.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    record_count = 0
    total_bp = 0
    kept_accessions: set[str] = set()
    keep = False
    with gzip.open(raw_fasta, "rt") as src, gzip.open(tmp_path, "wt", compresslevel=1) as dst:
        for raw_line in src:
            if raw_line.startswith(">"):
                record_id = raw_line[1:].split()[0]
                accession = accession_from_header_id(record_id)
                keep = accession in wanted
                if keep:
                    record_count += 1
                    kept_accessions.add(accession)
                    dst.write(raw_line)
                continue
            if keep:
                seq = normalize_dna(raw_line.strip())
                total_bp += len(seq)
                for start in range(0, len(seq), FASTA_WIDTH):
                    dst.write(seq[start : start + FASTA_WIDTH] + "\n")
    if record_count == 0:
        tmp_path.unlink(missing_ok=True)
        return 0, 0, set()
    tmp_path.replace(filtered_fasta)
    return record_count, total_bp, kept_accessions


def candidate_paths(candidate: AssemblyCandidate, insect_root: Path) -> tuple[Path, Path, Path]:
    basename = url_base(candidate.ftp_path).rstrip("/").split("/")[-1]
    raw_fasta = insect_root / "raw" / f"{basename}_genomic.fna.gz"
    report = insect_root / "assembly_reports" / f"{basename}_assembly_report.txt"
    filtered = insect_root / "assemblies" / f"{basename}_genomic.fna.gz"
    return raw_fasta, report, filtered


def prepare_candidate(candidate: AssemblyCandidate, insect_root: Path, force: bool) -> PreparedAssembly:
    raw_fasta, report, filtered = candidate_paths(candidate, insect_root)
    download_url(candidate.report_url, report)
    chromosome_records = parse_assembly_report(report)
    wanted = accepted_accessions(chromosome_records)
    if not wanted:
        return PreparedAssembly(candidate, raw_fasta, report, filtered, [], 0, "skipped", "no nuclear chromosome records in assembly report")
    if force or not filtered.exists():
        download_url(candidate.fasta_url, raw_fasta)
        _path, ok, error = validate_gzip(raw_fasta)
        if not ok:
            raw_fasta.unlink(missing_ok=True)
            download_url(candidate.fasta_url, raw_fasta)
            _path, ok, error = validate_gzip(raw_fasta)
        if not ok:
            raise RuntimeError(f"raw FASTA gzip validation failed for {candidate.accession}: {error}")
    record_count, retained_bp, kept_accessions = write_filtered_fasta(raw_fasta, filtered, wanted, force)
    if record_count < 1:
        return PreparedAssembly(candidate, raw_fasta, report, filtered, [], 0, "skipped", "FASTA contained none of the report chromosome accessions")
    retained_records = [
        record
        for record in chromosome_records
        if record.genbank_accession in kept_accessions or record.refseq_accession in kept_accessions
    ]
    return PreparedAssembly(candidate, raw_fasta, report, filtered, retained_records, retained_bp, "selected", "")


def prepare_genus(
    genus: str,
    genus_candidates: list[AssemblyCandidate],
    insect_root: Path,
    force: bool,
) -> tuple[str, PreparedAssembly | None, list[PreparedAssembly]]:
    skipped: list[PreparedAssembly] = []
    for candidate in genus_candidates:
        try:
            item = prepare_candidate(candidate, insect_root, force)
        except Exception as exc:  # noqa: BLE001 - do not silently drop genera on transport or gzip failures
            raise RuntimeError(f"genus={genus} accession={candidate.accession} failed during download/filter: {exc}") from exc
        if item.status == "selected":
            return genus, item, skipped
        skipped.append(item)
    return genus, None, skipped


def write_candidate_metadata(metadata_dir: Path, ranked: dict[str, list[AssemblyCandidate]]) -> None:
    rows = []
    for genus, candidates in ranked.items():
        for rank, candidate in enumerate(candidates, start=1):
            rows.append(
                (
                    genus,
                    rank,
                    candidate.accession,
                    candidate.source,
                    candidate.assembly_status,
                    candidate.assembly_name,
                    candidate.organism,
                    candidate.taxid,
                    candidate.seq_release_date,
                    candidate.last_update_date,
                    int(candidate.is_latest),
                    candidate.anomalous,
                    candidate.excluded_from_refseq,
                    candidate.refseq_category,
                    candidate.contig_n50,
                    candidate.scaffold_n50,
                    candidate.total_length,
                    candidate.ftp_path,
                )
            )
    write_tsv(
        metadata_dir / "insect_candidates.tsv",
        [
            "genus",
            "rank_within_genus",
            "accession",
            "source",
            "assembly_status",
            "assembly_name",
            "organism",
            "taxid",
            "seq_release_date",
            "last_update_date",
            "is_latest",
            "anomalous",
            "excluded_from_refseq",
            "refseq_category",
            "contig_n50",
            "scaffold_n50",
            "total_length",
            "ftp_path",
        ],
        rows,
    )


def load_candidate_metadata(metadata_dir: Path) -> dict[str, list[AssemblyCandidate]]:
    path = metadata_dir / "insect_candidates.tsv"
    rows = read_tsv(path)
    ranked_with_order: dict[str, list[tuple[int, AssemblyCandidate]]] = defaultdict(list)
    genus_order: list[str] = []
    for row in rows:
        ftp_path = row["ftp_path"]
        base = url_base(ftp_path)
        basename = base.rstrip("/").split("/")[-1]
        seq_release_date = row["seq_release_date"]
        last_update_date = row["last_update_date"]
        candidate = AssemblyCandidate(
            uid="",
            accession=row["accession"],
            source=row["source"],
            ftp_path=ftp_path,
            fasta_url=f"{base}/{basename}_genomic.fna.gz",
            report_url=f"{base}/{basename}_assembly_report.txt",
            assembly_name=row["assembly_name"],
            organism=row["organism"],
            species_name="",
            taxid=row["taxid"],
            genus=row["genus"],
            assembly_status=row["assembly_status"],
            seq_release_date=seq_release_date,
            last_update_date=last_update_date,
            date_sort=max(parse_ncbi_date(seq_release_date), parse_ncbi_date(last_update_date)),
            refseq_category=row["refseq_category"],
            anomalous=row["anomalous"],
            excluded_from_refseq=row["excluded_from_refseq"],
            is_latest=row["is_latest"] == "1",
            contig_n50=as_int(row["contig_n50"]),
            scaffold_n50=as_int(row["scaffold_n50"]),
            total_length=as_int(row["total_length"]),
            property_list="",
        )
        if candidate.genus not in ranked_with_order:
            genus_order.append(candidate.genus)
        ranked_with_order[candidate.genus].append((as_int(row["rank_within_genus"], 1), candidate))

    ordered: dict[str, list[AssemblyCandidate]] = {}
    for genus in genus_order:
        ordered[genus] = [candidate for _rank, candidate in sorted(ranked_with_order[genus], key=lambda item: item[0])]
    return ordered


def write_outputs(insect_root: Path, prepared: list[PreparedAssembly], skipped: list[PreparedAssembly]) -> None:
    metadata_dir = insect_root / "metadata"
    selected = [item for item in prepared if item.status == "selected"]
    selected_filtered = {item.filtered_fasta.resolve() for item in selected}
    unused_dir = insect_root / "unused_assemblies"
    unused_dir.mkdir(parents=True, exist_ok=True)
    for path in (insect_root / "assemblies").glob("*.fna.gz"):
        if path.resolve() not in selected_filtered:
            shutil.move(str(path), str(unused_dir / path.name))
    (insect_root / "insect.txt").write_text(
        "".join(f"{item.candidate.accession}\n" for item in sorted(selected, key=lambda item: item.candidate.genus)),
        encoding="utf-8",
    )
    write_tsv(
        metadata_dir / "insect_assemblies.tsv",
        [
            "genus",
            "accession",
            "source",
            "assembly_status",
            "assembly_name",
            "organism",
            "taxid",
            "seq_release_date",
            "last_update_date",
            "is_latest",
            "anomalous",
            "excluded_from_refseq",
            "refseq_category",
            "contig_n50",
            "scaffold_n50",
            "total_length",
            "raw_fasta",
            "assembly_report",
            "filtered_fasta",
            "retained_chromosome_count",
            "retained_bp",
            "status",
            "notes",
        ],
        (
            (
                item.candidate.genus,
                item.candidate.accession,
                item.candidate.source,
                item.candidate.assembly_status,
                item.candidate.assembly_name,
                item.candidate.organism,
                item.candidate.taxid,
                item.candidate.seq_release_date,
                item.candidate.last_update_date,
                int(item.candidate.is_latest),
                item.candidate.anomalous,
                item.candidate.excluded_from_refseq,
                item.candidate.refseq_category,
                item.candidate.contig_n50,
                item.candidate.scaffold_n50,
                item.candidate.total_length,
                item.raw_fasta,
                item.assembly_report,
                item.filtered_fasta,
                len(item.retained_records),
                item.retained_bp,
                item.status,
                item.notes,
            )
            for item in sorted(selected, key=lambda item: item.candidate.genus)
        ),
    )
    write_tsv(
        metadata_dir / "insect_chromosomes.tsv",
        [
            "genus",
            "assembly_accession",
            "source",
            "sequence_name",
            "sequence_role",
            "assigned_molecule",
            "molecule_type",
            "genbank_accession",
            "refseq_accession",
            "assembly_unit",
            "sequence_length",
            "ucsc_name",
        ],
        (
            (
                item.candidate.genus,
                item.candidate.accession,
                item.candidate.source,
                record.sequence_name,
                record.role,
                record.assigned_molecule,
                record.molecule_type,
                record.genbank_accession,
                record.refseq_accession,
                record.assembly_unit,
                record.length,
                record.ucsc_name,
            )
            for item in sorted(selected, key=lambda item: item.candidate.genus)
            for record in item.retained_records
        ),
    )
    write_tsv(
        metadata_dir / "insect_skipped_selected_candidates.tsv",
        ["genus", "accession", "source", "assembly_status", "status", "notes"],
        (
            (
                item.candidate.genus,
                item.candidate.accession,
                item.candidate.source,
                item.candidate.assembly_status,
                item.status,
                item.notes,
            )
            for item in skipped
        ),
    )


def validate_outputs(insect_root: Path, prepared: list[PreparedAssembly], threads: int) -> None:
    metadata_dir = insect_root / "metadata"
    selected = [item for item in prepared if item.status == "selected"]
    accessions = [item.candidate.accession for item in selected]
    genus_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        genus_counts[item.candidate.genus] += 1
    over_limit = {genus: count for genus, count in genus_counts.items() if count > 1}
    missing_records = [item.candidate.accession for item in selected if len(item.retained_records) < 1]

    workers = max(1, min(threads, 16))
    gzip_failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(validate_gzip, item.filtered_fasta): item for item in selected}
        for future in as_completed(futures):
            path, ok, error = future.result()
            if not ok:
                gzip_failures.append(f"{path}:{error}")

    insect_txt = [line.strip() for line in (insect_root / "insect.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    checks = [
        ("insect_txt_count_matches_selected", "pass" if len(insect_txt) == len(accessions) else "fail", f"insect_txt={len(insect_txt)} selected={len(accessions)}"),
        ("selected_genus_unique", "pass" if not over_limit else "fail", str(over_limit)),
        ("selected_have_chromosome_records", "pass" if not missing_records else "fail", ",".join(missing_records[:20])),
        ("filtered_fasta_gzip_valid", "pass" if not gzip_failures else "fail", ";".join(gzip_failures[:20])),
    ]
    write_tsv(metadata_dir / "insect_preparation_verification.tsv", ["check", "status", "details"], checks)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=RAW_DATA_ROOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--download-workers", type=int, default=0, help="Parallel genus download/filter workers. Default: min(8, threads).")
    parser.add_argument("--max-genera", type=int, default=0, help="Deterministically limit preparation to the first N ranked genera. Default: 0 means all genera.")
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL", "realm-rank-builder@example.com"))
    parser.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY", ""))
    parser.add_argument("--ncbi-batch-size", type=int, default=200)
    parser.add_argument("--ncbi-delay", type=float, default=0.34)
    parser.add_argument("--retmax", type=int, default=100000)
    parser.add_argument("--refresh-candidates", action="store_true", help="Refresh NCBI Assembly candidate metadata instead of reusing metadata/insect_candidates.tsv.")
    parser.add_argument("--force", action="store_true", help="Re-filter chromosome FASTA outputs even if they already exist.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.ncbi_batch_size <= 0:
        raise RuntimeError("--ncbi-batch-size must be positive")
    threads = max(1, args.threads)
    download_workers = args.download_workers if args.download_workers and args.download_workers > 0 else min(8, threads)
    insect_root = args.data_root / "insect"
    for path in (insect_root / "assemblies", insect_root / "raw", insect_root / "assembly_reports", insect_root / "metadata"):
        path.mkdir(parents=True, exist_ok=True)

    metadata_dir = insect_root / "metadata"
    candidate_tsv = metadata_dir / "insect_candidates.tsv"
    if candidate_tsv.exists() and not args.refresh_candidates:
        ranked = load_candidate_metadata(metadata_dir)
        candidate_count = sum(len(rows) for rows in ranked.values())
        log(f"reusing candidate metadata: {candidate_count:,} candidates across {len(ranked):,} genera")
    else:
        uids = query_assembly_uids(args.email, args.api_key, args.ncbi_delay, args.retmax)
        candidates = fetch_assembly_candidates(uids, args.email, args.api_key, args.ncbi_batch_size, args.ncbi_delay)
        ranked = rank_candidates_by_genus(candidates, args.seed)
        write_candidate_metadata(metadata_dir, ranked)
    if args.max_genera and args.max_genera > 0:
        original_genera = len(ranked)
        ranked = dict(list(ranked.items())[: args.max_genera])
        log(f"bounded mode: using first {len(ranked):,}/{original_genera:,} ranked genera")

    prepared: list[PreparedAssembly] = []
    skipped: list[PreparedAssembly] = []
    total_genera = len(ranked)
    log(f"preparing selected assemblies with {download_workers} parallel genus workers")
    if download_workers == 1:
        for index, (genus, genus_candidates) in enumerate(ranked.items(), start=1):
            log(f"selecting genus {index:,}/{total_genera:,}: {genus} ({len(genus_candidates)} candidates)")
            _genus, selected_item, skipped_items = prepare_genus(genus, genus_candidates, insect_root, args.force)
            skipped.extend(skipped_items)
            if selected_item is not None:
                prepared.append(selected_item)
            else:
                log(f"no strict chromosome assembly retained for genus {genus}")
    else:
        with ThreadPoolExecutor(max_workers=download_workers) as executor:
            future_to_genus = {
                executor.submit(prepare_genus, genus, genus_candidates, insect_root, args.force): genus
                for genus, genus_candidates in ranked.items()
            }
            for completed, future in enumerate(as_completed(future_to_genus), start=1):
                genus = future_to_genus[future]
                _genus, selected_item, skipped_items = future.result()
                skipped.extend(skipped_items)
                if selected_item is not None:
                    prepared.append(selected_item)
                    log(
                        f"completed genus {completed:,}/{total_genera:,}: {genus} "
                        f"selected {selected_item.candidate.accession}"
                    )
                else:
                    log(f"completed genus {completed:,}/{total_genera:,}: {genus} no strict chromosome assembly retained")

    write_outputs(insect_root, prepared, skipped)
    validate_outputs(insect_root, prepared, threads)
    log(f"selected {len(prepared):,} strict Insecta assemblies from {len(ranked):,} genera")
    log(f"done: {insect_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
