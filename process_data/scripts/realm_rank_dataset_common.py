#!/usr/bin/env python3
"""Build Realm-Rank v3 train/test FASTA files with leakage controls.

The pipeline is intentionally self-contained: it streams FASTA inputs, uses
NCBI Entrez only for viral taxonomy, delegates sequence-similarity filtering to
BLAST, and stores fragment metadata in SQLite while sampling.

Realm-Rank v3 adds a strict Insecta assembly source while leaving v1/v2 output
directories untouched.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import math
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from Bio import Entrez


SCRIPT_DIR = Path(__file__).resolve().parent
GAMIL_ROOT = Path(os.environ.get("GAMIL_ROOT", SCRIPT_DIR.parents[1])).resolve()
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT", GAMIL_ROOT / "raw_data" / "local_sources")).resolve()
PROCESSED_DATA_ROOT = Path(os.environ.get("PROCESSED_DATA_ROOT", GAMIL_ROOT / "processed_data")).resolve()
MAJOR_REALMS = {"Duplodnaviria", "Monodnaviria", "Riboviria", "Varidnaviria"}
SMALL_REALM_INPUTS = {"Adnaviria", "Ribozyviria"}
REALM_GROUPS = sorted(MAJOR_REALMS | {"SmallRealm"})
NONVIRAL_SOURCES = ("bacteria", "archaea", "fungi", "plasmid", "protozoa", "insect")
ASSEMBLY_SOURCE_ORDER = ("bacteria", "archaea", "fungi", "insect")
ASSEMBLY_SOURCES = set(ASSEMBLY_SOURCE_ORDER)
NUCCORE_SOURCES = {"plasmid", "protozoa"}
GENUS_CAPS = {"bacteria": 2, "insect": 1}
LARGE_DOWNSAMPLE_SOURCES = ("bacteria", "insect")
DEFAULT_SEED = 1729
FASTA_WIDTH = 80
NON_DNA_RE = re.compile(r"[^ACGTN]")


@dataclass
class GenomeRow:
    genome_id: str
    source: str
    supergroup: str
    label: str
    path: str
    unit: str
    original_id: str
    description: str
    genus: str = ""
    selected: int = 1
    split: str = ""
    total_bp: int = 0
    sequence_count: int = 0
    accession: str = ""
    taxid: str = ""
    scientific_name: str = ""
    realm: str = ""
    query_status: str = ""
    notes: str = ""


def log(message: str) -> None:
    print(f"[realm-rank] {message}", flush=True)


def open_text(path: Path, mode: str = "rt"):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def iter_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    header: str | None = None
    seq_parts: list[str] = []
    with open_text(path, "rt") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = normalize_dna("".join(seq_parts))
                    yield header.split()[0], header, seq
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            seq = normalize_dna("".join(seq_parts))
            yield header.split()[0], header, seq


def normalize_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return NON_DNA_RE.sub("N", seq)


def first_fasta_header(path: Path) -> str:
    with open_text(path, "rt") as handle:
        for raw_line in handle:
            if raw_line.startswith(">"):
                return raw_line[1:].strip()
    return ""


def write_fasta(handle, record_id: str, seq: str, description: str = "") -> None:
    header = record_id if not description else f"{record_id} {description}"
    handle.write(f">{header}\n")
    for start in range(0, len(seq), FASTA_WIDTH):
        handle.write(seq[start : start + FASTA_WIDTH] + "\n")


def safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("_")
    return safe or "unknown"


def strip_fasta_suffix(path: Path) -> str:
    name = path.name
    for suffix in (".fasta.gz", ".fna.gz", ".fa.gz", ".fasta", ".fna", ".fa"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def assembly_accession_from_stem(stem: str) -> str:
    match = re.match(r"^(GC[AF]_\d+\.\d+)", stem)
    return match.group(1) if match else stem


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


def extract_genus(header: str) -> str:
    tokens = header.split()
    if len(tokens) < 2:
        return "unknown"
    words = tokens[1:]
    first = re.sub(r"[^A-Za-z0-9_.-]", "", words[0])
    if not first:
        return "unknown"
    if first.lower() in {"candidatus", "candidate"} and len(words) > 1:
        second = re.sub(r"[^A-Za-z0-9_.-]", "", words[1])
        return safe_id(f"{first}_{second}") if second else safe_id(first)
    return safe_id(first)


def stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def stable_hash_hex(*parts: object, n: int = 20) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:n]


def length_bin(length: int, min_len: int, bin_width: int) -> int:
    return min_len + ((length - min_len) // bin_width) * bin_width


def resolve_threads(value: int | None) -> int:
    if value and value > 0:
        return value
    cpus = os.cpu_count() or 1
    return max(1, int(round(cpus * 0.85)))


def resolve_tool(name: str, explicit: str | None = None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    found = shutil.which(name)
    if found:
        candidates.append(found)
    blast_env = Path.home() / "software/miniconda3/envs/blast/bin" / name
    candidates.append(str(blast_env))
    for candidate in candidates:
        if candidate and Path(candidate).exists() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(f"Could not find executable for {name}; pass --{name} PATH")


def run_command(cmd: list[str], stdout_path: Path | None = None) -> None:
    log("running: " + " ".join(cmd))
    if stdout_path is None:
        subprocess.run(cmd, check=True)
    else:
        with open(stdout_path, "wt") as out:
            subprocess.run(cmd, stdout=out, check=True)


def ensure_dirs(out_dir: Path, force: bool) -> tuple[Path, Path, Path]:
    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    if out_dir.exists() and (out_dir / "train.fasta.gz").exists() and not force:
        raise RuntimeError(f"{out_dir} already contains final output; use --force to rebuild")
    meta_dir = out_dir / "metadata"
    qc_dir = out_dir / "qc"
    work_dir = out_dir / "work"
    for path in (meta_dir, qc_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)
    return meta_dir, qc_dir, work_dir


def discover_files(input_root: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    files["virus"] = sorted((input_root / "virus/nuccore").glob("*.fasta.gz"))
    files["bacteria"] = sorted((input_root / "bacteria/assemblies").glob("*.fna.gz"))
    files["archaea"] = sorted((input_root / "archaea/assemblies").glob("*.fna.gz"))
    files["fungi"] = sorted((input_root / "fungi/assemblies").glob("*.fna.gz"))
    files["insect"] = sorted((input_root / "insect/assemblies").glob("*.fna.gz"))
    files["plasmid"] = sorted((input_root / "plasmid/nuccore").glob("*.fasta.gz"))
    files["protozoa"] = sorted((input_root / "protozoa/nuccore").glob("*.fasta.gz"))
    missing = [source for source, paths in files.items() if not paths]
    if missing:
        raise RuntimeError(f"Missing input files for: {', '.join(missing)}")
    return files


def validate_gzip_file(path: Path) -> tuple[Path, bool, str]:
    if not str(path).endswith(".gz"):
        return path, True, ""
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return path, True, ""
    except Exception as exc:  # noqa: BLE001 - validation should report all file-level issues
        return path, False, str(exc)


def validate_input_files(files: dict[str, list[Path]], meta_dir: Path, threads: int) -> dict[str, list[Path]]:
    all_files = [(source, path) for source, paths in files.items() for path in paths]
    workers = max(1, min(threads, 16))
    source_by_path = {path: source for source, path in all_files}
    invalid_rows: list[tuple[str, str, str]] = []
    valid_by_source: dict[str, list[Path]] = {source: [] for source in files}
    log(f"validating gzip integrity for {len(all_files)} inputs with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_path = {executor.submit(validate_gzip_file, path): path for _source, path in all_files}
        for future in as_completed(future_to_path):
            path, ok, error = future.result()
            source = source_by_path[path]
            if ok:
                valid_by_source[source].append(path)
            else:
                invalid_rows.append((source, path, error))
    for source in valid_by_source:
        valid_by_source[source].sort()
    write_tsv(meta_dir / "input_invalid_gzip.tsv", ["source", "path", "error"], invalid_rows)
    if invalid_rows:
        log(f"excluded {len(invalid_rows)} corrupt gzip inputs; see {meta_dir / 'input_invalid_gzip.tsv'}")
    missing = [source for source, paths in valid_by_source.items() if not paths]
    if missing:
        raise RuntimeError(f"No valid input files remain for: {', '.join(missing)}")
    return valid_by_source


def build_virus_fasta(files: list[Path], work_dir: Path) -> tuple[list[GenomeRow], Path, Path]:
    virus_fasta = work_dir / "virus_all.fasta"
    contigs_tsv = work_dir / "virus_contigs.tsv"
    rows: list[GenomeRow] = []
    seen: set[str] = set()
    with open(virus_fasta, "wt") as fasta, open(contigs_tsv, "wt") as contigs:
        contigs.write("contig_id\tgenome_id\taccession\tpath\toriginal_id\tdescription\tlength\n")
        for path in files:
            for record_id, header, seq in iter_fasta(path):
                accession = accession_from_header_id(record_id)
                if accession in seen:
                    raise RuntimeError(f"Duplicate viral accession found: {accession}")
                seen.add(accession)
                genome_id = f"virus__{safe_id(accession)}"
                contig_id = genome_id
                description = header[len(record_id) :].strip()
                rows.append(
                    GenomeRow(
                        genome_id=genome_id,
                        source="virus",
                        supergroup="virus",
                        label="SmallRealm",
                        path=str(path),
                        unit="nuccore_record",
                        original_id=record_id,
                        description=description,
                        selected=1,
                        total_bp=len(seq),
                        sequence_count=1,
                        accession=accession,
                    )
                )
                write_fasta(fasta, contig_id, seq)
                contigs.write(
                    f"{contig_id}\t{genome_id}\t{accession}\t{path}\t{record_id}\t"
                    f"{description}\t{len(seq)}\n"
                )
    return rows, virus_fasta, contigs_tsv


def read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, "rt") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        return [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]


def write_tsv(path: Path, header: list[str], rows: Iterable[Iterable[object]]) -> None:
    with open(path, "wt") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(value) for value in row) + "\n")


def entrez_read_with_retries(callable_obj, label: str, retries: int = 4):
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with callable_obj() as handle:
                return Entrez.read(handle)
        except Exception as exc:  # noqa: BLE001 - preserve retry detail for Entrez/network errors
            last_error = exc
            sleep_for = min(30, 2**attempt)
            log(f"{label} failed on attempt {attempt}/{retries}: {exc}; retrying in {sleep_for}s")
            time.sleep(sleep_for)
    raise RuntimeError(f"{label} failed after {retries} attempts: {last_error}")


def load_accession_taxid_cache(path: Path) -> dict[str, tuple[str, str, str]]:
    cache: dict[str, tuple[str, str, str]] = {}
    if not path.exists():
        return cache
    for row in read_tsv(path):
        cache[row["accession"]] = (row.get("taxid", ""), row.get("title", ""), row.get("status", ""))
    return cache


def write_accession_taxid_cache(path: Path, cache: dict[str, tuple[str, str, str]]) -> None:
    rows = [(acc, taxid, title, status) for acc, (taxid, title, status) in sorted(cache.items())]
    write_tsv(path, ["accession", "taxid", "title", "status"], rows)


def fetch_accession_taxids(
    accessions: list[str], cache_path: Path, batch_size: int, delay: float
) -> dict[str, tuple[str, str, str]]:
    cache = load_accession_taxid_cache(cache_path)
    missing = [acc for acc in accessions if acc not in cache]
    if not missing:
        return cache
    log(f"querying NCBI nuccore summaries for {len(missing)} viral accessions")
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]

        def call():
            return Entrez.esummary(db="nuccore", id=",".join(batch), retmode="xml")

        try:
            docs = entrez_read_with_retries(call, f"nuccore esummary {start + 1}-{start + len(batch)}")
            returned: set[str] = set()
            for doc in docs:
                accession = str(doc.get("AccessionVersion", ""))
                taxid = str(int(doc["TaxId"])) if doc.get("TaxId") else ""
                title = str(doc.get("Title", ""))
                if accession:
                    cache[accession] = (taxid, title, "ok" if taxid else "missing_taxid")
                    returned.add(accession)
            for accession in batch:
                if accession not in returned and accession not in cache:
                    cache[accession] = ("", "", "missing_esummary")
        except Exception as exc:  # noqa: BLE001 - missing accessions are mapped to SmallRealm later
            log(f"nuccore esummary batch failed, marking {len(batch)} accessions missing: {exc}")
            for accession in batch:
                cache.setdefault(accession, ("", "", "esummary_failed"))
        write_accession_taxid_cache(cache_path, cache)
        time.sleep(delay)
    return cache


def load_taxonomy_cache(path: Path) -> dict[str, tuple[str, str, str, str, str]]:
    cache: dict[str, tuple[str, str, str, str, str]] = {}
    if not path.exists():
        return cache
    for row in read_tsv(path):
        cache[row["taxid"]] = (
            row.get("scientific_name", ""),
            row.get("lineage", ""),
            row.get("realm", ""),
            row.get("realm_group", ""),
            row.get("status", ""),
        )
    return cache


def write_taxonomy_cache(path: Path, cache: dict[str, tuple[str, str, str, str, str]]) -> None:
    rows = [
        (taxid, scientific_name, lineage, realm, realm_group, status)
        for taxid, (scientific_name, lineage, realm, realm_group, status) in sorted(cache.items())
    ]
    write_tsv(path, ["taxid", "scientific_name", "lineage", "realm", "realm_group", "status"], rows)


def map_realm_group(realm: str, lineage: str) -> str:
    if realm in MAJOR_REALMS:
        return realm
    if realm in SMALL_REALM_INPUTS or not realm:
        return "SmallRealm"
    for major in MAJOR_REALMS:
        if major in lineage:
            return major
    return "SmallRealm"


def fetch_taxonomy(
    taxids: list[str], cache_path: Path, batch_size: int, delay: float
) -> dict[str, tuple[str, str, str, str, str]]:
    cache = load_taxonomy_cache(cache_path)
    missing = [taxid for taxid in taxids if taxid and taxid not in cache]
    if not missing:
        return cache
    log(f"querying NCBI taxonomy for {len(missing)} taxids")
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]

        def call():
            return Entrez.efetch(db="taxonomy", id=",".join(batch), retmode="xml")

        try:
            docs = entrez_read_with_retries(call, f"taxonomy efetch {start + 1}-{start + len(batch)}")
            returned: set[str] = set()
            for doc in docs:
                taxid = str(doc.get("TaxId", ""))
                scientific_name = str(doc.get("ScientificName", ""))
                lineage = str(doc.get("Lineage", ""))
                realm = ""
                for item in doc.get("LineageEx", []):
                    if str(item.get("Rank", "")) == "realm":
                        realm = str(item.get("ScientificName", ""))
                        break
                realm_group = map_realm_group(realm, lineage)
                if taxid:
                    cache[taxid] = (scientific_name, lineage, realm, realm_group, "ok")
                    returned.add(taxid)
            for taxid in batch:
                if taxid not in returned and taxid not in cache:
                    cache[taxid] = ("", "", "", "SmallRealm", "missing_taxonomy")
        except Exception as exc:  # noqa: BLE001
            log(f"taxonomy efetch batch failed, marking {len(batch)} taxids SmallRealm: {exc}")
            for taxid in batch:
                cache.setdefault(taxid, ("", "", "", "SmallRealm", "taxonomy_failed"))
        write_taxonomy_cache(cache_path, cache)
        time.sleep(delay)
    return cache


def assign_viral_realms(
    virus_rows: list[GenomeRow],
    meta_dir: Path,
    work_dir: Path,
    email: str,
    api_key: str,
    batch_size: int,
    delay: float,
) -> None:
    Entrez.email = email
    Entrez.tool = "realm_rank_dataset_builder"
    if api_key:
        Entrez.api_key = api_key
    accessions = [row.accession for row in virus_rows]
    accession_cache = fetch_accession_taxids(accessions, work_dir / "nuccore_accession_taxid.tsv", batch_size, delay)
    taxids = sorted({taxid for taxid, _, _ in accession_cache.values() if taxid})
    taxonomy_cache = fetch_taxonomy(taxids, work_dir / "taxonomy_lineage.tsv", batch_size, delay)
    for row in virus_rows:
        taxid, title, status = accession_cache.get(row.accession, ("", "", "missing_esummary"))
        scientific_name, lineage, realm, realm_group, tax_status = taxonomy_cache.get(
            taxid, ("", "", "", "SmallRealm", "missing_taxid")
        )
        row.taxid = taxid
        row.scientific_name = scientific_name
        row.realm = realm
        row.label = realm_group
        row.query_status = status if status != "ok" else tax_status
        row.notes = title
    write_virus_metadata(meta_dir / "virus_accession_realm.tsv", virus_rows)


def write_virus_metadata(path: Path, rows: list[GenomeRow]) -> None:
    header = [
        "accession",
        "genome_id",
        "path",
        "description",
        "length",
        "taxid",
        "scientific_name",
        "realm",
        "realm_group",
        "query_status",
    ]
    write_tsv(
        path,
        header,
        (
            (
                row.accession,
                row.genome_id,
                row.path,
                row.description,
                row.total_bp,
                row.taxid,
                row.scientific_name,
                row.realm,
                row.label,
                row.query_status,
            )
            for row in rows
        ),
    )


def discover_nonviral_genomes(files: dict[str, list[Path]], seed: int) -> list[GenomeRow]:
    rows: list[GenomeRow] = []
    capped_by_source_genus: dict[tuple[str, str], list[GenomeRow]] = defaultdict(list)
    for source in ASSEMBLY_SOURCE_ORDER:
        for path in files[source]:
            header = first_fasta_header(path)
            stem = strip_fasta_suffix(path)
            if stem.endswith("_genomic"):
                stem = stem[: -len("_genomic")]
            accession = assembly_accession_from_stem(stem)
            original_id = accession if source == "insect" else stem
            genome_key = accession if source == "insect" else stem
            genome_id = f"{source}__{safe_id(genome_key)}"
            genus = extract_genus(header)
            row = GenomeRow(
                genome_id=genome_id,
                source=source,
                supergroup="nonvirus",
                label=source,
                path=str(path),
                unit="assembly_file",
                original_id=original_id,
                description=header,
                genus=genus,
                selected=0 if source in GENUS_CAPS else 1,
                accession=accession,
            )
            rows.append(row)
            if source in GENUS_CAPS:
                capped_by_source_genus[(source, genus)].append(row)
    for (source, genus), genus_rows in sorted(capped_by_source_genus.items()):
        ordered = sorted(genus_rows, key=lambda row: row.genome_id)
        rng = random.Random(stable_int(seed, f"{source}-genus", genus))
        rng.shuffle(ordered)
        for row in ordered[: GENUS_CAPS[source]]:
            row.selected = 1
    for source in ("plasmid", "protozoa"):
        for path in files[source]:
            for record_id, header, seq in iter_fasta(path):
                accession = accession_from_header_id(record_id)
                rows.append(
                    GenomeRow(
                        genome_id=f"{source}__{safe_id(accession)}",
                        source=source,
                        supergroup="nonvirus",
                        label=source,
                        path=str(path),
                        unit="nuccore_record",
                        original_id=record_id,
                        description=header[len(record_id) :].strip(),
                        genus=extract_genus(header),
                        selected=1,
                        total_bp=len(seq),
                        sequence_count=1,
                        accession=accession,
                    )
                )
    return rows


def split_genomes(virus_rows: list[GenomeRow], nonviral_rows: list[GenomeRow], seed: int, test_fraction: float) -> None:
    selected = [row for row in virus_rows] + [row for row in nonviral_rows if row.selected]
    by_label: dict[str, list[GenomeRow]] = defaultdict(list)
    for row in selected:
        by_label[row.label].append(row)
    for label, rows in sorted(by_label.items()):
        ordered = sorted(rows, key=lambda row: row.genome_id)
        rng = random.Random(stable_int(seed, "split", label))
        rng.shuffle(ordered)
        if len(ordered) <= 1:
            test_count = 0
        else:
            test_count = max(1, int(round(len(ordered) * test_fraction)))
            test_count = min(test_count, len(ordered) - 1)
        test_ids = {row.genome_id for row in ordered[:test_count]}
        for row in rows:
            row.split = "test" if row.genome_id in test_ids else "train"
    for row in nonviral_rows:
        if not row.selected:
            row.split = "excluded"


def write_genome_split(meta_dir: Path, virus_rows: list[GenomeRow], nonviral_rows: list[GenomeRow]) -> None:
    header = ["genome_id", "source", "supergroup", "label", "split", "selected", "genus", "path"]
    selected_rows = [row for row in virus_rows] + [row for row in nonviral_rows if row.selected]
    write_tsv(
        meta_dir / "genome_split.tsv",
        header,
        (
            (row.genome_id, row.source, row.supergroup, row.label, row.split, row.selected, row.genus, row.path)
            for row in sorted(selected_rows, key=lambda r: r.genome_id)
        ),
    )


def write_nonviral_metadata(meta_dir: Path, rows: list[GenomeRow]) -> None:
    header = [
        "genome_id",
        "source",
        "label",
        "split",
        "selected",
        "genus",
        "unit",
        "path",
        "original_id",
        "accession",
        "description",
        "sequence_count",
        "total_bp",
    ]
    write_tsv(
        meta_dir / "nonviral_genomes.tsv",
        header,
        (
            (
                row.genome_id,
                row.source,
                row.label,
                row.split,
                row.selected,
                row.genus,
                row.unit,
                row.path,
                row.original_id,
                row.accession,
                row.description,
                row.sequence_count,
                row.total_bp,
            )
            for row in sorted(rows, key=lambda r: (r.source, r.genome_id))
        ),
    )


def unique_contig_id(base: str, seen: dict[str, int]) -> str:
    safe = safe_id(base)
    count = seen[safe]
    seen[safe] += 1
    if count == 0:
        return safe
    return f"{safe}__dup{count}"


def write_selected_nonviral_fasta(
    files: dict[str, list[Path]], rows: list[GenomeRow], work_dir: Path, meta_dir: Path
) -> tuple[Path, Path]:
    selected_by_id = {row.genome_id: row for row in rows if row.selected}
    assembly_paths = {Path(row.path): row for row in selected_by_id.values() if row.unit == "assembly_file"}
    nuccore_by_path: dict[Path, set[str]] = defaultdict(set)
    for row in selected_by_id.values():
        if row.unit == "nuccore_record":
            nuccore_by_path[Path(row.path)].add(row.genome_id)
    nonviral_fasta = work_dir / "nonviral_selected.fasta"
    contigs_tsv = work_dir / "nonviral_contigs.tsv"
    seen_contigs: dict[str, int] = defaultdict(int)
    with open(nonviral_fasta, "wt") as fasta, open(contigs_tsv, "wt") as contigs:
        contigs.write("contig_id\tgenome_id\tsource\tpath\toriginal_id\tdescription\toriginal_length\n")
        for source in ASSEMBLY_SOURCE_ORDER:
            for path in files[source]:
                row = assembly_paths.get(path)
                if row is None:
                    continue
                sequence_count = 0
                total_bp = 0
                for record_id, header, seq in iter_fasta(path):
                    accession = accession_from_header_id(record_id)
                    contig_id = unique_contig_id(f"{row.genome_id}__{accession}", seen_contigs)
                    description = header[len(record_id) :].strip()
                    write_fasta(fasta, contig_id, seq)
                    contigs.write(
                        f"{contig_id}\t{row.genome_id}\t{row.source}\t{path}\t{record_id}\t"
                        f"{description}\t{len(seq)}\n"
                    )
                    sequence_count += 1
                    total_bp += len(seq)
                row.sequence_count = sequence_count
                row.total_bp = total_bp
        for source in ("plasmid", "protozoa"):
            for path in files[source]:
                wanted = nuccore_by_path.get(path)
                if not wanted:
                    continue
                for record_id, header, seq in iter_fasta(path):
                    accession = accession_from_header_id(record_id)
                    genome_id = f"{source}__{safe_id(accession)}"
                    row = selected_by_id.get(genome_id)
                    if row is None:
                        continue
                    contig_id = unique_contig_id(genome_id, seen_contigs)
                    description = header[len(record_id) :].strip()
                    write_fasta(fasta, contig_id, seq)
                    contigs.write(
                        f"{contig_id}\t{genome_id}\t{source}\t{path}\t{record_id}\t"
                        f"{description}\t{len(seq)}\n"
                    )
    write_nonviral_metadata(meta_dir, rows)
    return nonviral_fasta, contigs_tsv


def make_blast_db(makeblastdb: str, fasta: Path, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [makeblastdb, "-in", str(fasta), "-dbtype", "nucl", "-out", str(out_prefix)]
    run_command(cmd)


def run_blastn(blastn: str, query: Path, db_prefix: Path, out_tsv: Path, threads: int, evalue: str = "1e-10") -> None:
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
        evalue,
        "-num_threads",
        str(threads),
    ]
    run_command(cmd)


def merge_intervals(intervals: list[tuple[int, int]], qlen: int | None = None) -> list[tuple[int, int]]:
    if not intervals:
        return []
    normalized = []
    for start, end in intervals:
        lo, hi = sorted((int(start), int(end)))
        if qlen is not None:
            lo = max(1, min(lo, qlen))
            hi = max(1, min(hi, qlen))
        if lo <= hi:
            normalized.append((lo, hi))
    normalized.sort()
    merged: list[tuple[int, int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_coverage(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


def build_decontam_intervals(blast_tsv: Path, intervals_tsv: Path, pident_min: float, hsp_qcov_min: float) -> dict[str, list[tuple[int, int]]]:
    raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
    qlens: dict[str, int] = {}
    if blast_tsv.exists():
        with open(blast_tsv, "rt") as handle:
            for line in handle:
                if not line.strip():
                    continue
                qseqid, qstart, qend, pident, length, qlen, evalue, _bitscore = line.rstrip("\n").split("\t")
                qlen_i = int(qlen)
                qlens[qseqid] = qlen_i
                hsp_qcov = int(length) / qlen_i if qlen_i else 0.0
                if float(pident) >= pident_min and float(evalue) <= 1e-10 and hsp_qcov >= hsp_qcov_min:
                    raw[qseqid].append((int(qstart), int(qend)))
    merged: dict[str, list[tuple[int, int]]] = {}
    with open(intervals_tsv, "wt") as out:
        out.write("contig_id\tstart\tend\tqlen\n")
        for contig_id, intervals in sorted(raw.items()):
            merged_intervals = merge_intervals(intervals, qlens.get(contig_id))
            merged[contig_id] = merged_intervals
            for start, end in merged_intervals:
                out.write(f"{contig_id}\t{start}\t{end}\t{qlens.get(contig_id, '')}\n")
    return merged


def load_intervals(intervals_tsv: Path) -> dict[str, list[tuple[int, int]]]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    if not intervals_tsv.exists():
        return intervals
    with open(intervals_tsv, "rt") as handle:
        next(handle, None)
        for line in handle:
            contig_id, start, end, _qlen = line.rstrip("\n").split("\t")
            intervals[contig_id].append((int(start), int(end)))
    return {contig: merge_intervals(vals) for contig, vals in intervals.items()}


def clean_sequence(seq: str, intervals: list[tuple[int, int]]) -> tuple[str, int]:
    if not intervals:
        return seq, 0
    chunks: list[str] = []
    cursor = 1
    deleted = 0
    seq_len = len(seq)
    for start, end in merge_intervals(intervals, seq_len):
        if cursor < start:
            chunks.append(seq[cursor - 1 : start - 1])
        deleted += end - start + 1
        cursor = end + 1
    if cursor <= seq_len:
        chunks.append(seq[cursor - 1 :])
    return "".join(chunks), deleted


def decontaminate_nonviral_fasta(
    nonviral_fasta: Path,
    contigs_tsv: Path,
    intervals: dict[str, list[tuple[int, int]]],
    work_dir: Path,
    meta_dir: Path,
) -> Path:
    cleaned_fasta = work_dir / "nonviral_cleaned.fasta"
    contig_to_genome: dict[str, tuple[str, str]] = {}
    for row in read_tsv(contigs_tsv):
        contig_to_genome[row["contig_id"]] = (row["genome_id"], row["source"])
    genome_stats: dict[str, dict[str, object]] = defaultdict(
        lambda: {"source": "", "contig_count": 0, "original_bp": 0, "deleted_bp": 0, "cleaned_bp": 0}
    )
    contig_rows: list[tuple[object, ...]] = []
    with open(cleaned_fasta, "wt") as out:
        for contig_id, _header, seq in iter_fasta(nonviral_fasta):
            genome_id, source = contig_to_genome[contig_id]
            cleaned, deleted = clean_sequence(seq, intervals.get(contig_id, []))
            if cleaned:
                write_fasta(out, contig_id, cleaned)
            stats = genome_stats[genome_id]
            stats["source"] = source
            stats["contig_count"] = int(stats["contig_count"]) + 1
            stats["original_bp"] = int(stats["original_bp"]) + len(seq)
            stats["deleted_bp"] = int(stats["deleted_bp"]) + deleted
            stats["cleaned_bp"] = int(stats["cleaned_bp"]) + len(cleaned)
            contig_rows.append((contig_id, genome_id, source, len(seq), deleted, len(cleaned), deleted / len(seq) if seq else 0))
    write_tsv(
        meta_dir / "nonviral_decontamination_contigs.tsv",
        ["contig_id", "genome_id", "source", "original_bp", "deleted_bp", "cleaned_bp", "deleted_fraction"],
        contig_rows,
    )
    genome_rows = []
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
        meta_dir / "nonviral_decontamination.tsv",
        ["genome_id", "source", "contig_count", "original_bp", "deleted_bp", "cleaned_bp", "deleted_fraction"],
        genome_rows,
    )
    return cleaned_fasta


def load_genome_split(meta_dir: Path) -> dict[str, tuple[str, str, str, str]]:
    mapping: dict[str, tuple[str, str, str, str]] = {}
    for row in read_tsv(meta_dir / "genome_split.tsv"):
        mapping[row["genome_id"]] = (row["source"], row["supergroup"], row["label"], row["split"])
    return mapping


def load_contig_info(contigs_tsv: Path, genome_split: dict[str, tuple[str, str, str, str]], viral: bool) -> dict[str, dict[str, str]]:
    info: dict[str, dict[str, str]] = {}
    for row in read_tsv(contigs_tsv):
        genome_id = row["genome_id"]
        source, supergroup, label, split = genome_split[genome_id]
        contig_id = row["contig_id"]
        info[contig_id] = {
            "genome_id": genome_id,
            "source": source,
            "supergroup": supergroup,
            "label": label,
            "split": split,
        }
    return info


def init_fragment_db(db_path: Path) -> sqlite3.Connection:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        """
        CREATE TABLE fragments (
            fragment_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            label TEXT NOT NULL,
            supergroup TEXT NOT NULL,
            source TEXT NOT NULL,
            genome_id TEXT NOT NULL,
            contig_id TEXT NOT NULL,
            start INTEGER NOT NULL,
            end INTEGER NOT NULL,
            length INTEGER NOT NULL,
            length_bin INTEGER NOT NULL,
            selected INTEGER NOT NULL DEFAULT 1,
            filter_reason TEXT NOT NULL DEFAULT '',
            removed_by_train_blast INTEGER NOT NULL DEFAULT 0,
            train_coverage REAL
        )
        """
    )
    return conn


def create_fragment_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_frag_selected_split ON fragments(selected, split)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_frag_group ON fragments(selected, split, label)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_frag_super ON fragments(selected, split, supergroup)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_frag_source ON fragments(selected, split, source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_frag_lenbin ON fragments(selected, split, supergroup, length_bin)")
    conn.commit()


def iter_fragment_coords(seq_len: int, min_len: int, max_len: int, seed: int, contig_id: str) -> Iterator[tuple[int, int, int]]:
    if seq_len < min_len:
        return
    rng = random.Random(stable_int(seed, "fragment", contig_id))
    pos = 0
    while pos + min_len <= seq_len:
        remaining = seq_len - pos
        upper = min(max_len, remaining)
        if upper < min_len:
            break
        frag_len = rng.randint(min_len, upper)
        start = pos + 1
        end = pos + frag_len
        yield start, end, frag_len
        pos += frag_len


def insert_fragments_for_fasta(
    conn: sqlite3.Connection,
    fasta_path: Path,
    contig_info: dict[str, dict[str, str]],
    min_len: int,
    max_len: int,
    bin_width: int,
    seed: int,
) -> int:
    batch: list[tuple[object, ...]] = []
    count = 0
    sql = """
        INSERT INTO fragments (
            fragment_id, split, label, supergroup, source, genome_id, contig_id,
            start, end, length, length_bin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with conn:
        for contig_id, _header, seq in iter_fasta(fasta_path):
            info = contig_info.get(contig_id)
            if info is None:
                raise RuntimeError(f"Missing contig metadata for {contig_id}")
            for start, end, frag_len in iter_fragment_coords(len(seq), min_len, max_len, seed, contig_id):
                fragment_id = "frag_" + stable_hash_hex(contig_id, start, end, frag_len)
                batch.append(
                    (
                        fragment_id,
                        info["split"],
                        info["label"],
                        info["supergroup"],
                        info["source"],
                        info["genome_id"],
                        contig_id,
                        start,
                        end,
                        frag_len,
                        length_bin(frag_len, min_len, bin_width),
                    )
                )
                count += 1
                if len(batch) >= 10000:
                    conn.executemany(sql, batch)
                    batch = []
        if batch:
            conn.executemany(sql, batch)
    return count


def selected_bp_by_split(conn: sqlite3.Connection, where: str, params: tuple[object, ...] = ()) -> dict[str, int]:
    query = f"SELECT split, COALESCE(SUM(length), 0) FROM fragments WHERE selected=1 AND {where} GROUP BY split"
    return {split: int(bp) for split, bp in conn.execute(query, params)}


def stable_row_order(seed: int, fragment_id: str, purpose: str) -> int:
    return stable_int(seed, purpose, fragment_id)


def update_unselected(conn: sqlite3.Connection, fragment_ids: list[str], reason: str) -> None:
    if not fragment_ids:
        return
    with conn:
        conn.executemany(
            "UPDATE fragments SET selected=0, filter_reason=? WHERE fragment_id=?",
            [(reason, fragment_id) for fragment_id in fragment_ids],
        )


def downsample_large_sources_to_virus_bp(conn: sqlite3.Connection, seed: int, qc_dir: Path) -> None:
    virus_bp = selected_bp_by_split(conn, "supergroup='virus'")
    rows_out: list[tuple[object, ...]] = []
    for source in LARGE_DOWNSAMPLE_SOURCES:
        for split in ("train", "test"):
            target_bp = virus_bp.get(split, 0)
            rows = list(
                conn.execute(
                    """
                    SELECT fragment_id, length FROM fragments
                    WHERE selected=1 AND source=? AND split=?
                    """,
                    (source, split),
                )
            )
            before_bp = sum(int(length) for _fragment_id, length in rows)
            if before_bp <= target_bp or target_bp <= 0:
                rows_out.append((source, split, before_bp, target_bp, before_bp, 0, "kept_all"))
                continue
            rows.sort(key=lambda row: stable_row_order(seed, row[0], f"{source}-{split}"))
            keep: set[str] = set()
            kept_bp = 0
            for fragment_id, length in rows:
                length = int(length)
                if kept_bp + length <= target_bp:
                    keep.add(fragment_id)
                    kept_bp += length
            drop = [fragment_id for fragment_id, _length in rows if fragment_id not in keep]
            update_unselected(conn, drop, f"{source}_bp_downsample")
            rows_out.append((source, split, before_bp, target_bp, kept_bp, len(drop), "downsampled"))
    write_tsv(
        qc_dir / "large_source_downsample.tsv",
        ["source", "split", "source_bp_before", "virus_bp_target", "source_bp_after", "fragments_dropped", "status"],
        rows_out,
    )


def bp_by_length_bin(conn: sqlite3.Connection, split: str, supergroup: str) -> dict[int, int]:
    return {
        int(bin_start): int(total_bp or 0)
        for bin_start, total_bp in conn.execute(
            """
            SELECT length_bin, SUM(length) FROM fragments
            WHERE selected=1 AND split=? AND supergroup=?
            GROUP BY length_bin
            """,
            (split, supergroup),
        )
    }


def source_bp_by_length_bin(conn: sqlite3.Connection, split: str) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = defaultdict(dict)
    for bin_start, source, total_bp in conn.execute(
        """
        SELECT length_bin, source, SUM(length) FROM fragments
        WHERE selected=1 AND split=? AND supergroup='nonvirus'
        GROUP BY length_bin, source
        """,
        (split,),
    ):
        result[int(bin_start)][str(source)] = int(total_bp or 0)
    return result


def allocate_equal_source_bp_targets(available_by_source: dict[str, int], target_bp: int) -> dict[str, int]:
    if target_bp <= 0 or not available_by_source:
        return {source: 0 for source in NONVIRAL_SOURCES}
    available_total = sum(available_by_source.values())
    if available_total <= target_bp:
        return {source: available_by_source.get(source, 0) for source in NONVIRAL_SOURCES}
    targets = {source: 0 for source in NONVIRAL_SOURCES}
    active = {source for source in NONVIRAL_SOURCES if available_by_source.get(source, 0) > 0}
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


def downsample_nonvirus_to_virus_bp_by_length_bin(conn: sqlite3.Connection, seed: int, qc_dir: Path) -> None:
    """Match nonvirus bp to viral bp within each split, length bin, and source.

    This keeps the final binary classes close to bp-balanced while preserving
    the viral length-bin profile. Within each length bin, the viral bp target is
    distributed as evenly as possible across the six nonviral source categories;
    source/bin deficits are redistributed only within that same bin.
    """

    del seed  # fragment_id is a stable SHA1-derived identifier and supplies reproducible ordering.
    existing = conn.execute(
        "SELECT COUNT(*) FROM fragments WHERE filter_reason='nonvirus_bp_downsample'"
    ).fetchone()[0]
    if existing:
        return
    before_split_bp = selected_bp_by_split(conn, "supergroup='nonvirus'")
    before_bin_bp: dict[tuple[str, int], int] = {}
    for split, bin_start, total_bp in conn.execute(
        """
        SELECT split, length_bin, SUM(length)
        FROM fragments
        WHERE selected=1 AND supergroup='nonvirus'
        GROUP BY split, length_bin
        """
    ):
        before_bin_bp[(str(split), int(bin_start))] = int(total_bp or 0)
    before_source_bp: dict[tuple[str, str], int] = {}
    for split, source, total_bp in conn.execute(
        """
        SELECT split, source, SUM(length)
        FROM fragments
        WHERE selected=1 AND supergroup='nonvirus'
        GROUP BY split, source
        """
    ):
        before_source_bp[(str(split), str(source))] = int(total_bp or 0)
    before_bin_source_bp: dict[tuple[str, int, str], int] = {}
    for split, bin_start, source, total_bp in conn.execute(
        """
        SELECT split, length_bin, source, SUM(length)
        FROM fragments
        WHERE selected=1 AND supergroup='nonvirus'
        GROUP BY split, length_bin, source
        """
    ):
        before_bin_source_bp[(str(split), int(bin_start), str(source))] = int(total_bp or 0)
    target_rows: list[tuple[str, int, str, int]] = []
    for split in ("train", "test"):
        virus_bin_bp = bp_by_length_bin(conn, split, "virus")
        source_bin_bp = source_bp_by_length_bin(conn, split)
        for bin_start in sorted(set(source_bin_bp) | set(virus_bin_bp)):
            bin_target_bp = virus_bin_bp.get(bin_start, 0)
            available_by_source = source_bin_bp.get(bin_start, {})
            source_targets = allocate_equal_source_bp_targets(available_by_source, bin_target_bp)
            target_rows.extend((split, bin_start, source, source_targets.get(source, 0)) for source in NONVIRAL_SOURCES)
    conn.execute("DROP TABLE IF EXISTS temp.nonvirus_bp_targets")
    conn.execute(
        """
        CREATE TEMP TABLE nonvirus_bp_targets (
            split TEXT NOT NULL,
            length_bin INTEGER NOT NULL,
            source TEXT NOT NULL,
            target_bp INTEGER NOT NULL,
            PRIMARY KEY (split, length_bin, source)
        )
        """
    )
    conn.executemany(
        "INSERT INTO nonvirus_bp_targets (split, length_bin, source, target_bp) VALUES (?, ?, ?, ?)",
        target_rows,
    )
    conn.execute("DROP TABLE IF EXISTS temp.nonvirus_keep")
    conn.execute(
        """
        CREATE TEMP TABLE nonvirus_keep AS
        WITH ranked AS (
            SELECT
                f.fragment_id,
                f.split,
                f.length_bin,
                f.source,
                SUM(f.length) OVER (
                    PARTITION BY f.split, f.length_bin, f.source
                    ORDER BY f.fragment_id
                    ROWS UNBOUNDED PRECEDING
                ) AS cumulative_bp
            FROM fragments f
            JOIN nonvirus_bp_targets t
              ON t.split = f.split
             AND t.length_bin = f.length_bin
             AND t.source = f.source
            WHERE f.selected=1
              AND f.supergroup='nonvirus'
              AND t.target_bp > 0
        )
        SELECT r.fragment_id
        FROM ranked r
        JOIN nonvirus_bp_targets t
          ON t.split = r.split
         AND t.length_bin = r.length_bin
         AND t.source = r.source
        WHERE r.cumulative_bp <= t.target_bp
        """
    )
    conn.execute("CREATE UNIQUE INDEX idx_nonvirus_keep_fragment ON nonvirus_keep(fragment_id)")
    with conn:
        conn.execute(
            """
            UPDATE fragments
            SET selected=0, filter_reason='nonvirus_bp_downsample'
            WHERE selected=1
              AND supergroup='nonvirus'
              AND fragment_id NOT IN (SELECT fragment_id FROM nonvirus_keep)
            """
        )
    summary_rows: list[tuple[object, ...]] = []
    bin_rows: list[tuple[object, ...]] = []
    source_rows: list[tuple[object, ...]] = []
    bin_source_rows: list[tuple[object, ...]] = []
    after_split_bp = selected_bp_by_split(conn, "supergroup='nonvirus'")
    virus_split_bp = selected_bp_by_split(conn, "supergroup='virus'")
    dropped_by_split = {
        split: int(count)
        for split, count in conn.execute(
            """
            SELECT split, COUNT(*)
            FROM fragments
            WHERE filter_reason='nonvirus_bp_downsample'
            GROUP BY split
            """
        )
    }
    for split in ("train", "test"):
        before_bp = before_split_bp.get(split, 0)
        target_bp = virus_split_bp.get(split, 0)
        after_bp = after_split_bp.get(split, 0)
        summary_rows.append(
            (
                split,
                before_bp,
                target_bp,
                after_bp,
                before_bp - after_bp,
                dropped_by_split.get(split, 0),
                "downsampled" if before_bp > target_bp else "kept_all",
            )
        )
    after_bin_bp: dict[tuple[str, int], int] = {}
    for split, bin_start, total_bp in conn.execute(
        """
        SELECT split, length_bin, SUM(length)
        FROM fragments
        WHERE selected=1 AND supergroup='nonvirus'
        GROUP BY split, length_bin
        """
    ):
        after_bin_bp[(str(split), int(bin_start))] = int(total_bp or 0)
    after_source_bp: dict[tuple[str, str], int] = {}
    for split, source, total_bp in conn.execute(
        """
        SELECT split, source, SUM(length)
        FROM fragments
        WHERE selected=1 AND supergroup='nonvirus'
        GROUP BY split, source
        """
    ):
        after_source_bp[(str(split), str(source))] = int(total_bp or 0)
    after_bin_source_bp: dict[tuple[str, int, str], int] = {}
    for split, bin_start, source, total_bp in conn.execute(
        """
        SELECT split, length_bin, source, SUM(length)
        FROM fragments
        WHERE selected=1 AND supergroup='nonvirus'
        GROUP BY split, length_bin, source
        """
    ):
        after_bin_source_bp[(str(split), int(bin_start), str(source))] = int(total_bp or 0)
    target_lookup = {(split, bin_start, source): target_bp for split, bin_start, source, target_bp in target_rows}
    for split in ("train", "test"):
        virus_bin_bp = bp_by_length_bin(conn, split, "virus")
        all_bins = sorted({bin_start for s, bin_start in before_bin_bp if s == split} | set(virus_bin_bp))
        for bin_start in all_bins:
            before_bp = before_bin_bp.get((split, bin_start), 0)
            after_bp = after_bin_bp.get((split, bin_start), 0)
            bin_rows.append(
                (
                    split,
                    bin_start,
                    virus_bin_bp.get(bin_start, 0),
                    before_bp,
                    after_bp,
                    before_bp - after_bp,
                    "downsampled" if before_bp > after_bp else "kept_all",
                )
            )
        source_target = virus_split_bp.get(split, 0) / len(NONVIRAL_SOURCES)
        for source in NONVIRAL_SOURCES:
            before_bp = before_source_bp.get((split, source), 0)
            after_bp = after_source_bp.get((split, source), 0)
            source_rows.append(
                (
                    split,
                    source,
                    f"{source_target:.3f}",
                    before_bp,
                    after_bp,
                    before_bp - after_bp,
                    "downsampled" if before_bp > after_bp else "kept_all",
                )
            )
            for bin_start in all_bins:
                before_bp_bin_source = before_bin_source_bp.get((split, bin_start, source), 0)
                after_bp_bin_source = after_bin_source_bp.get((split, bin_start, source), 0)
                target_bp_bin_source = target_lookup.get((split, bin_start, source), 0)
                bin_source_rows.append(
                    (
                        split,
                        bin_start,
                        source,
                        target_bp_bin_source,
                        before_bp_bin_source,
                        after_bp_bin_source,
                        before_bp_bin_source - after_bp_bin_source,
                        "downsampled" if before_bp_bin_source > after_bp_bin_source else "kept_all",
                    )
                )
    write_tsv(
        qc_dir / "nonvirus_downsample.tsv",
        [
            "split",
            "nonvirus_bp_before",
            "virus_bp_target",
            "nonvirus_bp_after",
            "bp_dropped",
            "fragments_dropped",
            "status",
        ],
        summary_rows,
    )
    write_tsv(
        qc_dir / "nonvirus_downsample_by_bin.tsv",
        [
            "split",
            "length_bin",
            "virus_bp_target",
            "nonvirus_bp_before",
            "nonvirus_bp_after",
            "bp_dropped",
            "status",
        ],
        bin_rows,
    )
    write_tsv(
        qc_dir / "nonvirus_downsample_by_source.tsv",
        [
            "split",
            "source",
            "source_bp_target",
            "source_bp_before",
            "source_bp_after",
            "bp_dropped",
            "status",
        ],
        source_rows,
    )
    write_tsv(
        qc_dir / "nonvirus_downsample_by_bin_source.tsv",
        [
            "split",
            "length_bin",
            "source",
            "source_bin_bp_target",
            "source_bin_bp_before",
            "source_bin_bp_after",
            "bp_dropped",
            "status",
        ],
        bin_source_rows,
    )


def length_hist(conn: sqlite3.Connection, split: str, supergroup: str) -> dict[int, int]:
    return {
        int(bin_start): int(count)
        for bin_start, count in conn.execute(
            """
            SELECT length_bin, COUNT(*) FROM fragments
            WHERE selected=1 AND split=? AND supergroup=?
            GROUP BY length_bin
            """,
            (split, supergroup),
        )
    }


def max_distribution_delta(a: dict[int, int], b: dict[int, int]) -> float:
    total_a = sum(a.values())
    total_b = sum(b.values())
    if total_a == 0 or total_b == 0:
        return 0.0
    bins = set(a) | set(b)
    return max(abs(a.get(bin_start, 0) / total_a - b.get(bin_start, 0) / total_b) for bin_start in bins)


def balance_nonvirus_length_distribution(
    conn: sqlite3.Connection, seed: int, qc_dir: Path, threshold: float
) -> None:
    actions: list[tuple[object, ...]] = []
    for split in ("train", "test"):
        virus = length_hist(conn, split, "virus")
        nonvirus = length_hist(conn, split, "nonvirus")
        delta = max_distribution_delta(virus, nonvirus)
        if delta <= threshold or not virus or not nonvirus:
            actions.append((split, delta, 0, "not_needed"))
            continue
        virus_total = sum(virus.values())
        nonvirus_total = sum(nonvirus.values())
        proportions = {bin_start: count / virus_total for bin_start, count in virus.items() if count > 0}
        feasible_totals = [
            math.floor(nonvirus.get(bin_start, 0) / proportion)
            for bin_start, proportion in proportions.items()
            if proportion > 0
        ]
        target_total = min([nonvirus_total] + feasible_totals) if feasible_totals else 0
        if target_total <= 0:
            actions.append((split, delta, 0, "failed_no_feasible_nonvirus_target"))
            continue
        targets = {bin_start: min(nonvirus.get(bin_start, 0), int(round(target_total * prop))) for bin_start, prop in proportions.items()}
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
            rows = [row[0] for row in conn.execute(
                """
                SELECT fragment_id FROM fragments
                WHERE selected=1 AND split=? AND supergroup='nonvirus' AND length_bin=?
                """,
                (split, bin_start),
            )]
            rows.sort(key=lambda fragment_id: stable_row_order(seed, fragment_id, f"length-balance-{split}-{bin_start}"))
            drop = rows[target:]
            update_unselected(conn, drop, "length_bin_downsample")
            dropped += len(drop)
        new_delta = max_distribution_delta(length_hist(conn, split, "virus"), length_hist(conn, split, "nonvirus"))
        actions.append((split, delta, dropped, f"downsampled_new_delta={new_delta:.6g}"))
    write_tsv(
        qc_dir / "length_balance_actions.tsv",
        ["split", "initial_max_abs_bin_delta", "fragments_dropped", "status"],
        actions,
    )


def load_selected_fragment_ids(conn: sqlite3.Connection) -> dict[str, set[str]]:
    result = {"train": set(), "test": set()}
    for split, fragment_id in conn.execute("SELECT split, fragment_id FROM fragments WHERE selected=1"):
        result[split].add(fragment_id)
    return result


def write_selected_fragment_fastas(
    conn: sqlite3.Connection,
    fasta_infos: list[tuple[Path, dict[str, dict[str, str]]]],
    work_dir: Path,
    min_len: int,
    max_len: int,
    seed: int,
) -> tuple[Path, Path]:
    selected = load_selected_fragment_ids(conn)
    train_fasta = work_dir / "train.prefilter.fasta"
    test_fasta = work_dir / "test.prefilter.fasta"
    with open(train_fasta, "wt") as train, open(test_fasta, "wt") as test:
        for fasta_path, contig_info in fasta_infos:
            for contig_id, _header, seq in iter_fasta(fasta_path):
                info = contig_info[contig_id]
                split = info["split"]
                out = train if split == "train" else test
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
                    write_fasta(out, fragment_id, subseq, desc)
    return train_fasta, test_fasta


def parse_train_blast_coverage(
    blast_tsv: Path,
    removed_tsv: Path,
    pident_min: float,
    coverage_threshold: float,
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
            rows.append((qseqid, qlen, covered, coverage, len(merged)))
    write_tsv(
        removed_tsv,
        ["fragment_id", "query_length", "covered_bp", "coverage", "merged_interval_count"],
        rows,
    )
    return removed


def update_removed_by_train_blast(conn: sqlite3.Connection, removed: dict[str, float]) -> None:
    if not removed:
        return
    with conn:
        conn.executemany(
            """
            UPDATE fragments
            SET selected=0, filter_reason='test_train_blast_coverage',
                removed_by_train_blast=1, train_coverage=?
            WHERE fragment_id=?
            """,
            [(coverage, fragment_id) for fragment_id, coverage in removed.items()],
        )


def copy_or_filter_final_fasta(
    prefilter: Path,
    output_gz: Path,
    removed_ids: set[str] | None = None,
    keep_ids: set[str] | None = None,
    threads: int = 1,
    compression_level: int = 1,
) -> None:
    removed_ids = removed_ids or set()
    pigz = shutil.which("pigz")
    tmp_path = output_gz.with_suffix(output_gz.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
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
                for raw_line in src:
                    if raw_line.startswith(b">"):
                        fragment_id = raw_line[1:].split(None, 1)[0].decode("ascii")
                        keep = (keep_ids is None or fragment_id in keep_ids) and fragment_id not in removed_ids
                    if keep:
                        proc.stdin.write(raw_line)
            finally:
                proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise subprocess.CalledProcessError(rc, proc.args)
        tmp_path.replace(output_gz)
        return
    with open(prefilter, "rt") as src, gzip.open(tmp_path, "wt", compresslevel=compression_level) as dst:
        keep = True
        for raw_line in src:
            if raw_line.startswith(">"):
                fragment_id = raw_line[1:].split()[0]
                keep = (keep_ids is None or fragment_id in keep_ids) and fragment_id not in removed_ids
            if keep:
                dst.write(raw_line)
    tmp_path.replace(output_gz)


def dump_final_fragments(conn: sqlite3.Connection, meta_dir: Path) -> None:
    header = [
        "fragment_id",
        "split",
        "label",
        "supergroup",
        "source",
        "genome_id",
        "contig_id",
        "start",
        "end",
        "length",
        "length_bin",
    ]
    with open(meta_dir / "fragments.tsv", "wt") as out:
        out.write("\t".join(header) + "\n")
        for row in conn.execute(
            """
            SELECT fragment_id, split, label, supergroup, source, genome_id, contig_id,
                   start, end, length, length_bin
            FROM fragments
            WHERE selected=1
            ORDER BY split, label, fragment_id
            """
        ):
            out.write("\t".join(str(value) for value in row) + "\n")


def quantiles_from_length_counts(counts: dict[int, int], probs: list[float]) -> list[int]:
    total = sum(counts.values())
    if total == 0:
        return [0 for _ in probs]
    sorted_counts = sorted(counts.items())
    targets = [max(1, math.ceil(total * prob)) for prob in probs]
    values: list[int] = []
    cumulative = 0
    target_index = 0
    for length, count in sorted_counts:
        cumulative += count
        while target_index < len(targets) and cumulative >= targets[target_index]:
            values.append(length)
            target_index += 1
    while len(values) < len(probs):
        values.append(sorted_counts[-1][0])
    return values


def write_qc(conn: sqlite3.Connection, qc_dir: Path, min_len: int, max_len: int, bin_width: int) -> None:
    summary_rows: list[tuple[object, ...]] = []
    groups = list(
        conn.execute(
            """
            SELECT split, label, supergroup, source, COUNT(*), SUM(length), AVG(length), MIN(length), MAX(length)
            FROM fragments
            WHERE selected=1
            GROUP BY split, label, supergroup, source
            ORDER BY split, label, source
            """
        )
    )
    for split, label, supergroup, source, count, total_bp, mean_len, min_observed, max_observed in groups:
        length_counts = {
            int(length): int(n)
            for length, n in conn.execute(
                """
                SELECT length, COUNT(*) FROM fragments
                WHERE selected=1 AND split=? AND label=? AND source=?
                GROUP BY length
                """,
                (split, label, source),
            )
        }
        p10, p25, median, p75, p90 = quantiles_from_length_counts(length_counts, [0.10, 0.25, 0.50, 0.75, 0.90])
        summary_rows.append(
            (
                split,
                label,
                supergroup,
                source,
                count,
                total_bp or 0,
                f"{mean_len:.3f}" if mean_len is not None else "0",
                median,
                p10,
                p25,
                p75,
                p90,
                min_observed or 0,
                max_observed or 0,
            )
        )
    write_tsv(
        qc_dir / "summary.tsv",
        [
            "split",
            "label",
            "supergroup",
            "source",
            "fragment_count",
            "total_bp",
            "mean_length",
            "median_length",
            "p10_length",
            "p25_length",
            "p75_length",
            "p90_length",
            "min_length",
            "max_length",
        ],
        summary_rows,
    )
    dist_rows = conn.execute(
        """
        SELECT split, label, supergroup, source, length_bin, COUNT(*), SUM(length)
        FROM fragments
        WHERE selected=1
        GROUP BY split, label, supergroup, source, length_bin
        ORDER BY split, label, source, length_bin
        """
    )
    write_tsv(
        qc_dir / "length_distribution.tsv",
        ["split", "label", "supergroup", "source", "bin_start", "bin_end", "fragment_count", "total_bp"],
        (
            (split, label, supergroup, source, bin_start, min(max_len, int(bin_start) + bin_width - 1), count, total_bp or 0)
            for split, label, supergroup, source, bin_start, count, total_bp in dist_rows
        ),
    )


def write_verification(
    conn: sqlite3.Connection,
    input_root: Path,
    meta_dir: Path,
    qc_dir: Path,
    min_len: int,
    max_len: int,
    removed: dict[str, float],
    coverage_threshold: float,
) -> None:
    checks: list[tuple[str, str, str]] = []
    virus_realms = {
        row["realm_group"]
        for row in read_tsv(meta_dir / "virus_accession_realm.tsv")
    }
    checks.append(("viral_realms_in_target_set", "pass" if virus_realms <= set(REALM_GROUPS) else "fail", ",".join(sorted(virus_realms))))
    nonviral_metadata_rows = read_tsv(meta_dir / "nonviral_genomes.tsv")
    genus_counts: dict[str, int] = defaultdict(int)
    for row in nonviral_metadata_rows:
        if row["source"] == "bacteria" and row["selected"] == "1":
            genus_counts[row["genus"]] += 1
    over_limit = {genus: count for genus, count in genus_counts.items() if count > 2}
    checks.append(("bacteria_selected_per_genus_le_2", "pass" if not over_limit else "fail", str(over_limit)))
    insect_genus_counts: dict[str, int] = defaultdict(int)
    selected_insect_accessions: set[str] = set()
    insect_without_retained_records: list[str] = []
    for row in nonviral_metadata_rows:
        if row["source"] == "insect" and row["selected"] == "1":
            insect_genus_counts[row["genus"]] += 1
            selected_insect_accessions.add(row.get("accession") or row["original_id"])
            if int(row.get("sequence_count") or 0) < 1:
                insect_without_retained_records.append(row["genome_id"])
    insect_over_limit = {genus: count for genus, count in insect_genus_counts.items() if count > 1}
    checks.append(("insect_selected_per_genus_le_1", "pass" if not insect_over_limit else "fail", str(insect_over_limit)))
    checks.append(
        (
            "insect_selected_assemblies_have_records",
            "pass" if not insect_without_retained_records else "fail",
            str(insect_without_retained_records[:10]),
        )
    )
    insect_txt = input_root / "insect/insect.txt"
    if insect_txt.exists():
        expected_insects = {line.strip() for line in insect_txt.read_text(encoding="utf-8").splitlines() if line.strip()}
        missing = sorted(expected_insects - selected_insect_accessions)
        extra = sorted(selected_insect_accessions - expected_insects)
        checks.append(
            (
                "insect_txt_matches_selected_accessions",
                "pass" if not missing and not extra else "fail",
                f"missing={missing[:10]} extra={extra[:10]}",
            )
        )
    else:
        checks.append(("insect_txt_matches_selected_accessions", "fail", f"missing {insect_txt}"))
    split_counts: dict[str, set[str]] = defaultdict(set)
    for row in read_tsv(meta_dir / "genome_split.tsv"):
        split_counts[row["genome_id"]].add(row["split"])
    cross_split = [genome for genome, splits in split_counts.items() if len(splits) > 1]
    checks.append(("genome_not_cross_split", "pass" if not cross_split else "fail", str(cross_split[:10])))
    min_observed, max_observed = conn.execute(
        "SELECT MIN(length), MAX(length) FROM fragments WHERE selected=1"
    ).fetchone()
    length_pass = (min_observed is None) or (int(min_observed) >= min_len and int(max_observed) <= max_len)
    checks.append(("final_fragment_lengths_in_range", "pass" if length_pass else "fail", f"{min_observed}-{max_observed}"))
    still_selected_removed = conn.execute(
        "SELECT COUNT(*) FROM fragments WHERE selected=1 AND removed_by_train_blast=1"
    ).fetchone()[0]
    checks.append(("removed_test_fragments_not_selected", "pass" if still_selected_removed == 0 else "fail", str(still_selected_removed)))
    max_removed_cov = max(removed.values()) if removed else 0.0
    checks.append(("test_train_coverage_filter_applied", "pass", f"removed={len(removed)} max_removed_coverage={max_removed_cov:.6g} threshold={coverage_threshold}"))
    balance_rows = []
    for split, virus_bp, nonvirus_bp in conn.execute(
        """
        SELECT v.split, v.total_bp, n.total_bp
        FROM (
            SELECT split, SUM(length) AS total_bp
            FROM fragments
            WHERE selected=1 AND supergroup='virus'
            GROUP BY split
        ) v
        JOIN (
            SELECT split, SUM(length) AS total_bp
            FROM fragments
            WHERE selected=1 AND supergroup='nonvirus'
            GROUP BY split
        ) n ON n.split = v.split
        ORDER BY v.split
        """
    ):
        virus_bp = int(virus_bp or 0)
        nonvirus_bp = int(nonvirus_bp or 0)
        diff = abs(nonvirus_bp - virus_bp)
        tolerance = max_len * len(NONVIRAL_SOURCES) * 25
        balance_rows.append((split, virus_bp, nonvirus_bp, diff, tolerance))
    balance_pass = all(diff <= tolerance for _split, _virus_bp, _nonvirus_bp, diff, tolerance in balance_rows)
    checks.append(
        (
            "nonvirus_bp_balanced_to_virus",
            "pass" if balance_pass else "fail",
            ";".join(
                f"{split}:virus={virus_bp},nonvirus={nonvirus_bp},diff={diff},tolerance={tolerance}"
                for split, virus_bp, nonvirus_bp, diff, tolerance in balance_rows
            ),
        )
    )
    source_balance_rows = []
    virus_bp_by_split = {
        str(split): int(total_bp or 0)
        for split, total_bp in conn.execute(
            """
            SELECT split, SUM(length)
            FROM fragments
            WHERE selected=1 AND supergroup='virus'
            GROUP BY split
            """
        )
    }
    source_bp_by_split = {
        (str(split), str(source)): int(total_bp or 0)
        for split, source, total_bp in conn.execute(
            """
            SELECT split, source, SUM(length)
            FROM fragments
            WHERE selected=1 AND supergroup='nonvirus'
            GROUP BY split, source
            """
        )
    }
    for split in ("train", "test"):
        target = virus_bp_by_split.get(split, 0) / len(NONVIRAL_SOURCES)
        tolerance = max_len * 25
        for source in NONVIRAL_SOURCES:
            observed = source_bp_by_split.get((split, source), 0)
            diff = abs(observed - target)
            source_balance_rows.append((split, source, target, observed, diff, tolerance))
    source_balance_pass = all(diff <= tolerance for _split, _source, _target, _observed, diff, tolerance in source_balance_rows)
    checks.append(
        (
            "nonvirus_source_bp_balanced",
            "pass" if source_balance_pass else "fail",
            ";".join(
                f"{split}/{source}:target={target:.3f},observed={observed},diff={diff:.3f},tolerance={tolerance}"
                for split, source, target, observed, diff, tolerance in source_balance_rows
            ),
        )
    )
    write_tsv(qc_dir / "verification.tsv", ["check", "status", "detail"], checks)


def write_processing_summary(
    output_dir: Path,
    input_root: Path,
    seed: int,
    threads: int,
    test_fraction: float,
    min_len: int,
    max_len: int,
    bin_width: int,
    coverage_threshold: float,
) -> None:
    summary_rows = read_tsv(output_dir / "qc/summary.tsv")
    verification_rows = read_tsv(output_dir / "qc/verification.tsv")
    split_rows = read_tsv(output_dir / "metadata/genome_split.tsv")
    nonviral_rows = read_tsv(output_dir / "metadata/nonviral_genomes.tsv")

    selected_by_source: dict[str, int] = defaultdict(int)
    for row in nonviral_rows:
        if row["selected"] == "1":
            selected_by_source[row["source"]] += 1

    split_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in split_rows:
        split_counts[(row["label"], row["split"])] += 1

    failed = [row for row in verification_rows if row["status"] != "pass"]
    lines = [
        "# Realm-Rank v3 Processing Summary",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Input root: `{input_root}`",
        f"Output directory: `{output_dir}`",
        "",
        "## Parameters",
        "",
        f"- Seed: `{seed}`",
        f"- Threads: `{threads}`",
        f"- Test fraction: `{test_fraction}`",
        f"- Fragment length: `{min_len}-{max_len}`",
        f"- Length bin width: `{bin_width}`",
        f"- Test-vs-train coverage removal threshold: `>{coverage_threshold}`",
        f"- Nonviral sources: `{', '.join(NONVIRAL_SOURCES)}`",
        f"- Genus caps: bacteria <= {GENUS_CAPS['bacteria']}; insect <= {GENUS_CAPS['insect']}",
        "",
        "## Selected Nonviral Genomes",
        "",
    ]
    for source in NONVIRAL_SOURCES:
        lines.append(f"- {source}: {selected_by_source.get(source, 0)}")
    lines.extend(["", "## Genome Split", ""])
    for label in sorted({label for label, _split in split_counts}):
        lines.append(
            f"- {label}: train={split_counts.get((label, 'train'), 0)} "
            f"test={split_counts.get((label, 'test'), 0)}"
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
            lines.append(f"- {row['check']}: {row['status']} ({row['detail']})")
    else:
        lines.append("- All checks in `qc/verification.tsv` passed.")
    lines.extend(
        [
            "",
            "## Key Outputs",
            "",
            "- `train.fasta.gz` and `test.fasta.gz`: final compressed FASTA files.",
            "- `metadata/fragments.tsv`: selected fragment coordinates.",
            "- `metadata/nonviral_genomes.tsv`: selected and excluded nonviral genome metadata.",
            "- `qc/large_source_downsample.tsv`: intermediate bacteria/insect bp caps.",
            "- `qc/nonvirus_downsample_by_bin_source.tsv`: final six-way source/bin bp targets.",
            "- `qc/verification.tsv`: integrity checks.",
        ]
    )
    (output_dir / "processing_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=RAW_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_ROOT / "realm_rank_common")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threads", type=int, default=0, help="Default: 85%% of available CPUs")
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--min-fragment-length", type=int, default=300)
    parser.add_argument("--max-fragment-length", type=int, default=2000)
    parser.add_argument("--length-bin-width", type=int, default=100)
    parser.add_argument("--length-balance-threshold", type=float, default=0.05)
    parser.add_argument("--decontam-pident", type=float, default=90.0)
    parser.add_argument("--decontam-hsp-qcov", type=float, default=0.80)
    parser.add_argument("--test-train-pident", type=float, default=90.0)
    parser.add_argument("--test-train-coverage", type=float, default=0.50)
    parser.add_argument("--ncbi-email", default=os.environ.get("NCBI_EMAIL", "realm-rank-builder@example.com"))
    parser.add_argument("--ncbi-api-key", default=os.environ.get("NCBI_API_KEY", ""))
    parser.add_argument("--ncbi-batch-size", type=int, default=200)
    parser.add_argument("--ncbi-delay", type=float, default=0.34)
    parser.add_argument("--blastn", default=None)
    parser.add_argument("--makeblastdb", default=None)
    parser.add_argument("--no-validate-gzip", action="store_true", help="Skip gzip integrity preflight")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the output directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    threads = resolve_threads(args.threads)
    blastn = resolve_tool("blastn", args.blastn)
    makeblastdb = resolve_tool("makeblastdb", args.makeblastdb)
    meta_dir, qc_dir, work_dir = ensure_dirs(args.output_dir, args.force)
    files = discover_files(args.input_root)
    if not args.no_validate_gzip:
        files = validate_input_files(files, meta_dir, threads)
    log(f"using {threads} BLAST threads")
    log("building viral FASTA and accession list")
    virus_rows, virus_fasta, virus_contigs = build_virus_fasta(files["virus"], work_dir)
    assign_viral_realms(
        virus_rows,
        meta_dir,
        work_dir,
        args.ncbi_email,
        args.ncbi_api_key,
        args.ncbi_batch_size,
        args.ncbi_delay,
    )
    log("discovering nonviral genomes and applying bacteria/insect genus caps")
    nonviral_rows = discover_nonviral_genomes(files, args.seed)
    split_genomes(virus_rows, nonviral_rows, args.seed, args.test_fraction)
    write_genome_split(meta_dir, virus_rows, nonviral_rows)
    log("writing selected nonviral FASTA")
    nonviral_fasta, nonviral_contigs = write_selected_nonviral_fasta(files, nonviral_rows, work_dir, meta_dir)
    log("building viral BLAST database")
    virus_db = work_dir / "blast/virus_db/virus"
    make_blast_db(makeblastdb, virus_fasta, virus_db)
    log("running nonviral-vs-virus BLAST for contamination intervals")
    nonviral_vs_virus = work_dir / "nonviral_vs_virus.tsv"
    run_blastn(blastn, nonviral_fasta, virus_db, nonviral_vs_virus, threads)
    log("merging viral-like intervals and cleaning nonviral sequences")
    intervals_tsv = work_dir / "nonviral_virus_intervals.tsv"
    intervals = build_decontam_intervals(nonviral_vs_virus, intervals_tsv, args.decontam_pident, args.decontam_hsp_qcov)
    cleaned_nonviral = decontaminate_nonviral_fasta(nonviral_fasta, nonviral_contigs, intervals, work_dir, meta_dir)
    log("generating fragment metadata")
    genome_split = load_genome_split(meta_dir)
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
    log(f"generated {virus_fragment_count} viral and {nonvirus_fragment_count} nonviral fragments")
    log("downsampling large bacteria/insect fragment pools against viral bp targets")
    downsample_large_sources_to_virus_bp(conn, args.seed, qc_dir)
    log("checking and balancing length-bin distributions")
    balance_nonvirus_length_distribution(conn, args.seed, qc_dir, args.length_balance_threshold)
    log("writing selected prefilter train/test FASTA")
    train_prefilter, test_prefilter = write_selected_fragment_fastas(
        conn,
        [(virus_fasta, virus_info), (cleaned_nonviral, nonviral_info)],
        work_dir,
        args.min_fragment_length,
        args.max_fragment_length,
        args.seed,
    )
    log("building final train-fragment BLAST database")
    train_db = work_dir / "blast/train_db/train"
    make_blast_db(makeblastdb, train_prefilter, train_db)
    log("running test-vs-train BLAST leakage check")
    test_vs_train = work_dir / "test_vs_train.tsv"
    run_blastn(blastn, test_prefilter, train_db, test_vs_train, threads)
    removed = parse_train_blast_coverage(
        test_vs_train,
        meta_dir / "test_removed_by_train_blast.tsv",
        args.test_train_pident,
        args.test_train_coverage,
    )
    update_removed_by_train_blast(conn, removed)
    log("downsampling all nonviral fragments against final viral bp targets by length bin")
    downsample_nonvirus_to_virus_bp_by_length_bin(conn, args.seed, qc_dir)
    log("writing final compressed FASTA and metadata")
    selected = load_selected_fragment_ids(conn)
    copy_or_filter_final_fasta(
        train_prefilter,
        args.output_dir / "train.fasta.gz",
        keep_ids=selected["train"],
        threads=threads,
    )
    copy_or_filter_final_fasta(
        test_prefilter,
        args.output_dir / "test.fasta.gz",
        keep_ids=selected["test"],
        threads=threads,
    )
    dump_final_fragments(conn, meta_dir)
    write_qc(conn, qc_dir, args.min_fragment_length, args.max_fragment_length, args.length_bin_width)
    write_verification(
        conn,
        args.input_root,
        meta_dir,
        qc_dir,
        args.min_fragment_length,
        args.max_fragment_length,
        removed,
        args.test_train_coverage,
    )
    write_processing_summary(
        args.output_dir,
        args.input_root,
        args.seed,
        threads,
        args.test_fraction,
        args.min_fragment_length,
        args.max_fragment_length,
        args.length_bin_width,
        args.test_train_coverage,
    )
    conn.close()
    log(f"done: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
