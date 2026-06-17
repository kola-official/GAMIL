#!/usr/bin/env python3
"""Upload GAMIL release archives to a Zenodo draft deposition."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import requests


RELEASE_FILES = (
    "gamil_core_data_v1.tar.zst",
    "gamil_euk_pro_benchmark_v1.tar.zst",
    "gamil_model_weights_v1.tar.zst",
    "SHA256SUMS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload GAMIL release archives to Zenodo.")
    parser.add_argument("--source-dir", required=True, help="Directory containing the release archives.")
    parser.add_argument("--token", default=os.environ.get("ZENODO_TOKEN", ""), help="Zenodo token, or set ZENODO_TOKEN.")
    parser.add_argument("--metadata-json", help="Path to a Zenodo metadata JSON file.")
    parser.add_argument("--deposition-id", type=int, help="Existing draft deposition id. Omit to create a new draft.")
    parser.add_argument("--sandbox", action="store_true", help="Use sandbox.zenodo.org instead of zenodo.org.")
    parser.add_argument("--publish", action="store_true", help="Publish after upload and metadata update.")
    return parser.parse_args()


def ensure_files(source_dir: Path, names: Iterable[str]) -> list[Path]:
    paths = []
    for name in names:
        path = source_dir / name
        if not path.is_file():
            raise SystemExit(f"Missing required release file: {path}")
        paths.append(path)
    return paths


def load_metadata(path: str | None) -> dict | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text())
    if "metadata" in payload:
        return payload
    return {"metadata": payload}


def request_json(method: str, url: str, token: str, **kwargs) -> dict:
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    response = requests.request(method, url, headers=headers, timeout=300, **kwargs)
    if response.status_code >= 400:
        raise SystemExit(f"{method} {url} failed: {response.status_code} {response.text}")
    return response.json() if response.text else {}


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit("Missing Zenodo token. Set ZENODO_TOKEN or pass --token.")

    source_dir = Path(args.source_dir).resolve()
    files = ensure_files(source_dir, RELEASE_FILES)
    metadata = load_metadata(args.metadata_json)

    base = "https://sandbox.zenodo.org" if args.sandbox else "https://zenodo.org"
    dep_url = f"{base}/api/deposit/depositions"

    if args.deposition_id:
        deposition = request_json("GET", f"{dep_url}/{args.deposition_id}", args.token)
    else:
        deposition = request_json(
            "POST",
            dep_url,
            args.token,
            headers={"Content-Type": "application/json"},
            json={},
        )

    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    html_url = deposition["links"].get("latest_draft_html", deposition["links"].get("html", ""))
    print(f"deposition_id={deposition_id}")
    print(f"draft_url={html_url}")

    for path in files:
        with path.open("rb") as handle:
            response = requests.put(
                f"{bucket_url}/{path.name}",
                data=handle,
                headers={"Authorization": f"Bearer {args.token}"},
                timeout=3600,
            )
        if response.status_code >= 400:
            raise SystemExit(f"Upload failed for {path.name}: {response.status_code} {response.text}")
        print(f"uploaded {path.name}")

    if metadata:
        updated = request_json(
            "PUT",
            f"{dep_url}/{deposition_id}",
            args.token,
            headers={"Content-Type": "application/json"},
            data=json.dumps(metadata),
        )
        doi = updated.get("metadata", {}).get("prereserve_doi", {}).get("doi", "")
        if doi:
            print(f"reserved_doi={doi}")

    if args.publish:
        if not metadata:
            raise SystemExit("--publish requires --metadata-json so Zenodo has publishable metadata.")
        published = request_json("POST", f"{dep_url}/{deposition_id}/actions/publish", args.token)
        record_html = published.get("links", {}).get("record_html", published.get("links", {}).get("html", ""))
        print(f"published_record={record_html}")


if __name__ == "__main__":
    main()
