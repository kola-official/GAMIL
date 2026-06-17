#!/usr/bin/env python3
"""Upload GAMIL release archives to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


RELEASE_FILES = (
    "gamil_core_data_v1.tar.zst",
    "gamil_euk_pro_benchmark_v1.tar.zst",
    "gamil_model_weights_v1.tar.zst",
    "SHA256SUMS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload GAMIL release archives to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Target repository, e.g. username/gamil-release")
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=("dataset", "model"),
        help="Use 'dataset' for release archives unless you specifically want a model repo.",
    )
    parser.add_argument("--source-dir", required=True, help="Directory containing the release archives.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""), help="HF token, or set HF_TOKEN.")
    parser.add_argument("--revision", default="main", help="Target branch/revision.")
    parser.add_argument("--private", action="store_true", help="Create the repository as private if it does not exist.")
    parser.add_argument("--commit-message", default="Upload GAMIL release archives", help="Commit message.")
    return parser.parse_args()


def ensure_files(source_dir: Path, names: Iterable[str]) -> list[Path]:
    paths = []
    for name in names:
        path = source_dir / name
        if not path.is_file():
            raise SystemExit(f"Missing required release file: {path}")
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit("Missing Hugging Face token. Set HF_TOKEN or pass --token.")

    source_dir = Path(args.source_dir).resolve()
    files = ensure_files(source_dir, RELEASE_FILES)

    api = HfApi(token=args.token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        exist_ok=True,
    )

    for path in files:
        result = api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
        )
        print(f"uploaded {path.name}: {result}")


if __name__ == "__main__":
    main()
