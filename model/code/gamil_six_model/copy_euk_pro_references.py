#!/usr/bin/env python3
"""Copy existing euk/pro teacher-reference results into this experiment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from experiment_config import EUK_PRO_EVAL_ROOT, EUK_PRO_REFERENCE_FILES, REFERENCE_ROOT
from shared import require_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy euk/pro reference files")
    parser.add_argument("--source-root", default=str(EUK_PRO_EVAL_ROOT))
    parser.add_argument("--output-dir", default=str(REFERENCE_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel_src, dst_name in EUK_PRO_REFERENCE_FILES.items():
        src = require_file(source_root / rel_src)
        dst = output_dir / dst_name
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "destination": str(dst)})
    write_json(output_dir / "reference_manifest.json", {"copied": copied})
    print(f"Copied {len(copied)} euk/pro reference files to {output_dir}")


if __name__ == "__main__":
    main()

