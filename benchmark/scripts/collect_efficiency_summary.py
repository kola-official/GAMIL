#!/usr/bin/env python3
import csv
import json
import os
from pathlib import Path


ROOT = Path(os.environ.get("EFFICIENCY_ROOT", Path(__file__).resolve().parents[1]))
SUMMARY_DIR = ROOT / "metrics" / "summary"
OUT_CSV = ROOT / "final_summary.csv"
OUT_JSON = ROOT / "final_summary.json"


def main():
    rows = []
    for path in sorted(SUMMARY_DIR.glob("*.json")):
        with open(path) as handle:
            item = json.load(handle)
        rows.append(item)

    fields = [
        "name",
        "return_code",
        "elapsed_sec",
        "max_rss_mb",
        "max_gpu_mem_mb",
        "max_cpu_seconds",
        "sample_count",
        "metrics_csv",
        "stdout",
        "stderr",
    ]
    with open(OUT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
