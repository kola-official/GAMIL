#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List

METRICS = ["f1_macro", "auprc_macro_ovr", "precision_macro", "recall_macro", "mcc", "accuracy"]


def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/*/*/seed*/summary.json")):
        data = json.loads(path.read_text())
        dataset = data.get("dataset_name") or path.parts[-5]
        model_name = data.get("model_name") or path.parts[-4]
        for method, metrics in data.get("methods", {}).items():
            avg = metrics.get("avg", {})
            model = data.get("model_name") or path.parts[-4]
            row = {
                "dataset_name": dataset,
                "model_name": model,
                "model_name": model_name,
                "method": method,
                "seed": data.get("seed"),
                "window_len": data.get("window_len"),
                "train_num_windows": data.get("train_num_windows"),
                "eval_num_windows": data.get("eval_num_windows"),
                "summary_path": str(path),
            }
            for metric in METRICS:
                row[metric] = avg.get(metric, "")
            rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["dataset_name"], row.get("model_name", ""), row["method"], row["window_len"], row["train_num_windows"], row["eval_num_windows"])
        buckets[key].append(row)
    out: List[Dict[str, Any]] = []
    for (dataset, model_name, method, window_len, train_windows, eval_windows), items in sorted(buckets.items()):
        row = {
            "dataset_name": dataset,
            "model_name": model_name,
            "method": method,
            "window_len": window_len,
            "train_num_windows": train_windows,
            "eval_num_windows": eval_windows,
            "n_seeds": len(items),
            "seeds": ",".join(str(x["seed"]) for x in sorted(items, key=lambda r: int(r["seed"]))),
        }
        for metric in METRICS:
            vals = [float(x[metric]) for x in items if x.get(metric) not in (None, "")]
            row[f"{metric}_mean"] = mean(vals) if vals else ""
            row[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0 if vals else ""
        out.append(row)
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize ViroBench-GAMIL summary.json files")
    ap.add_argument("--root", required=True, help="Result root")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "tables"
    rows = load_rows(root)
    agg = aggregate(rows)
    write_csv(out_dir / "per_seed_metrics.csv", rows)
    write_csv(out_dir / "aggregate_metrics.csv", agg)
    (out_dir / "aggregate_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"loaded {len(rows)} method-seed rows from {root}")
    print(f"wrote {out_dir / 'aggregate_metrics.csv'}")


if __name__ == "__main__":
    main()
