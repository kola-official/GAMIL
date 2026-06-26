#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


def group_mean_logits(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray):
    order = np.argsort(groups, kind="mergesort")
    g = groups[order]
    z = logits[order]
    y = labels[order]
    uniq, start = np.unique(g, return_index=True)
    end = np.append(start[1:], len(g))
    counts = (end - start).reshape(-1, 1)
    return uniq, np.add.reduceat(z, start, axis=0) / counts, y[start]


def attention_stats(attn: Dict[int, torch.Tensor]) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for group, weights in attn.items():
        w = weights.detach().cpu().float().numpy().reshape(-1)
        w = w[w > 0]
        if w.size == 0:
            continue
        w = w / max(w.sum(), 1e-12)
        entropy = float(-(w * np.log(np.clip(w, 1e-12, 1.0))).sum())
        out[int(group)] = {
            "num_windows": int(w.size),
            "attention_entropy": entropy,
            "attention_entropy_norm": float(entropy / np.log(w.size)) if w.size > 1 else 0.0,
            "top_attention_weight": float(w.max()),
            "effective_window_count": float(np.exp(entropy)),
        }
    return out


def task_keys(npz) -> List[str]:
    return sorted(k for k in npz.files if k.startswith("logits_"))


def summarize_seed(seed_dir: Path) -> List[Dict[str, Any]]:
    window_path = seed_dir / "window_logits_test.npz"
    gamil_path = seed_dir / "gamil_test_predictions.npz"
    attn_path = seed_dir / "gamil_test_attention.pt"
    if not (window_path.exists() and gamil_path.exists() and attn_path.exists()):
        return []
    window = np.load(window_path)
    gamil = np.load(gamil_path)
    attn = attention_stats(torch.load(attn_path, map_location="cpu"))
    groups_win = window["groups"]
    labels_win = window["labels"]
    groups_gamil = gamil["groups"]
    labels_gamil = gamil["labels"]
    w_tasks = task_keys(window)
    g_tasks = task_keys(gamil)
    rows: List[Dict[str, Any]] = []
    for key in w_tasks:
        task = key[len("logits_"):]
        if key not in g_tasks:
            continue
        task_index_w = w_tasks.index(key)
        task_index_g = g_tasks.index(key)
        y_win = labels_win[:, 0] if labels_win.ndim == 1 else labels_win[:, task_index_w]
        y_gamil = labels_gamil[:, 0] if labels_gamil.ndim == 1 else labels_gamil[:, task_index_g]
        g, vb_logits, vb_labels = group_mean_logits(window[key], y_win, groups_win)
        vb_pred = vb_logits.argmax(axis=1)
        gamil_pred = gamil[key].argmax(axis=1)
        gamil_by_group = {int(group): (int(y), int(pred)) for group, y, pred in zip(groups_gamil, y_gamil, gamil_pred)}
        rescued: List[int] = []
        lost: List[int] = []
        for group, y, pred in zip(g, vb_labels, vb_pred):
            group = int(group)
            if group not in gamil_by_group:
                continue
            gy, gp = gamil_by_group[group]
            if int(y) != gy:
                continue
            vb_ok = int(pred) == int(y)
            gamil_ok = int(gp) == gy
            if (not vb_ok) and gamil_ok:
                rescued.append(group)
            if vb_ok and (not gamil_ok):
                lost.append(group)
        for tag, group_list in (("rescued", rescued), ("lost", lost)):
            vals = [attn[x] for x in group_list if x in attn]
            row: Dict[str, Any] = {
                "seed_dir": str(seed_dir),
                "dataset_name": seed_dir.parents[2].name,
                "task": task,
                "case_type": tag,
                "count": len(group_list),
                "with_attention": len(vals),
            }
            for metric in ("num_windows", "attention_entropy", "attention_entropy_norm", "top_attention_weight", "effective_window_count"):
                row[f"{metric}_mean"] = float(np.mean([v[metric] for v in vals])) if vals else ""
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose GAMIL rescued/lost cases and attention concentration")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for seed_dir in sorted(root.glob("*/DNABERT2-virobench/*/seed*")):
        rows.extend(summarize_seed(seed_dir))
    out = Path(args.out) if args.out else root / "tables" / "gamil_attention_diagnostics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out.write_text("", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} diagnostic rows to {out}")


if __name__ == "__main__":
    main()
