from __future__ import annotations

from typing import Any, Dict, Mapping

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    return np.log(np.clip(probs, 1e-12, 1.0))


def compute_multiclass_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    num_classes: int | None = None,
) -> Dict[str, float]:
    logits = np.asarray(logits)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits, got shape {logits.shape}")
    if num_classes is None:
        num_classes = int(logits.shape[1])

    valid = (labels >= 0) & (labels < num_classes)
    logits = logits[valid]
    labels = labels[valid]
    if labels.size == 0:
        return {}

    preds = logits.argmax(axis=1)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)) if len(np.unique(labels)) > 1 else 0.0,
    }

    probs = softmax_np(logits)
    try:
        if len(np.unique(labels)) > 1:
            out["auc_macro_ovr"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
    except Exception:
        pass
    try:
        y_bin = label_binarize(labels, classes=np.arange(num_classes))
        out["auprc_macro_ovr"] = float(average_precision_score(y_bin, probs, average="macro"))
    except Exception:
        pass
    return out


def summarize_task_metrics(metrics_by_task: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"by_task": dict(metrics_by_task)}
    metric_names = sorted({k for m in metrics_by_task.values() for k in m})
    avg: Dict[str, float] = {}
    for name in metric_names:
        vals = [float(m[name]) for m in metrics_by_task.values() if name in m]
        if vals:
            avg[name] = float(np.mean(vals))
    summary["avg"] = avg
    return summary
