from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import compute_multiclass_metrics, logits_from_probs, softmax_np, summarize_task_metrics


def group_reduce_mean(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="mergesort")
    g = groups[order]
    z = logits[order]
    y = labels[order]
    uniq, start = np.unique(g, return_index=True)
    end = np.append(start[1:], len(g))
    counts = (end - start).reshape(-1, 1)
    sums = np.add.reduceat(z, start, axis=0)
    return sums / counts, y[start], uniq


def aggregate_quantile(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    q: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = softmax_np(logits)
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="mergesort")
    g = groups[order]
    p = probs[order]
    y = np.asarray(labels)[order]
    uniq, start = np.unique(g, return_index=True)
    end = np.append(start[1:], len(g))
    rows = []
    for a, b in zip(start, end):
        row = np.quantile(p[a:b], q=q, axis=0)
        row = row / np.clip(row.sum(), 1e-12, None)
        rows.append(row)
    return logits_from_probs(np.vstack(rows)), y[start], uniq


def aggregate_noisy_or(
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = softmax_np(logits)
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="mergesort")
    g = groups[order]
    p = probs[order]
    y = np.asarray(labels)[order]
    uniq, start = np.unique(g, return_index=True)
    end = np.append(start[1:], len(g))
    rows = []
    for a, b in zip(start, end):
        score = 1.0 - np.prod(1.0 - np.clip(p[a:b], 1e-9, 1.0 - 1e-9), axis=0)
        score = score / np.clip(score.sum(), 1e-12, None)
        rows.append(score)
    return logits_from_probs(np.vstack(rows)), y[start], uniq


def _topk_mean(values: np.ndarray, k: int) -> np.ndarray:
    k = min(k, values.shape[0])
    if k <= 0:
        return values.mean(axis=0)
    part = np.sort(values, axis=0)[-k:]
    return part.mean(axis=0)


def distribution_features(probs: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups)
    order = np.argsort(groups, kind="mergesort")
    g = groups[order]
    p = probs[order]
    uniq, start = np.unique(g, return_index=True)
    end = np.append(start[1:], len(g))
    feats = []
    for a, b in zip(start, end):
        block = p[a:b]
        mean = block.mean(axis=0)
        maxv = block.max(axis=0)
        std = block.std(axis=0)
        p75 = np.quantile(block, 0.75, axis=0)
        p90 = np.quantile(block, 0.90, axis=0)
        top3 = _topk_mean(block, 3)
        top5 = _topk_mean(block, 5)
        ratio50 = (block >= 0.5).mean(axis=0)
        ratio80 = (block >= 0.8).mean(axis=0)
        count = np.array([np.log1p(block.shape[0])], dtype=np.float64)
        feats.append(np.concatenate([mean, maxv, top3, top5, p75, p90, std, maxv - mean, ratio50, ratio80, count]))
    return np.vstack(feats), uniq


@dataclass
class LogRegResult:
    logits: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    selected_c: Optional[float]


def fit_logreg_posthoc(
    train_logits: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray,
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    val_groups: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    test_groups: np.ndarray,
    c_grid: Sequence[float] = (0.1, 0.3, 1.0, 3.0),
) -> LogRegResult:
    train_x, train_g = distribution_features(softmax_np(train_logits), train_groups)
    val_x, val_g = distribution_features(softmax_np(val_logits), val_groups)
    test_x, test_g = distribution_features(softmax_np(test_logits), test_groups)
    _, train_y, _ = group_reduce_mean(train_logits, train_labels, train_groups)
    _, val_y, _ = group_reduce_mean(val_logits, val_labels, val_groups)
    _, test_y, _ = group_reduce_mean(test_logits, test_labels, test_groups)

    if len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        z, y, g = group_reduce_mean(test_logits, test_labels, test_groups)
        return LogRegResult(logits=z, labels=y, groups=g, selected_c=None)

    best_c = float(c_grid[0])
    best_score = -1.0
    best_model = None
    for c in c_grid:
        try:
            model = LogisticRegression(C=float(c), max_iter=500, tol=1e-2, class_weight="balanced", multi_class="auto")
            model.fit(train_x, train_y)
            val_scores = model.decision_function(val_x)
            if val_scores.ndim == 1:
                val_scores = np.stack([-val_scores, val_scores], axis=1)
            score = compute_multiclass_metrics(val_scores, val_y).get("f1_macro", -1.0)
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_c = float(c)
            best_model = model
    if best_model is None:
        z, y, g = group_reduce_mean(test_logits, test_labels, test_groups)
        return LogRegResult(logits=z, labels=y, groups=g, selected_c=None)
    test_scores = best_model.decision_function(test_x)
    if test_scores.ndim == 1:
        test_scores = np.stack([-test_scores, test_scores], axis=1)
    return LogRegResult(logits=test_scores, labels=test_y, groups=test_g, selected_c=best_c)


def evaluate_window_posthoc(
    train_logits_by_task: Mapping[str, np.ndarray],
    val_logits_by_task: Mapping[str, np.ndarray],
    test_logits_by_task: Mapping[str, np.ndarray],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    train_groups: np.ndarray,
    val_groups: np.ndarray,
    test_groups: np.ndarray,
    task_names: Sequence[str],
    task_dims: Mapping[str, int],
    quantile: float = 0.95,
    c_grid: Sequence[float] = (0.1, 0.3, 1.0, 3.0),
) -> Dict[str, Dict[str, object]]:
    methods: Dict[str, Dict[str, object]] = {}
    for method in ("VB-Default", "PostHoc-Quantile", "PostHoc-NoisyOR", "PostHoc-LogReg"):
        by_task = {}
        details = {}
        for ti, task in enumerate(task_names):
            y_train = train_labels[:, ti]
            y_val = val_labels[:, ti]
            y_test = test_labels[:, ti]
            if method == "VB-Default":
                z, y, _ = group_reduce_mean(test_logits_by_task[task], y_test, test_groups)
            elif method == "PostHoc-Quantile":
                z, y, _ = aggregate_quantile(test_logits_by_task[task], y_test, test_groups, q=quantile)
            elif method == "PostHoc-NoisyOR":
                z, y, _ = aggregate_noisy_or(test_logits_by_task[task], y_test, test_groups)
            else:
                res = fit_logreg_posthoc(
                    train_logits_by_task[task], y_train, train_groups,
                    val_logits_by_task[task], y_val, val_groups,
                    test_logits_by_task[task], y_test, test_groups,
                    c_grid=c_grid,
                )
                z, y = res.logits, res.labels
                details[task] = {"selected_c": res.selected_c}
            by_task[task] = compute_multiclass_metrics(z, y, num_classes=int(task_dims[task]))
        methods[method] = summarize_task_metrics(by_task)
        if details:
            methods[method]["details"] = details
    return methods
