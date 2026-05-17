"""Operating-point analysis from labeled per-image anomaly scores.

Given a list of (label, score) pairs from a held-out test split, this module
computes the ROC curve, AUROC, and three threshold recommendations:

- Youden's J statistic (max TPR - FPR).
- F1-maximizing threshold.
- The lowest threshold whose FPR is at or below a target value.

It is dataset and model agnostic. Input scores can come from any anomaly
detector. Labels are 0 for normal and 1 for anomalous.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    tpr: float
    fpr: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "tpr": float(self.tpr),
            "fpr": float(self.fpr),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "tp": int(self.tp),
            "fp": int(self.fp),
            "tn": int(self.tn),
            "fn": int(self.fn),
        }


def _operating_point_at(
    threshold: float,
    labels: np.ndarray,
    scores: np.ndarray,
) -> OperatingPoint:
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    n_pos = tp + fn
    n_neg = fp + tn
    tpr = tp / n_pos if n_pos > 0 else 0.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return OperatingPoint(
        threshold=threshold,
        tpr=tpr,
        fpr=fpr,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def analyze_thresholds(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    target_fprs: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict[str, Any]:
    """Compute ROC/PR/threshold recommendations from labeled scores.

    Args:
        labels: 0 for normal, 1 for anomalous.
        scores: anomaly scores (higher = more anomalous).
        target_fprs: FPR ceilings to compute the lowest-threshold operating
            point for.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    scores_arr = np.asarray(scores, dtype=np.float64)

    if labels_arr.shape != scores_arr.shape:
        raise ValueError(
            f"labels and scores must have equal length, "
            f"got {labels_arr.shape} and {scores_arr.shape}"
        )
    if labels_arr.size == 0:
        raise ValueError("labels and scores must not be empty")
    if not np.all((labels_arr == 0) | (labels_arr == 1)):
        raise ValueError("labels must contain only 0 or 1 entries")
    if np.unique(labels_arr).size < 2:
        raise ValueError(
            "labels must include at least one positive and one negative example"
        )

    fpr, tpr, roc_thresholds = roc_curve(labels_arr, scores_arr)
    auroc = float(auc(fpr, tpr))

    precision, recall, pr_thresholds = precision_recall_curve(labels_arr, scores_arr)
    f1_curve = np.zeros_like(precision)
    denom = precision + recall
    mask = denom > 0
    f1_curve[mask] = 2 * precision[mask] * recall[mask] / denom[mask]

    if pr_thresholds.size == 0:
        f1_threshold = float(scores_arr.min())
    else:
        f1_best_idx = int(np.argmax(f1_curve[:-1])) if f1_curve.size > 1 else 0
        f1_best_idx = min(f1_best_idx, pr_thresholds.size - 1)
        f1_threshold = float(pr_thresholds[f1_best_idx])

    youden_idx = int(np.argmax(tpr - fpr))
    youden_threshold = float(roc_thresholds[youden_idx])

    target_points = {}
    for target in target_fprs:
        eligible = np.where(fpr <= target)[0]
        if eligible.size == 0:
            continue
        chosen = int(eligible[-1])
        target_points[f"fpr<= {target:.2f}"] = _operating_point_at(
            float(roc_thresholds[chosen]),
            labels_arr,
            scores_arr,
        ).to_dict()

    roc_table = [
        {"fpr": float(f), "tpr": float(t), "threshold": float(th)}
        for f, t, th in zip(fpr, tpr, roc_thresholds, strict=True)
    ]

    return {
        "schema": "inspectnet_cx.threshold_analysis.v1",
        "n_samples": int(labels_arr.size),
        "n_positive": int(np.sum(labels_arr == 1)),
        "n_negative": int(np.sum(labels_arr == 0)),
        "score_min": float(scores_arr.min()),
        "score_max": float(scores_arr.max()),
        "auroc": auroc,
        "youden": _operating_point_at(youden_threshold, labels_arr, scores_arr).to_dict(),
        "f1_max": _operating_point_at(f1_threshold, labels_arr, scores_arr).to_dict(),
        "target_fpr": target_points,
        "roc_curve": roc_table,
        "proof_note": (
            "Operating points are derived from labeled held-out scores only. "
            "They do not establish production thresholds; deployment thresholds "
            "require a separate calibration set, prior class balance, and the "
            "operator's tolerated false-positive rate."
        ),
    }
