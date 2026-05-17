"""Bootstrap percentile 95% CIs on AUROC, Youden-F1, F1-max-F1.

Reads a per-image score file produced by ``scripts/score_anomalib_test.py``
(schema ``inspectnet_cx.scores.v1``) and runs ``n`` stratified bootstrap
resamples, recomputing metrics on each resample. Reports the percentile
95% confidence interval and the bootstrap median.

The stratified resample preserves the (n_pos, n_neg) counts of the original
sample so that AUROC is always defined on each bootstrap iterate. This is
the standard approach for bounded class-imbalanced metrics.

Usage:
    PYTHONPATH=src python3 scripts/bootstrap_auroc.py \\
        --input reports/scores_padim_bottle.json \\
        --output reports/bootstrap_padim_bottle.json \\
        --n-bootstrap 1000 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def _youden_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    idx = int(np.argmax(tpr - fpr))
    thr = float(thresholds[idx])
    predicted = scores >= thr
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    denom = precision + recall
    f1 = np.zeros_like(precision)
    mask = denom > 0
    f1[mask] = 2 * precision[mask] * recall[mask] / denom[mask]
    return float(np.max(f1))


def bootstrap_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Stratified percentile bootstrap on AUROC, Youden-F1, F1-max-F1.

    Args:
        labels: 1-D int array of 0/1 labels.
        scores: 1-D float array of anomaly scores (higher = more anomalous).
        n_bootstrap: Number of resamples.
        seed: RNG seed for determinism.

    Returns:
        Mapping ``{metric: {point, ci_low, ci_high, median, std}}``.
    """
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have equal shape")
    if labels.size == 0:
        raise ValueError("labels and scores must not be empty")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must contain only 0/1 entries")
    if np.unique(labels).size < 2:
        raise ValueError("need at least one positive and one negative")

    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos = pos_idx.size
    n_neg = neg_idx.size

    aurocs = np.empty(n_bootstrap, dtype=np.float64)
    youdens = np.empty(n_bootstrap, dtype=np.float64)
    f1maxes = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        sp = rng.choice(pos_idx, size=n_pos, replace=True)
        sn = rng.choice(neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([sp, sn])
        bl = labels[idx]
        bs = scores[idx]
        try:
            aurocs[i] = _auroc(bl, bs)
            youdens[i] = _youden_f1(bl, bs)
            f1maxes[i] = _f1_max(bl, bs)
        except ValueError:
            aurocs[i] = np.nan
            youdens[i] = np.nan
            f1maxes[i] = np.nan

    point_auroc = _auroc(labels, scores)
    point_youden = _youden_f1(labels, scores)
    point_f1max = _f1_max(labels, scores)

    def _summary(arr: np.ndarray, point: float) -> dict[str, float]:
        clean = arr[~np.isnan(arr)]
        return {
            "point": float(point),
            "ci_low": float(np.percentile(clean, 2.5)),
            "ci_high": float(np.percentile(clean, 97.5)),
            "median": float(np.percentile(clean, 50.0)),
            "std": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
            "n_valid": int(clean.size),
        }

    return {
        "auroc": _summary(aurocs, point_auroc),
        "youden_f1": _summary(youdens, point_youden),
        "f1_max_f1": _summary(f1maxes, point_f1max),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    payload = json.loads(args.input.read_text())
    items = payload["items"]
    labels = np.asarray([int(it["label"]) for it in items], dtype=np.int64)
    scores = np.asarray([float(it["score"]) for it in items], dtype=np.float64)

    result = bootstrap_metrics(
        labels, scores, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    out = {
        "schema": "inspectnet_cx.bootstrap_auroc.v1",
        "source_score_file": str(args.input),
        "dataset": payload.get("dataset"),
        "category": payload.get("category"),
        "method": payload.get("method"),
        "n_samples": int(labels.size),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "metrics": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
