"""Build the 4x4 PaDiM cross-category AUROC transfer matrix + drop CI.

Reads same-category scores from ``reports/scores_padim_<cat>.json`` and
cross-category scores from ``reports/cross_padim_<train>_to_<test>.json``,
computes per-cell AUROC, and runs a stratified percentile bootstrap on
the (cross-cell minus diagonal-cell) AUROC drop, paired by test_category.

Output: ``reports/cross_padim_matrix.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, roc_curve


def _auroc_from_file(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    labels = np.asarray([int(it["label"]) for it in payload["items"]], dtype=np.int64)
    scores = np.asarray([float(it["score"]) for it in payload["items"]], dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr)), labels, scores


def _stratified_auroc(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> float:
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    sp = rng.choice(pos_idx, size=pos_idx.size, replace=True)
    sn = rng.choice(neg_idx, size=neg_idx.size, replace=True)
    idx = np.concatenate([sp, sn])
    fpr, tpr, _ = roc_curve(labels[idx], scores[idx])
    return float(auc(fpr, tpr))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    reports = repo / "reports"
    cats = ["bottle", "cable", "capsule", "leather"]

    matrix: dict[str, dict[str, float]] = {}
    cell_data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for train in cats:
        matrix[train] = {}
        for test in cats:
            if train == test:
                path = reports / f"scores_padim_{test}.json"
            else:
                path = reports / f"cross_padim_{train}_to_{test}.json"
            auroc, lbl, scr = _auroc_from_file(path)
            matrix[train][test] = auroc
            cell_data[(train, test)] = (lbl, scr)

    diag = {c: matrix[c][c] for c in cats}
    drops: list[float] = []
    for train in cats:
        for test in cats:
            if train == test:
                continue
            drops.append(diag[test] - matrix[train][test])
    drop_arr = np.asarray(drops, dtype=np.float64)
    drop_mean = float(np.mean(drop_arr))
    drop_std = float(np.std(drop_arr, ddof=1))

    rng = np.random.default_rng(0)
    n_boot = 1000
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        per_drop = []
        for train in cats:
            for test in cats:
                if train == test:
                    continue
                lbl_t, scr_t = cell_data[(test, test)]
                lbl_x, scr_x = cell_data[(train, test)]
                a_t = _stratified_auroc(lbl_t, scr_t, rng)
                a_x = _stratified_auroc(lbl_x, scr_x, rng)
                per_drop.append(a_t - a_x)
        boot_means[b] = float(np.mean(per_drop))

    drop_ci_low = float(np.percentile(boot_means, 2.5))
    drop_ci_high = float(np.percentile(boot_means, 97.5))
    drop_median = float(np.percentile(boot_means, 50.0))

    out = {
        "schema": "inspectnet_cx.cross_category_matrix.v1",
        "categories": cats,
        "matrix": matrix,
        "diagonal_auroc": diag,
        "off_diagonal_drop_summary": {
            "mean_point_drop": drop_mean,
            "std_point_drop": drop_std,
            "n_off_diagonal_cells": len(drops),
            "bootstrap_n": n_boot,
            "bootstrap_seed": 0,
            "bootstrap_drop_mean_median": drop_median,
            "bootstrap_drop_mean_ci_low": drop_ci_low,
            "bootstrap_drop_mean_ci_high": drop_ci_high,
        },
    }
    out_path = reports / "cross_padim_matrix.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
