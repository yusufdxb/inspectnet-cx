"""Unit tests for inspectnet_cx.calibration.threshold_analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from inspectnet_cx.calibration.threshold_analysis import analyze_thresholds


def _synthetic_scores(rng_seed: int = 0) -> tuple[list[int], list[float]]:
    rng = np.random.default_rng(rng_seed)
    normal = rng.normal(loc=0.2, scale=0.1, size=100)
    anomaly = rng.normal(loc=0.8, scale=0.15, size=50)
    scores = np.concatenate([normal, anomaly]).tolist()
    labels = [0] * 100 + [1] * 50
    return labels, scores


def test_analyze_thresholds_basic_structure() -> None:
    labels, scores = _synthetic_scores()
    report = analyze_thresholds(labels, scores)
    assert report["schema"] == "inspectnet_cx.threshold_analysis.v1"
    assert report["n_samples"] == 150
    assert report["n_positive"] == 50
    assert report["n_negative"] == 100
    for key in ("youden", "f1_max", "auroc", "roc_curve", "target_fpr"):
        assert key in report


def test_analyze_thresholds_auroc_separable() -> None:
    # Anomaly scores cleanly separated from normal scores should give AUROC ~= 1.
    labels = [0] * 50 + [1] * 50
    scores = [float(i) * 0.01 for i in range(50)] + [
        1.0 + float(i) * 0.01 for i in range(50)
    ]
    report = analyze_thresholds(labels, scores)
    assert report["auroc"] == pytest.approx(1.0, abs=1e-6)
    youden = report["youden"]
    assert youden["tpr"] == pytest.approx(1.0, abs=1e-6)
    assert youden["fpr"] == pytest.approx(0.0, abs=1e-6)
    assert youden["f1"] == pytest.approx(1.0, abs=1e-6)


def test_analyze_thresholds_auroc_random_is_near_half() -> None:
    rng = np.random.default_rng(42)
    labels = ([0] * 500) + ([1] * 500)
    scores = rng.random(1000).tolist()
    report = analyze_thresholds(labels, scores)
    # Random scores -> AUROC should be near 0.5 (within sampling noise).
    assert 0.40 < report["auroc"] < 0.60


def test_analyze_thresholds_target_fpr_respects_ceiling() -> None:
    labels, scores = _synthetic_scores()
    report = analyze_thresholds(labels, scores, target_fprs=(0.05, 0.10))
    for key, point in report["target_fpr"].items():
        target = float(key.split("<=")[1])
        assert point["fpr"] <= target + 1e-9, (
            f"target {key} exceeded its ceiling: fpr={point['fpr']}"
        )


def test_analyze_thresholds_rejects_empty() -> None:
    with pytest.raises(ValueError):
        analyze_thresholds([], [])


def test_analyze_thresholds_rejects_single_class() -> None:
    with pytest.raises(ValueError):
        analyze_thresholds([0, 0, 0], [0.1, 0.2, 0.3])


def test_analyze_thresholds_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        analyze_thresholds([0, 1], [0.1, 0.2, 0.3])


def test_analyze_thresholds_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError):
        analyze_thresholds([0, 2], [0.1, 0.9])


def test_threshold_analysis_cli_roundtrip(tmp_path: Path) -> None:
    labels, scores = _synthetic_scores()
    score_file = tmp_path / "scores.json"
    score_file.write_text(
        json.dumps(
            {
                "schema": "inspectnet_cx.scores.v1",
                "items": [
                    {"path": f"img_{i}.png", "label": labels[i], "score": scores[i]}
                    for i in range(len(labels))
                ],
            }
        )
    )
    output = tmp_path / "report.json"
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "threshold_analysis.py"),
        "--input",
        str(score_file),
        "--output",
        str(output),
    ]
    env_pythonpath = str(repo_root / "src")
    result = subprocess.run(
        cmd,
        env={"PYTHONPATH": env_pythonpath, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=True,
    )
    assert output.exists()
    payload = json.loads(output.read_text())
    assert payload["schema"] == "inspectnet_cx.threshold_analysis.v1"
    assert payload["source_score_file"] == str(score_file)
    assert "youden" in payload
    assert b"\"youden\"" in result.stdout
