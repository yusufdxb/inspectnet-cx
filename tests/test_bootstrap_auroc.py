"""Tests for the bootstrap AUROC script logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_auroc.py"
)
_spec = importlib.util.spec_from_file_location("bootstrap_auroc", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bootstrap_auroc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bootstrap_auroc)


def test_bootstrap_deterministic_separable() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.15, 0.25, 0.8, 0.9, 0.85, 0.95], dtype=np.float64)
    out = bootstrap_auroc.bootstrap_metrics(labels, scores, n_bootstrap=200, seed=42)
    assert out["auroc"]["point"] == 1.0
    assert out["auroc"]["ci_low"] >= 0.9
    assert out["auroc"]["ci_high"] == 1.0
    assert out["youden_f1"]["point"] == 1.0
    assert out["f1_max_f1"]["point"] == 1.0


def test_bootstrap_deterministic_seed() -> None:
    rng = np.random.default_rng(123)
    labels = (rng.random(60) > 0.5).astype(np.int64)
    if labels.sum() == 0 or labels.sum() == labels.size:
        labels[0] = 1
        labels[-1] = 0
    scores = rng.random(60) + 0.4 * labels
    a = bootstrap_auroc.bootstrap_metrics(labels, scores, n_bootstrap=300, seed=7)
    b = bootstrap_auroc.bootstrap_metrics(labels, scores, n_bootstrap=300, seed=7)
    assert a == b


def test_bootstrap_rejects_single_class() -> None:
    labels = np.zeros(10, dtype=np.int64)
    scores = np.linspace(0.0, 1.0, 10)
    try:
        bootstrap_auroc.bootstrap_metrics(labels, scores, n_bootstrap=10, seed=0)
    except ValueError as exc:
        assert "positive" in str(exc) or "negative" in str(exc)
    else:
        raise AssertionError("expected ValueError on single-class input")
