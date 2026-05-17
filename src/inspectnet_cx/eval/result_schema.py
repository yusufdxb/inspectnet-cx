from __future__ import annotations

from typing import Any

REQUIRED_RESULT_FIELDS = {
    "method": str,
    "dataset": str,
    "category": str,
    "image_auroc": (float, int, str),
    "pixel_auroc": (float, int, str),
    "au_pro": (float, int, str),
    "pixel_f1": (float, int, str),
    "latency_ms_per_image": (float, int, str),
    "peak_vram_mb": (float, int, str),
    "model_size_mb": (float, int, str),
    "status": str,
}

CLASSICAL_NUMERIC_FIELDS = (
    "image_auroc",
    "image_f1",
    "accuracy",
    "threshold",
    "latency_ms_per_image",
    "peak_vram_mb",
    "model_size_mb",
)

CLASSICAL_PROPORTION_FIELDS = ("image_auroc", "image_f1", "accuracy")
CLASSICAL_POSITIVE_INT_FIELDS = ("train_image_count", "test_image_count", "feature_count")

PATCHDIFF_NUMERIC_FIELDS = (
    "image_f1",
    "threshold",
    "latency_ms_per_image",
    "peak_vram_mb",
    "model_size_mb",
)
PATCHDIFF_POSITIVE_INT_FIELDS = ("train_normal_count", "test_sample_count")

ANOMALIB_NUMERIC_FIELDS = (
    "image_auroc",
    "pixel_auroc",
    "pixel_f1",
    "fit_elapsed_s",
    "test_elapsed_s",
)
ANOMALIB_PROPORTION_FIELDS = ("image_auroc", "image_f1", "pixel_auroc", "pixel_f1")
ANOMALIB_POSITIVE_INT_FIELDS = ("train_normal_count", "test_sample_count")


def validate_result_payload(payload: dict[str, Any]) -> list[str]:
    errors = []
    for field, expected_type in REQUIRED_RESULT_FIELDS.items():
        if field not in payload:
            errors.append(f"missing field: {field}")
            continue
        if not isinstance(payload[field], expected_type):
            errors.append(f"invalid type for {field}: {type(payload[field]).__name__}")

    for metric in (
        "image_auroc",
        "pixel_auroc",
        "au_pro",
        "pixel_f1",
        "latency_ms_per_image",
        "peak_vram_mb",
        "model_size_mb",
    ):
        value = payload.get(metric)
        if isinstance(value, str) and value != "TBD":
            errors.append(f"{metric} string values must be TBD")
        if isinstance(value, int | float) and value < 0:
            errors.append(f"{metric} must be non-negative")
    if payload.get("status") == "completed_classical_cpu_baseline":
        errors.extend(_validate_classical_numeric_baseline(payload))
    if payload.get("status") == "classical_baseline_completed":
        errors.extend(_validate_patchdiff_classical_baseline(payload))
    if payload.get("status") == "completed_anomalib_baseline":
        errors.extend(_validate_anomalib_baseline(payload))
    return errors


def _validate_classical_numeric_baseline(payload: dict[str, Any]) -> list[str]:
    errors = []
    if payload.get("baseline_kind") != "classical_cpu_image":
        errors.append("baseline_kind must be classical_cpu_image")
    if not isinstance(payload.get("baseline_version"), str):
        errors.append("baseline_version must be a string")
    if payload.get("metrics_scope") != "image_level_only":
        errors.append("metrics_scope must be image_level_only")

    for field in CLASSICAL_NUMERIC_FIELDS:
        value = payload.get(field)
        if not _is_number(value):
            errors.append(f"{field} must be numeric for completed classical baselines")
            continue
        if value < 0:
            errors.append(f"{field} must be non-negative")

    for field in CLASSICAL_PROPORTION_FIELDS:
        value = payload.get(field)
        if _is_number(value) and not 0.0 <= value <= 1.0:
            errors.append(f"{field} must be between 0 and 1")

    for field in CLASSICAL_POSITIVE_INT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{field} must be an integer")
            continue
        if value <= 0:
            errors.append(f"{field} must be positive")

    for field in ("pixel_auroc", "au_pro", "pixel_f1"):
        if payload.get(field) != "TBD":
            errors.append(f"{field} must remain TBD for image-level classical baselines")
    return errors


def _validate_patchdiff_classical_baseline(payload: dict[str, Any]) -> list[str]:
    errors = []
    if payload.get("method") != "classical_patchdiff":
        errors.append("method must be classical_patchdiff")

    for field in PATCHDIFF_NUMERIC_FIELDS:
        value = payload.get(field)
        if not _is_number(value):
            errors.append(f"{field} must be numeric for completed classical baselines")
            continue
        if value < 0:
            errors.append(f"{field} must be non-negative")

    for field in PATCHDIFF_POSITIVE_INT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{field} must be an integer")
            continue
        if value <= 0:
            errors.append(f"{field} must be positive")

    if _has_both_image_classes(payload):
        value = payload.get("image_auroc")
        if not _is_number(value):
            errors.append("image_auroc must be numeric when normal and anomaly tests exist")

    for field in ("image_auroc", "image_f1", "pixel_auroc", "pixel_f1"):
        value = payload.get(field)
        if isinstance(value, str) and value != "TBD":
            errors.append(f"{field} string values must be TBD")
        if _is_number(value) and not 0.0 <= value <= 1.0:
            errors.append(f"{field} must be between 0 and 1")
    return errors


def _validate_anomalib_baseline(payload: dict[str, Any]) -> list[str]:
    errors = []
    if payload.get("baseline_kind") != "anomalib":
        errors.append("baseline_kind must be anomalib")

    for field in ANOMALIB_NUMERIC_FIELDS:
        value = payload.get(field)
        if not _is_number(value):
            errors.append(f"{field} must be numeric for completed Anomalib baselines")
            continue
        if value < 0:
            errors.append(f"{field} must be non-negative")

    for field in ANOMALIB_PROPORTION_FIELDS:
        value = payload.get(field)
        if _is_number(value) and not 0.0 <= value <= 1.0:
            errors.append(f"{field} must be between 0 and 1")

    for field in ANOMALIB_POSITIVE_INT_FIELDS:
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{field} must be an integer")
            continue
        if value <= 0:
            errors.append(f"{field} must be positive")
    return errors


def _has_both_image_classes(payload: dict[str, Any]) -> bool:
    normal_count = payload.get("test_normal_count")
    anomaly_count = payload.get("test_anomaly_count")
    return (
        isinstance(normal_count, int)
        and not isinstance(normal_count, bool)
        and normal_count > 0
        and isinstance(anomaly_count, int)
        and not isinstance(anomaly_count, bool)
        and anomaly_count > 0
    )


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
