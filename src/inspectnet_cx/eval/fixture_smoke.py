from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor
from inspectnet_cx.calibration.normal_split import build_normal_calibration_report
from inspectnet_cx.data.dataset_check import check_datasets
from inspectnet_cx.eval.classical_baseline import run_classical_baseline
from inspectnet_cx.eval.validate_results import validate_results
from inspectnet_cx.release.create_phase0_model import create_phase0_model


def create_tiny_mvtec_fixture(root: Path, category: str = "bottle") -> dict[str, Any]:
    dataset_root = root.expanduser()
    normal_dir = dataset_root / "mvtec_ad" / category / "train" / "good"
    test_good = dataset_root / "mvtec_ad" / category / "test" / "good"
    test_defect = dataset_root / "mvtec_ad" / category / "test" / "scratch"
    for path in (normal_dir, test_good, test_defect):
        path.mkdir(parents=True, exist_ok=True)
    _write_image(normal_dir / "000.png", defect=False)
    _write_image(normal_dir / "001.png", defect=False, shade=170)
    _write_image(test_good / "000.png", defect=False, shade=150)
    _write_image(test_defect / "000.png", defect=True)
    return {
        "status": "fixture_created",
        "dataset": "mvtec_ad",
        "category": category,
        "dataset_root": str(dataset_root),
        "proof_note": "Tiny synthetic local fixture for smoke tests only; not benchmark evidence.",
    }


def run_fixture_smoke(
    output_dir: Path,
    *,
    category: str = "bottle",
    image_size: int = 32,
) -> dict[str, Any]:
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = output_dir / "datasets"
    model_dir = output_dir / "model"
    create_tiny_mvtec_fixture(dataset_root, category=category)
    dataset_check = check_datasets(dataset_root)
    create_phase0_model(model_dir, image_size=image_size)
    calibration = build_normal_calibration_report(
        model_dir=model_dir,
        dataset_root=dataset_root,
        dataset="mvtec_ad",
        category=category,
        quantile=0.5,
        batch_size=2,
        device="cpu",
    )
    threshold = float(calibration["threshold"]) if calibration["status"] != "blocked" else None
    samples = _collect_fixture_samples(dataset_root, category)
    predictions = _score_samples(model_dir, samples, threshold)
    classical_baseline = run_classical_baseline(
        dataset_root=dataset_root,
        dataset="mvtec_ad",
        category=category,
        output=output_dir / "classical_patchdiff_fixture.json",
        image_size=image_size,
        quantile=0.5,
    )
    report = {
        "status": "fixture_smoke_completed",
        "dataset": "mvtec_ad",
        "category": category,
        "dataset_root": str(dataset_root),
        "model_dir": str(model_dir),
        "dataset_check": dataset_check,
        "calibration": calibration,
        "evaluation": _summarize_predictions(predictions),
        "classical_baseline": {
            "report_path": str(output_dir / "classical_patchdiff_fixture.json"),
            "status": classical_baseline["status"],
            "image_auroc": classical_baseline["image_auroc"],
            "image_f1": classical_baseline.get("image_f1", "TBD"),
            "pixel_auroc": classical_baseline["pixel_auroc"],
            "pixel_f1": classical_baseline["pixel_f1"],
            "proof_note": classical_baseline["proof_note"],
        },
        "predictions": predictions,
        "result_validation": {"status": "not_run"},
        "proof_note": (
            "This exercises dataset loading, normal-only calibration, inference, and threshold "
            "application on a tiny synthetic fixture. It is not real anomaly-detection evidence."
        ),
    }
    report_path = output_dir / "fixture_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    validation_failures = validate_results(output_dir)
    report["result_validation"] = {
        "status": "passed" if not validation_failures else "failed",
        "input_dir": str(output_dir),
        "failures": validation_failures,
        "proof_note": (
            "Validates that non-benchmark fixture proof reports are not misread as "
            "benchmark result JSON."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiny fixture-backed InspectNet-CX smoke evidence."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/fixture_smoke"))
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--image-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_fixture_smoke(args.output_dir, category=args.category, image_size=args.image_size)
    print(json.dumps(report, indent=2))


def _write_image(path: Path, *, defect: bool, shade: int = 128) -> None:
    image = Image.new("RGB", (48, 48), (shade, shade, shade))
    if defect:
        draw = ImageDraw.Draw(image)
        draw.rectangle((16, 16, 32, 32), fill=(255, 40, 40))
    image.save(path)


def _collect_fixture_samples(dataset_root: Path, category: str) -> list[dict[str, Any]]:
    category_root = dataset_root / "mvtec_ad" / category / "test"
    samples = []
    for path in sorted(category_root.rglob("*.png")):
        samples.append(
            {
                "path": str(path),
                "label": "normal" if path.parent.name == "good" else "anomaly",
            }
        )
    return samples


def _score_samples(
    model_dir: Path,
    samples: list[dict[str, Any]],
    threshold: float | None,
) -> list[dict[str, Any]]:
    processor = InspectNetCXProcessor.from_pretrained(model_dir)
    model = InspectNetCXForAnomalyDetection.from_pretrained(model_dir).eval()
    rows = []
    with torch.inference_mode():
        for sample in samples:
            image = Image.open(sample["path"])
            inputs = processor(images=image, return_tensors="pt")
            output = model(**inputs, threshold=threshold)
            score = float(output.image_score.detach().cpu().item())
            threshold_value = float(output.threshold.detach().cpu().item())
            rows.append(
                {
                    **sample,
                    "image_score": score,
                    "threshold": threshold_value,
                    "predicted_anomaly": bool(score >= threshold_value),
                }
            )
    return rows


def _summarize_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    normal = [row for row in predictions if row["label"] == "normal"]
    anomaly = [row for row in predictions if row["label"] == "anomaly"]
    false_positive = sum(row["predicted_anomaly"] for row in normal)
    true_positive = sum(row["predicted_anomaly"] for row in anomaly)
    return {
        "sample_count": len(predictions),
        "normal_count": len(normal),
        "anomaly_count": len(anomaly),
        "true_positives": int(true_positive),
        "false_positives": int(false_positive),
        "fixture_accuracy": float(
            sum((row["label"] == "anomaly") == row["predicted_anomaly"] for row in predictions)
            / max(1, len(predictions))
        ),
    }


if __name__ == "__main__":
    main()
