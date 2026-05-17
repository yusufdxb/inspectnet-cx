from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor
from inspectnet_cx.calibration.normal_quantile import NormalQuantileCalibrator
from inspectnet_cx.data.dataset_check import COMMON_IMAGE_EXTENSIONS

NORMAL_SUBDIRS = (
    Path("train/good"),
    Path("train/normal"),
    Path("normal"),
    Path("Data/Images/Normal"),
)


def find_normal_images(dataset_root: Path, dataset: str, category: str) -> list[Path]:
    dataset_path = dataset_root.expanduser() / dataset
    category_paths = [dataset_path / category]
    if dataset == "visa":
        category_paths.insert(0, dataset_path / "visa_pytorch" / category)

    images = []
    for category_path in category_paths:
        for subdir in NORMAL_SUBDIRS:
            normal_dir = category_path / subdir
            if normal_dir.exists():
                images.extend(_iter_images(normal_dir))
        if images:
            break
    return sorted(images)


def build_normal_calibration_report(
    model_dir: Path,
    dataset_root: Path,
    dataset: str,
    category: str,
    quantile: float = 0.995,
    batch_size: int = 8,
    device: str = "cpu",
    max_images: int | None = None,
) -> dict[str, Any]:
    images = find_normal_images(dataset_root, dataset, category)
    if max_images is not None:
        images = images[:max_images]
    if not images:
        return {
            "status": "blocked",
            "blocked_reasons": [
                (
                    "no normal validation/training images found; expected paths such as "
                    f"{dataset_root / dataset / category / 'train' / 'good'}"
                )
            ],
            "dataset": dataset,
            "category": category,
            "dataset_root": str(dataset_root.expanduser()),
            "model_dir": str(model_dir.expanduser()),
            "threshold_source": "none",
            "proof_note": (
                "No threshold was calibrated. Real calibration requires normal-only validation "
                "images and later threshold-dependent evaluation on held-out labeled test data."
            ),
        }

    runtime_device = _resolve_device(device)
    processor = InspectNetCXProcessor.from_pretrained(model_dir.expanduser())
    model = InspectNetCXForAnomalyDetection.from_pretrained(model_dir.expanduser())
    model.to(runtime_device)
    model.eval()

    scores = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch_paths = images[start : start + batch_size]
            pil_images = [Image.open(path) for path in batch_paths]
            inputs = processor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(runtime_device)
            output = model(pixel_values=pixel_values)
            if output.image_score is None:
                msg = "model did not return image_score"
                raise RuntimeError(msg)
            scores.append(output.image_score.detach().cpu())

    normal_scores = torch.cat(scores)
    calibrator = NormalQuantileCalibrator(quantile=quantile)
    threshold = calibrator.fit(normal_scores)
    return {
        "status": "calibrated_phase0_threshold",
        "dataset": dataset,
        "category": category,
        "dataset_root": str(dataset_root.expanduser()),
        "model_dir": str(model_dir.expanduser()),
        "normal_image_count": len(images),
        "quantile": quantile,
        "threshold": float(threshold.item()),
        "score_min": float(normal_scores.min().item()),
        "score_median": float(normal_scores.median().item()),
        "score_max": float(normal_scores.max().item()),
        "threshold_source": "normal_only_local_split",
        "proof_note": (
            "This calibrates a threshold from normal images only. It does not prove anomaly "
            "detection quality, F1, AUROC, or production readiness."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a Phase 0 threshold from normal-only local images."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--dataset", required=True, choices=("mvtec_ad", "visa"))
    parser.add_argument("--category", required=True)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--output", type=Path, default=Path("reports/normal_threshold.json"))
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_normal_calibration_report(
        model_dir=args.model,
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        category=args.category,
        quantile=args.quantile,
        batch_size=args.batch_size,
        device=args.device,
        max_images=args.max_images,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)


def _iter_images(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    ]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA requested but torch.cuda.is_available() is false."
        raise RuntimeError(msg)
    return torch.device(device)


if __name__ == "__main__":
    main()
