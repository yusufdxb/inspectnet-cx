from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inspectnet_cx.data.dataset_check import COMMON_IMAGE_EXTENSIONS

METHOD = "classical_patchdiff"
DATASETS = ("mvtec_ad", "visa")
NORMAL_NAMES = {"good", "normal"}


def run_classical_baseline(
    dataset_root: Path,
    dataset: str,
    category: str,
    output: Path | None = None,
    image_size: int = 128,
    quantile: float = 0.995,
    max_train_images: int | None = None,
    max_test_images: int | None = None,
) -> dict[str, Any]:
    """Fit a normal-only pixel baseline and evaluate a local MVTec-style category.

    This is intentionally classical and dependency-light. It is not a replacement for
    PatchCore/EfficientAD, but it produces real numeric evidence for dataset plumbing,
    thresholding, and result validation when Anomalib is unavailable.
    """

    dataset_root = dataset_root.expanduser()
    category_root = _category_root(dataset_root, dataset, category)
    train_images = _normal_train_images(category_root)
    test_samples = _test_samples(category_root)
    if max_train_images is not None:
        train_images = train_images[:max_train_images]
    if max_test_images is not None:
        test_samples = test_samples[:max_test_images]

    blockers = []
    if not train_images:
        blockers.append(f"no normal training images found under {category_root}")
    if not test_samples:
        blockers.append(f"no test images found under {category_root / 'test'}")
    if blockers:
        report = _blocked_report(dataset_root, dataset, category, category_root, blockers)
        if output is not None:
            _write_json(output, report)
        return report

    train_stack = np.stack([_load_image(path, image_size) for path in train_images])
    mean = train_stack.mean(axis=0)
    std = np.maximum(train_stack.std(axis=0), 1.0 / 255.0)
    normal_scores = np.asarray([float(_score(_heatmap(image, mean, std))) for image in train_stack])
    threshold = float(np.quantile(normal_scores, quantile))

    rows = []
    pixel_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    start_time = time.perf_counter()
    for sample in test_samples:
        image = _load_image(sample["path"], image_size)
        heatmap = _heatmap(image, mean, std)
        score = float(_score(heatmap))
        mask = _load_mask(category_root, sample, image_size)
        if mask is not None:
            pixel_scores.append(heatmap.reshape(-1))
            pixel_labels.append(mask.reshape(-1).astype(np.int64))
        rows.append(
            {
                "path": str(sample["path"]),
                "defect_type": sample["defect_type"],
                "label": sample["label"],
                "image_score": score,
                "predicted_anomaly": bool(score >= threshold),
                "mask_available": mask is not None,
            }
        )
    elapsed = time.perf_counter() - start_time

    y_true = np.asarray([1 if row["label"] == "anomaly" else 0 for row in rows], dtype=np.int64)
    y_score = np.asarray([row["image_score"] for row in rows], dtype=np.float64)
    y_pred = np.asarray([row["predicted_anomaly"] for row in rows], dtype=bool)
    image_auroc = _binary_auroc(y_true, y_score)
    pixel_auroc: float | str = "TBD"
    pixel_f1: float | str = "TBD"
    if pixel_scores and pixel_labels:
        flat_scores = np.concatenate(pixel_scores)
        flat_labels = np.concatenate(pixel_labels)
        if len(np.unique(flat_labels)) == 2:
            pixel_auroc = _binary_auroc(flat_labels, flat_scores)
            pixel_f1 = _binary_f1(flat_labels, flat_scores >= threshold)

    report = {
        "method": METHOD,
        "dataset": dataset,
        "category": category,
        "dataset_root": str(dataset_root),
        "category_root": str(category_root),
        "train_normal_count": len(train_images),
        "test_sample_count": len(rows),
        "test_normal_count": int((y_true == 0).sum()),
        "test_anomaly_count": int((y_true == 1).sum()),
        "threshold": threshold,
        "threshold_quantile": quantile,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "au_pro": "TBD",
        "pixel_f1": pixel_f1,
        "image_f1": _binary_f1(y_true, y_pred),
        "latency_ms_per_image": float(elapsed / max(1, len(rows)) * 1000.0),
        "peak_vram_mb": 0.0,
        "model_size_mb": float((mean.nbytes + std.nbytes) / 1_000_000.0),
        "status": "classical_baseline_completed",
        "samples": rows,
        "proof_note": (
            "This is a normal-only classical pixel-difference baseline. It is real numeric "
            "local evidence for this dataset layout, but it is not Anomalib/PatchCore evidence "
            "and must not be compared as a state-of-the-art anomaly detector."
        ),
    }
    if output is not None:
        _write_json(output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a dependency-light normal-only classical anomaly baseline."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--category", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output
    if output is None:
        output = Path("reports") / f"{METHOD}_{args.dataset}_{args.category}.json"
    report = run_classical_baseline(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        category=args.category,
        output=output,
        image_size=args.image_size,
        quantile=args.quantile,
        max_train_images=args.max_train_images,
        max_test_images=args.max_test_images,
    )
    print(json.dumps(report, indent=2))
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)


def _category_root(dataset_root: Path, dataset: str, category: str) -> Path:
    dataset_path = dataset_root / dataset
    if dataset == "visa" and (dataset_path / "visa_pytorch" / category).exists():
        return dataset_path / "visa_pytorch" / category
    return dataset_path / category


def _normal_train_images(category_root: Path) -> list[Path]:
    candidates = [
        category_root / "train" / "good",
        category_root / "train" / "normal",
        category_root / "normal",
        category_root / "Data" / "Images" / "Normal",
    ]
    images: list[Path] = []
    for path in candidates:
        if path.exists():
            images.extend(_iter_images(path))
    return sorted(images)


def _test_samples(category_root: Path) -> list[dict[str, Any]]:
    test_root = category_root / "test"
    if not test_root.exists():
        return []
    samples = []
    for path in sorted(_iter_images(test_root)):
        defect_type = path.parent.name
        label = "normal" if defect_type in NORMAL_NAMES else "anomaly"
        samples.append({"path": path, "defect_type": defect_type, "label": label})
    return samples


def _iter_images(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    ]


def _load_image(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (image_size, image_size),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def _load_mask(category_root: Path, sample: dict[str, Any], image_size: int) -> np.ndarray | None:
    if sample["label"] == "normal":
        return np.zeros((image_size, image_size), dtype=bool)
    path = Path(sample["path"])
    candidates = [
        category_root / "ground_truth" / sample["defect_type"] / f"{path.stem}_mask.png",
        category_root / "ground_truth" / sample["defect_type"] / f"{path.stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            mask = Image.open(candidate).convert("L").resize(
                (image_size, image_size),
                Image.Resampling.NEAREST,
            )
            return np.asarray(mask, dtype=np.uint8) > 0
    return None


def _heatmap(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.abs((image - mean) / std).mean(axis=2)


def _score(heatmap: np.ndarray) -> float:
    return float(np.quantile(heatmap, 0.99))


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float | str:
    labels = labels.astype(np.int64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return "TBD"
    wins = 0.0
    total = float(positives.size * negatives.size)
    for positive in positives:
        wins += float((positive > negatives).sum())
        wins += 0.5 * float((positive == negatives).sum())
    return float(wins / total)


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    tp = int((labels & predictions).sum())
    fp = int((~labels & predictions).sum())
    fn = int((labels & ~predictions).sum())
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def _blocked_report(
    dataset_root: Path,
    dataset: str,
    category: str,
    category_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "dataset": dataset,
        "category": category,
        "dataset_root": str(dataset_root),
        "category_root": str(category_root),
        "image_auroc": "TBD",
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": "TBD",
        "peak_vram_mb": "TBD",
        "model_size_mb": "TBD",
        "status": "blocked",
        "blocked_reasons": blockers,
        "proof_note": (
            "No baseline metrics were produced because required local dataset files are absent."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
