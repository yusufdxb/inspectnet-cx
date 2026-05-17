from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inspectnet_cx.data.layout import COMMON_IMAGE_EXTENSIONS, resolve_dataset_layout

CLASSICAL_METHOD = "classical-range"
CLASSICAL_METHOD_ALIASES = (CLASSICAL_METHOD, "classical-mahalanobis")
METHODS = ("patchcore", "efficientad", "padim", "simplenet", *CLASSICAL_METHOD_ALIASES)
DATASETS = ("mvtec_ad", "visa", "mvtec_ad2", "mvtec_loco")
CLASSICAL_THRESHOLD_FLOOR = 1.0e-4
ANOMALIB_MODELS = {
    "patchcore": "Patchcore",
    "efficientad": "EfficientAd",
    "padim": "Padim",
    "simplenet": "Simplenet",
}
ANOMALIB_DATA = {
    "mvtec_ad": "MVTecAD",
    "visa": "Visa",
    "mvtec_ad2": "MVTecAD2",
    "mvtec_loco": "MVTecLOCO",
}


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    if args.method in CLASSICAL_METHOD_ALIASES:
        return build_classical_result(args)

    if not args.plan_only and args.method == "padim":
        return build_anomalib_padim_result(args)

    dataset_path = args.data_root.expanduser() / args.dataset
    anomalib_command = build_anomalib_command(args)
    status = "phase0_placeholder"
    blockers = []
    if args.plan_only:
        status = "blocked_plan_only"
        if not dataset_path.exists():
            blockers.append(f"dataset path not found: {dataset_path}")
        blockers.append("real Anomalib execution is not implemented by this Phase 0 harness")

    return {
        "method": args.method,
        "dataset": args.dataset,
        "category": args.category,
        "device": args.device,
        "dataset_path": str(dataset_path),
        "anomalib_model": ANOMALIB_MODELS[args.method],
        "anomalib_data": ANOMALIB_DATA[args.dataset],
        "anomalib_command": anomalib_command,
        "image_auroc": "TBD",
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": "TBD",
        "peak_vram_mb": "TBD",
        "model_size_mb": "TBD",
        "status": status,
        "blocked_reasons": blockers,
        "proof_note": (
            "This file is a command/result contract only. It is not benchmark evidence until "
            "Anomalib actually trains/tests on the named dataset and numeric metrics replace TBD."
        ),
    }


def build_anomalib_padim_result(args: argparse.Namespace) -> dict[str, Any]:
    """Run a real Anomalib PaDiM fit/test cycle on a local MVTec AD category."""

    if args.dataset != "mvtec_ad":
        return _blocked_anomalib_result(
            args,
            [
                "real local Anomalib execution is currently wired only for "
                f"mvtec_ad, got {args.dataset}"
            ],
        )

    try:
        from anomalib.data import MVTecAD
        from anomalib.engine import Engine
        from anomalib.models import Padim
    except Exception as exc:  # pragma: no cover - exercised only when optional deps are absent.
        return _blocked_anomalib_result(
            args,
            [f"could not import Anomalib PaDiM stack: {type(exc).__name__}: {exc}"],
        )

    dataset_path, category_path = _resolve_category_path(
        args.data_root,
        args.dataset,
        args.category,
    )
    train_images = _find_training_normal_images(category_path)
    test_samples = _find_test_samples(category_path)
    blocked_reasons = []
    if not train_images:
        blocked_reasons.append(f"no normal training images found under {category_path}")
    if not test_samples:
        blocked_reasons.append(f"no labeled test images found under {category_path / 'test'}")
    if blocked_reasons:
        return _blocked_anomalib_result(args, blocked_reasons)

    layers = [item.strip() for item in args.layers.split(",") if item.strip()]
    run_root = args.run_root or (
        Path("reports") / "anomalib" / f"padim_{args.dataset}_{args.category}"
    )
    run_root = run_root.expanduser()
    accelerator = "gpu" if args.device == "cuda" else "cpu"

    datamodule = MVTecAD(
        root=dataset_path,
        category=args.category,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = Padim(
        backbone=args.backbone,
        layers=layers,
        pre_trained=not args.no_pretrained,
    )
    engine = Engine(
        default_root_dir=run_root,
        accelerator=accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        logger=False,
    )

    start = time.perf_counter()
    metrics = engine.train(model=model, datamodule=datamodule)
    elapsed_s = time.perf_counter() - start
    metric_rows = [_jsonify_metrics(row) for row in metrics]
    checkpoints = sorted(str(path) for path in run_root.rglob("*.ckpt"))

    primary = metric_rows[0] if metric_rows else {}
    return {
        "method": "padim",
        "baseline_kind": "anomalib",
        "baseline_version": "anomalib-padim-v1",
        "dataset": args.dataset,
        "category": args.category,
        "device": accelerator,
        "dataset_path": str(dataset_path),
        "category_path": str(category_path),
        "anomalib_model": "Padim",
        "anomalib_data": "MVTecAD",
        "backbone": args.backbone,
        "pre_trained": not args.no_pretrained,
        "layers": layers,
        "train_image_count": len(train_images),
        "test_image_count": len(test_samples),
        "normal_test_count": sum(1 for sample in test_samples if sample["label"] == 0),
        "anomaly_test_count": sum(1 for sample in test_samples if sample["label"] == 1),
        "image_auroc": primary.get("image_AUROC", "TBD"),
        "image_f1": primary.get("image_F1Score", "TBD"),
        "pixel_auroc": primary.get("pixel_AUROC", "TBD"),
        "pixel_f1": primary.get("pixel_F1Score", "TBD"),
        "au_pro": primary.get("pixel_AUPRO", "TBD"),
        "latency_ms_per_image": "TBD",
        "peak_vram_mb": "TBD" if accelerator == "gpu" else 0.0,
        "model_size_mb": "TBD",
        "status": "completed_anomalib_padim",
        "blocked_reasons": [],
        "elapsed_s": elapsed_s,
        "metrics": metric_rows,
        "run_root": str(run_root),
        "checkpoints": checkpoints,
        "proof_note": (
            "Real Anomalib PaDiM fit/test on the local MVTec AD category. Metrics are local "
            "benchmark evidence for this exact dataset, model configuration, and environment. "
            "They do not prove workstation deployment readiness beyond the measured numbers."
        ),
    }


def _blocked_anomalib_result(
    args: argparse.Namespace,
    blocked_reasons: list[str],
) -> dict[str, Any]:
    dataset_path = args.data_root.expanduser() / args.dataset
    return {
        "method": args.method,
        "baseline_kind": "anomalib",
        "dataset": args.dataset,
        "category": args.category,
        "device": args.device,
        "dataset_path": str(dataset_path),
        "status": "blocked",
        "blocked_reasons": blocked_reasons,
        "image_auroc": "TBD",
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": "TBD",
        "peak_vram_mb": "TBD",
        "model_size_mb": "TBD",
        "proof_note": "No Anomalib benchmark evidence was produced because execution was blocked.",
    }


def _jsonify_metrics(row: dict[str, Any]) -> dict[str, float | int | str | bool | None]:
    out: dict[str, float | int | str | bool | None] = {}
    for key, value in row.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (float, int, str, bool)) or value is None:
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


def build_classical_result(args: argparse.Namespace) -> dict[str, Any]:
    if args.image_size < 8 or args.image_size % 8 != 0:
        msg = "--image-size must be a multiple of 8 and at least 8 for pooled features."
        raise ValueError(msg)
    if not 0.0 < args.threshold_quantile < 1.0:
        msg = "--threshold-quantile must be between 0 and 1."
        raise ValueError(msg)

    dataset_path, category_path = _resolve_category_path(
        args.data_root,
        args.dataset,
        args.category,
    )
    train_images = _find_training_normal_images(category_path)
    test_samples = _find_test_samples(category_path)
    if args.max_train_images is not None:
        train_images = train_images[: args.max_train_images]
    if args.max_test_images is not None:
        test_samples = test_samples[: args.max_test_images]

    blocked_reasons = []
    if not train_images:
        blocked_reasons.append(
            "no normal training images found; expected train/good, train/normal, or normal"
        )
    if not test_samples:
        blocked_reasons.append("no labeled test images found under test/<defect_type>")
    if test_samples and len({sample["label"] for sample in test_samples}) < 2:
        blocked_reasons.append(
            "test split must contain at least one normal and one anomaly image for AUROC/F1"
        )
    if blocked_reasons:
        return {
            "method": args.method,
            "baseline_kind": "classical_cpu_image",
            "baseline_version": "classical-range-v1",
            "dataset": args.dataset,
            "category": args.category,
            "device": "cpu",
            "dataset_path": str(dataset_path),
            "category_path": str(category_path),
            "status": "blocked",
            "blocked_reasons": blocked_reasons,
            "image_auroc": "TBD",
            "pixel_auroc": "TBD",
            "au_pro": "TBD",
            "pixel_f1": "TBD",
            "latency_ms_per_image": "TBD",
            "peak_vram_mb": "TBD",
            "model_size_mb": "TBD",
            "proof_note": (
                "Classical CPU baseline did not run because local data was incomplete. This is "
                "not benchmark evidence."
            ),
        }

    train_features = np.stack([_extract_features(path, args.image_size) for path in train_images])
    model = _fit_normal_range_model(train_features)
    train_scores = _score_features(train_features, model)
    threshold = max(
        float(np.quantile(train_scores, args.threshold_quantile)),
        CLASSICAL_THRESHOLD_FLOOR,
    )

    predictions = []
    elapsed_s = 0.0
    for sample in test_samples:
        start = time.perf_counter()
        feature = _extract_features(sample["path"], args.image_size)
        score = float(_score_features(feature[None, :], model)[0])
        elapsed_s += time.perf_counter() - start
        predictions.append(
            {
                "path": str(sample["path"]),
                "label": sample["label_name"],
                "target": int(sample["label"]),
                "image_score": score,
                "predicted_anomaly": bool(score >= threshold),
            }
        )

    targets = np.asarray([row["target"] for row in predictions], dtype=np.int64)
    scores = np.asarray([row["image_score"] for row in predictions], dtype=np.float64)
    decisions = np.asarray([row["predicted_anomaly"] for row in predictions], dtype=bool)
    confusion = _confusion(targets, decisions)
    image_f1 = _binary_f1(
        confusion["true_positives"],
        confusion["false_positives"],
        confusion["false_negatives"],
    )
    accuracy = float(np.mean(decisions == targets.astype(bool)))
    latency_ms = (elapsed_s / max(1, len(predictions))) * 1000.0

    return {
        "method": args.method,
        "baseline_kind": "classical_cpu_image",
        "baseline_version": "classical-range-v1",
        "dataset": args.dataset,
        "category": args.category,
        "device": "cpu",
        "dataset_path": str(dataset_path),
        "category_path": str(category_path),
        "image_size": args.image_size,
        "feature_count": int(train_features.shape[1]),
        "train_image_count": len(train_images),
        "test_image_count": len(predictions),
        "normal_test_count": int(np.sum(targets == 0)),
        "anomaly_test_count": int(np.sum(targets == 1)),
        "threshold_source": "normal_train_quantile",
        "threshold_floor": CLASSICAL_THRESHOLD_FLOOR,
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "train_score_min": float(np.min(train_scores)),
        "train_score_median": float(np.median(train_scores)),
        "train_score_max": float(np.max(train_scores)),
        "image_auroc": _binary_auroc(targets, scores),
        "image_f1": image_f1,
        "accuracy": accuracy,
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": latency_ms,
        "peak_vram_mb": 0.0,
        "model_size_mb": 0.0,
        "status": "completed_classical_cpu_baseline",
        "blocked_reasons": [],
        "metrics_scope": "image_level_only",
        "confusion": confusion,
        "predictions": predictions,
        "proof_note": (
            "CPU-only normal-range baseline over simple image features. Metrics are local "
            "image-level evidence for this exact dataset path only; pixel metrics, trained model "
            "quality, hardware latency, and factory readiness are not proven."
        ),
    }


def build_anomalib_command(args: argparse.Namespace) -> list[str]:
    dataset_path = args.data_root.expanduser() / args.dataset
    return [
        "anomalib",
        "train",
        "--model",
        ANOMALIB_MODELS[args.method],
        "--data",
        ANOMALIB_DATA[args.dataset],
        "--data.root",
        str(dataset_path),
        "--data.category",
        args.category,
        "--trainer.accelerator",
        "gpu" if args.device == "cuda" else "cpu",
    ]


def _resolve_category_path(data_root: Path, dataset: str, category: str) -> tuple[Path, Path]:
    layout = resolve_dataset_layout(data_root, dataset)
    dataset_path = layout.dataset_path
    category_path = dataset_path / category
    if dataset == "visa" and (dataset_path / "visa_pytorch" / category).exists():
        category_path = dataset_path / "visa_pytorch" / category
    return dataset_path, category_path


def _find_training_normal_images(category_path: Path) -> list[Path]:
    images = []
    for subdir in (Path("train/good"), Path("train/normal"), Path("normal")):
        path = category_path / subdir
        if path.exists():
            images.extend(_iter_images(path))
    return sorted(images)


def _find_test_samples(category_path: Path) -> list[dict[str, Any]]:
    test_root = category_path / "test"
    if not test_root.exists():
        return []
    samples = []
    for path in sorted(_iter_images(test_root)):
        label_name = path.parent.name
        is_normal = label_name in {"good", "normal"}
        samples.append(
            {
                "path": path,
                "label": 0 if is_normal else 1,
                "label_name": "normal" if is_normal else "anomaly",
            }
        )
    return samples


def _iter_images(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    ]


def _extract_features(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    grad_y, grad_x = np.gradient(gray)
    pooled_gray = gray.reshape(8, image_size // 8, 8, image_size // 8).mean(axis=(1, 3)).reshape(-1)
    return np.concatenate(
        [
            arr.mean(axis=(0, 1)),
            arr.std(axis=(0, 1)),
            np.asarray([gray.mean(), gray.std(), np.abs(grad_x).mean(), np.abs(grad_y).mean()]),
            pooled_gray,
        ]
    ).astype(np.float32)


def _fit_normal_range_model(features: np.ndarray) -> dict[str, np.ndarray]:
    feature_min = features.min(axis=0)
    feature_max = features.max(axis=0)
    feature_range = feature_max - feature_min
    return {
        "min": feature_min,
        "max": feature_max,
        "scale": np.maximum.reduce(
            [
                features.std(axis=0),
                feature_range / 2.0,
                np.full(features.shape[1], 1.0e-3, dtype=np.float32),
            ]
        ),
    }


def _score_features(features: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    below_normal_range = np.maximum(model["min"] - features, 0.0)
    above_normal_range = np.maximum(features - model["max"], 0.0)
    normalized_distance = (below_normal_range + above_normal_range) / model["scale"]
    return np.sqrt(np.mean(np.square(normalized_distance), axis=1))


def _binary_auroc(targets: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[targets == 1]
    negatives = scores[targets == 0]
    if len(positives) == 0 or len(negatives) == 0:
        msg = "AUROC requires at least one positive and one negative sample."
        raise ValueError(msg)
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for positive in positives:
        wins += float(np.sum(positive > negatives))
        wins += 0.5 * float(np.sum(positive == negatives))
    return wins / total


def _confusion(targets: np.ndarray, decisions: np.ndarray) -> dict[str, int]:
    positives = targets.astype(bool)
    return {
        "true_positives": int(np.sum(decisions & positives)),
        "false_positives": int(np.sum(decisions & ~positives)),
        "true_negatives": int(np.sum(~decisions & ~positives)),
        "false_negatives": int(np.sum(~decisions & positives)),
    }


def _binary_f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
    denominator = (2 * true_positives) + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return (2 * true_positives) / denominator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline runner. PaDiM executes a real Anomalib fit/test on local MVTec AD; "
            "classical-range runs a CPU-only local image baseline; other Anomalib methods "
            "currently emit command scaffolds unless --plan-only is used."
        )
    )
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--category", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the placeholder JSON without writing a file.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Emit the concrete Anomalib command that should be run later. This does not train or "
            "evaluate a baseline."
        ),
    )
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-test-images", type=int)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--layers", default="layer1,layer2,layer3")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet-pretrained backbone weights for fully offline PaDiM smoke checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = build_result(args)
    payload = json.dumps(result, indent=2) + "\n"

    if args.dry_run:
        print(payload, end="")
        return

    output = args.output
    if output is None:
        if args.method in CLASSICAL_METHOD_ALIASES:
            output = (
                Path("reports")
                / "baseline_classical_range_v1"
                / f"{args.dataset}_{args.category}"
                / "result.json"
            )
        else:
            output = Path("reports") / f"{args.method}_{args.dataset}_{args.category}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    print(f"Wrote baseline result to {output}")


if __name__ == "__main__":
    main()
