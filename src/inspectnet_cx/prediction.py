from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inspectnet_cx.data.dataset_check import COMMON_IMAGE_EXTENSIONS

NORMAL_NAMES = {"good", "normal"}
DEFAULT_PADIM_CHECKPOINT = Path(
    "artifacts/verification/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt"
)


def predict_images(
    input_path: Path,
    dataset_root: Path,
    dataset: str,
    category: str,
    backend: str,
    output: Path,
    checkpoint: Path | None = None,
    image_size: int = 128,
    threshold_quantile: float = 0.995,
) -> dict[str, Any]:
    """Run image-level anomaly prediction for real local images."""

    image_paths = _resolve_input_images(input_path)
    if not image_paths:
        msg = f"no supported images found under {input_path}"
        raise ValueError(msg)
    if backend == "classical_patchdiff":
        report = _predict_classical_patchdiff(
            image_paths=image_paths,
            dataset_root=dataset_root,
            dataset=dataset,
            category=category,
            output=output,
            image_size=image_size,
            threshold_quantile=threshold_quantile,
        )
    elif backend == "anomalib_padim":
        report = _predict_anomalib_padim(
            image_paths=image_paths,
            dataset_root=dataset_root,
            dataset=dataset,
            category=category,
            output=output,
            checkpoint=checkpoint or DEFAULT_PADIM_CHECKPOINT,
        )
    else:
        msg = "backend must be classical_patchdiff or anomalib_padim"
        raise ValueError(msg)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _predict_classical_patchdiff(
    *,
    image_paths: list[Path],
    dataset_root: Path,
    dataset: str,
    category: str,
    output: Path,
    image_size: int,
    threshold_quantile: float,
) -> dict[str, Any]:
    category_root = _category_root(dataset_root, dataset, category)
    train_images = _normal_train_images(category_root)
    if not train_images:
        msg = f"no normal training images found under {category_root}"
        raise ValueError(msg)

    train_stack = np.stack([_load_image(path, image_size) for path in train_images])
    mean = train_stack.mean(axis=0)
    std = np.maximum(train_stack.std(axis=0), 1.0 / 255.0)
    normal_scores = np.asarray([float(_score(_heatmap(image, mean, std))) for image in train_stack])
    threshold = float(np.quantile(normal_scores, threshold_quantile))

    mask_dir = output.parent / f"{output.stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    predictions = []
    for image_path in image_paths:
        image = _load_image(image_path, image_size)
        heatmap = _heatmap(image, mean, std)
        score = float(_score(heatmap))
        mask = heatmap >= threshold
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        _save_mask(mask, mask_path)
        predictions.append(
            {
                "path": str(image_path),
                "expected_label": _expected_label(image_path),
                "image_score": score,
                "predicted_label": "anomaly" if score >= threshold else "normal",
                "predicted_anomaly": bool(score >= threshold),
                "mask_path": str(mask_path),
                "mask_semantics": "classical_patchdiff heatmap >= normal-score threshold",
            }
        )
    elapsed_s = time.perf_counter() - started
    return {
        "status": "completed_predictions",
        "backend": "classical_patchdiff",
        "dataset": dataset,
        "category": category,
        "dataset_root": str(dataset_root.expanduser()),
        "category_root": str(category_root),
        "input_count": len(image_paths),
        "train_normal_count": len(train_images),
        "threshold_info": {
            "threshold": threshold,
            "threshold_quantile": threshold_quantile,
            "threshold_source": "normal training image score quantile",
        },
        "latency_ms_per_image": float(elapsed_s / max(1, len(image_paths)) * 1000.0),
        "predictions": predictions,
        "proof_note": (
            "Prediction path for a dependency-light classical sanity baseline. This is real "
            "inference on local images, but it is not the primary Anomalib PaDiM benchmark."
        ),
    }


def _predict_anomalib_padim(
    *,
    image_paths: list[Path],
    dataset_root: Path,
    dataset: str,
    category: str,
    output: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    if dataset != "mvtec_ad":
        msg = "anomalib_padim prediction is currently wired for mvtec_ad only"
        raise ValueError(msg)
    checkpoint = checkpoint.expanduser()
    if not checkpoint.exists():
        msg = f"Anomalib checkpoint not found: {checkpoint}"
        raise FileNotFoundError(msg)

    try:
        import torch
        from anomalib.engine import Engine  # type: ignore[import-not-found]
        from anomalib.models import Padim  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency failure path.
        msg = f"could not import Anomalib prediction stack: {type(exc).__name__}: {exc}"
        raise RuntimeError(msg) from exc

    mask_dir = output.parent / f"{output.stem}_masks"
    map_dir = output.parent / f"{output.stem}_maps"
    mask_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    engine = Engine(accelerator="cpu", devices=1, logger=False)
    predictions = []
    started = time.perf_counter()
    for image_path in image_paths:
        batches = engine.predict(
            model=model,
            ckpt_path=checkpoint,
            data_path=image_path,
            return_predictions=True,
        )
        if not batches:
            continue
        batch = batches[0]
        score = _scalar(batch.pred_score)
        pred_anomaly = bool(_scalar(batch.pred_label))
        mask = _to_numpy(batch.pred_mask)[0].astype(bool)
        anomaly_map = _to_numpy(batch.anomaly_map)[0].astype(np.float32)
        mask_path = mask_dir / f"{image_path.stem}_mask.png"
        map_path = map_dir / f"{image_path.stem}_anomaly_map.png"
        _save_mask(mask, mask_path)
        _save_heatmap(anomaly_map, map_path)
        predictions.append(
            {
                "path": str(image_path),
                "expected_label": _expected_label(image_path),
                "image_score": score,
                "predicted_label": "anomaly" if pred_anomaly else "normal",
                "predicted_anomaly": pred_anomaly,
                "mask_path": str(mask_path),
                "anomaly_map_path": str(map_path),
                "threshold_info": {
                    "threshold_source": "Anomalib checkpoint post_processor",
                    "threshold_value": None,
                },
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elapsed_s = time.perf_counter() - started
    return {
        "status": "completed_predictions",
        "backend": "anomalib_padim",
        "dataset": dataset,
        "category": category,
        "dataset_root": str(dataset_root.expanduser()),
        "checkpoint": str(checkpoint),
        "input_count": len(image_paths),
        "latency_ms_per_image": float(elapsed_s / max(1, len(image_paths)) * 1000.0),
        "predictions": predictions,
        "proof_note": (
            "Reusable Anomalib PaDiM checkpoint inference on local images. This is not an "
            "InspectNet-CX-native exported model and does not prove edge deployment."
        ),
    }


def _resolve_input_images(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser()
    if input_path.is_file() and input_path.suffix.lower() in COMMON_IMAGE_EXTENSIONS:
        return [input_path]
    if input_path.is_dir():
        return [
            path
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in COMMON_IMAGE_EXTENSIONS
        ]
    return []


def _category_root(dataset_root: Path, dataset: str, category: str) -> Path:
    return dataset_root.expanduser() / dataset / category


def _normal_train_images(category_root: Path) -> list[Path]:
    images: list[Path] = []
    for subdir in (Path("train/good"), Path("train/normal"), Path("normal")):
        path = category_root / subdir
        if path.exists():
            images.extend(_resolve_input_images(path))
    return sorted(images)


def _load_image(path: Path, image_size: int) -> np.ndarray:
    image = (
        Image.open(path)
        .convert("RGB")
        .resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def _heatmap(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.abs((image - mean) / std).mean(axis=2)


def _score(heatmap: np.ndarray) -> float:
    return float(np.quantile(heatmap, 0.99))


def _expected_label(path: Path) -> str:
    parent = path.parent.name
    return "normal" if parent in NORMAL_NAMES else "anomaly"


def _save_mask(mask: np.ndarray, path: Path) -> None:
    arr = mask.astype(np.uint8) * 255
    Image.fromarray(arr).save(path)


def _save_heatmap(heatmap: np.ndarray, path: Path) -> None:
    values = heatmap.astype(np.float32)
    span = float(values.max() - values.min())
    if span <= 1.0e-12:
        arr = np.zeros_like(values, dtype=np.uint8)
    else:
        arr = ((values - values.min()) / span * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "cpu"):
        return value.cpu().numpy()
    return np.asarray(value)


def _scalar(value: Any) -> float:
    arr = _to_numpy(value)
    return float(arr.reshape(-1)[0])
