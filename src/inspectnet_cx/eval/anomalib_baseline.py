from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from inspectnet_cx.data.dataset_check import COMMON_IMAGE_EXTENSIONS

METHODS = ("padim", "patchcore")
DATASETS = ("mvtec_ad",)
NORMAL_NAMES = {"good", "normal"}


def run_anomalib_baseline(
    dataset_root: Path,
    dataset: str,
    category: str,
    method: str = "padim",
    device: str = "cpu",
    output: Path | None = None,
    work_dir: Path = Path("artifacts/anomalib"),
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, Any]:
    """Fit/test a real Anomalib baseline on a local MVTec-style category."""

    if dataset not in DATASETS:
        msg = f"unsupported dataset for real Anomalib runner: {dataset}"
        raise ValueError(msg)
    if method not in METHODS:
        msg = f"unsupported Anomalib method: {method}"
        raise ValueError(msg)
    if device not in {"cpu", "cuda", "auto"}:
        msg = "--device must be cpu, cuda, or auto"
        raise ValueError(msg)

    dataset_root = dataset_root.expanduser()
    work_dir = work_dir.expanduser()
    category_root = dataset_root / dataset / category
    counts = _count_mvtec_category(category_root)
    blockers = _dataset_blockers(category_root, counts)
    if blockers:
        report = _blocked_report(dataset_root, dataset, category, method, category_root, blockers)
        if output is not None:
            _write_json(output, report)
        return report

    try:
        from anomalib.data import MVTecAD  # type: ignore[import-not-found]
        from anomalib.engine import Engine  # type: ignore[import-not-found]
        from anomalib.models import Padim, Patchcore  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional dependency presence.
        report = _blocked_report(
            dataset_root,
            dataset,
            category,
            method,
            category_root,
            [f"Anomalib import failed: {type(exc).__name__}: {exc}"],
        )
        if output is not None:
            _write_json(output, report)
        return report

    datamodule = MVTecAD(
        root=dataset_root / dataset,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    model = (
        Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
        if method == "padim"
        else Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"))
    )
    engine = Engine(
        accelerator=_resolve_accelerator(device),
        devices=1,
        default_root_dir=work_dir,
        logger=False,
        max_epochs=1,
    )

    started = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_elapsed_s = time.perf_counter() - started
    test_started = time.perf_counter()
    metrics = [_to_jsonable(dict(row)) for row in engine.test(model=model, datamodule=datamodule)]
    test_elapsed_s = time.perf_counter() - test_started
    summary = _metric_summary(metrics)

    report = {
        "method": f"anomalib_{method}",
        "baseline_kind": "anomalib",
        "baseline_version": "anomalib-2.4.1",
        "dataset": dataset,
        "category": category,
        "dataset_root": str(dataset_root),
        "category_root": str(category_root),
        "device": device,
        "anomalib_model": method,
        "train_normal_count": counts["train_normal_count"],
        "test_sample_count": counts["test_sample_count"],
        "test_normal_count": counts["test_normal_count"],
        "test_anomaly_count": counts["test_anomaly_count"],
        "image_auroc": summary.get("image_auroc", "TBD"),
        "image_f1": summary.get("image_f1", "TBD"),
        "pixel_auroc": summary.get("pixel_auroc", "TBD"),
        "au_pro": summary.get("au_pro", "TBD"),
        "pixel_f1": summary.get("pixel_f1", "TBD"),
        "latency_ms_per_image": "TBD",
        "peak_vram_mb": 0.0 if device == "cpu" else "TBD",
        "model_size_mb": "TBD",
        "fit_elapsed_s": fit_elapsed_s,
        "test_elapsed_s": test_elapsed_s,
        "metrics": metrics,
        "status": "completed_anomalib_baseline",
        "blocked_reasons": [],
        "proof_note": (
            "Real Anomalib fit/test on local data. Metrics are valid only for this exact "
            "dataset path, category, package stack, and device; latency/model-size/deployment "
            "claims require separate runtime evidence."
        ),
    }
    if output is not None:
        _write_json(output, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real Anomalib baseline on local data.")
    parser.add_argument("--dataset-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--category", required=True)
    parser.add_argument("--method", default="padim", choices=METHODS)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/anomalib"))
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output
    if output is None:
        output = Path("reports") / f"anomalib_{args.method}_{args.dataset}_{args.category}.json"
    report = run_anomalib_baseline(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        category=args.category,
        method=args.method,
        device=args.device,
        output=output,
        work_dir=args.work_dir,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(report, indent=2))
    if args.fail_on_blocked and report["status"] == "blocked":
        raise SystemExit(1)


def _count_mvtec_category(category_root: Path) -> dict[str, int]:
    train_normal_count = sum(
        len(_iter_images(path))
        for path in (
            category_root / "train" / "good",
            category_root / "train" / "normal",
        )
        if path.exists()
    )
    test_normal_count = 0
    test_anomaly_count = 0
    test_root = category_root / "test"
    if test_root.exists():
        for path in sorted(test_root.iterdir()):
            if not path.is_dir():
                continue
            image_count = len(_iter_images(path))
            if path.name in NORMAL_NAMES:
                test_normal_count += image_count
            else:
                test_anomaly_count += image_count
    return {
        "train_normal_count": train_normal_count,
        "test_sample_count": test_normal_count + test_anomaly_count,
        "test_normal_count": test_normal_count,
        "test_anomaly_count": test_anomaly_count,
    }


def _dataset_blockers(category_root: Path, counts: dict[str, int]) -> list[str]:
    blockers = []
    if not category_root.exists():
        blockers.append(f"category path not found: {category_root}")
    if counts["train_normal_count"] == 0:
        blockers.append("no normal training images found under train/good or train/normal")
    if counts["test_normal_count"] == 0:
        blockers.append("no normal test images found under test/good or test/normal")
    if counts["test_anomaly_count"] == 0:
        blockers.append("no anomaly test images found under test/<defect_type>")
    return blockers


def _iter_images(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    ]


def _resolve_accelerator(device: str) -> str:
    if device == "auto":
        return "auto"
    return "gpu" if device == "cuda" else "cpu"


def _metric_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        return {}
    row = metrics[0]
    aliases = {
        "image_auroc": ("image_AUROC", "image_auroc"),
        "image_f1": ("image_F1Score", "image_F1", "image_f1"),
        "pixel_auroc": ("pixel_AUROC", "pixel_auroc"),
        "au_pro": ("AU_PRO", "au_pro", "pixel_AUPRO"),
        "pixel_f1": ("pixel_F1Score", "pixel_F1", "pixel_f1"),
    }
    summary = {}
    for field, names in aliases.items():
        for name in names:
            if name in row:
                summary[field] = row[name]
                break
    return summary


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _blocked_report(
    dataset_root: Path,
    dataset: str,
    category: str,
    method: str,
    category_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "method": f"anomalib_{method}",
        "baseline_kind": "anomalib",
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
        "proof_note": "No Anomalib metrics were produced because the run was blocked.",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
