"""Batch inference: folder of images -> CSV row per image.

Each row contains the image path (relative to the input root if requested),
the model image score, the threshold used to decide, and the decision
(``anomaly`` or ``normal``).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor
from inspectnet_cx.data.dataset_check import COMMON_IMAGE_EXTENSIONS


def iter_images(root: Path, *, recursive: bool = True) -> Iterator[Path]:
    """Yield image files under ``root`` in sorted order."""
    if root.is_file():
        if root.suffix.lower() in COMMON_IMAGE_EXTENSIONS:
            yield root
        return
    walker = root.rglob("*") if recursive else root.iterdir()
    for candidate in sorted(walker):
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS:
            yield candidate


def score_images(
    model_path: Path,
    image_paths: Iterable[Path],
    *,
    batch_size: int = 8,
    threshold: float | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Score a list of images and return one record per image."""
    processor = InspectNetCXProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = InspectNetCXForAnomalyDetection.from_pretrained(model_path, trust_remote_code=True)
    model.to(_resolve_device(device))
    model.eval()

    paths = list(image_paths)
    results: list[dict[str, Any]] = []

    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=pil_images, return_tensors="pt")
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(model.device)
            output = model(**inputs, threshold=threshold)
            scores = _tensor_to_floats(output.image_score)
            used_threshold = _tensor_to_floats(output.threshold)
            for i, path in enumerate(batch_paths):
                score = scores[i] if i < len(scores) else float("nan")
                used = used_threshold[i] if i < len(used_threshold) else used_threshold[0]
                cutoff = used if used is not None else 0.5
                results.append(
                    {
                        "path": str(path),
                        "score": float(score),
                        "threshold": float(used) if used is not None else None,
                        "decision": "anomaly" if score >= cutoff else "normal",
                    }
                )
    return results


def write_csv(
    records: list[dict[str, Any]],
    output: Path,
    *,
    root: Path | None = None,
) -> None:
    """Write per-image records to a CSV at ``output``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["path", "score", "threshold", "decision"])
        for record in records:
            path = Path(record["path"])
            if root is not None:
                with contextlib.suppress(ValueError):
                    path = path.relative_to(root)
            writer.writerow(
                [
                    str(path),
                    f"{record['score']:.6f}",
                    f"{record['threshold']:.6f}" if record["threshold"] is not None else "",
                    record["decision"],
                ]
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run InspectNet-CX inference on every image under a folder.",
    )
    parser.add_argument("--model", required=True, type=Path, help="Path to a saved model dir.")
    parser.add_argument("--input", required=True, type=Path, help="Folder (or file) of images.")
    parser.add_argument("--output", required=True, type=Path, help="CSV output path.")
    parser.add_argument("--threshold", type=float, help="Optional threshold override.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only consider direct children of --input, not subdirectories.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    image_paths = list(iter_images(args.input, recursive=not args.no_recursive))
    if not image_paths:
        msg = f"no images found under {args.input}"
        raise SystemExit(msg)
    records = score_images(
        model_path=args.model,
        image_paths=image_paths,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
    )
    root = args.input if args.input.is_dir() else None
    write_csv(records, args.output, root=root)
    n_anomaly = sum(1 for r in records if r["decision"] == "anomaly")
    n_normal = len(records) - n_anomaly
    print(
        f"Wrote {len(records)} rows to {args.output} "
        f"({n_anomaly} anomaly, {n_normal} normal)."
    )


def _tensor_to_floats(value: torch.Tensor | None) -> list[float]:
    if value is None:
        return []
    return [float(x) for x in value.detach().cpu().reshape(-1).tolist()]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA requested but torch.cuda.is_available() is false."
        raise RuntimeError(msg)
    return torch.device(device)


if __name__ == "__main__":
    main()
