from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectnet_cx.prediction import DEFAULT_PADIM_CHECKPOINT, predict_images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anomaly prediction on local image files.")
    parser.add_argument("--input", type=Path, required=True, help="Image path or directory.")
    parser.add_argument("--dataset-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--dataset", default="mvtec_ad", choices=("mvtec_ad",))
    parser.add_argument("--category", default="bottle")
    parser.add_argument(
        "--backend",
        default="classical_patchdiff",
        choices=("classical_patchdiff", "anomalib_padim"),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PADIM_CHECKPOINT)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--threshold-quantile", type=float, default=0.995)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = predict_images(
        input_path=args.input,
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        category=args.category,
        backend=args.backend,
        output=args.output,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        threshold_quantile=args.threshold_quantile,
    )
    print(json.dumps(report, indent=2))


__all__ = ["main"]
