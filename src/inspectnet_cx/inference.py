from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor


def output_to_jsonable(output: Any) -> dict[str, Any]:
    return {
        "image_score": _tensor_to_list(output.image_score),
        "threshold": _tensor_to_list(output.threshold),
        "confidence": _tensor_to_list(output.confidence),
        "defect_regions": output.defect_regions,
        "heatmap_shape": list(output.anomaly_heatmap.shape),
        "mask_shape": list(output.binary_mask.shape),
    }


def run_inference(
    model_path: Path,
    image_path: Path,
    output_path: Path | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    processor = InspectNetCXProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = InspectNetCXForAnomalyDetection.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs, threshold=threshold)

    payload = output_to_jsonable(output)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 0 InspectNet-CX inference on one image."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to a saved model directory.",
    )
    parser.add_argument("--image", required=True, type=Path, help="Path to an input image.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument("--threshold", type=float, help="Optional threshold override.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_inference(
        model_path=args.model,
        image_path=args.image,
        output_path=args.output,
        threshold=args.threshold,
    )
    if args.output is None:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Wrote inference result to {args.output}")


def _tensor_to_list(value: torch.Tensor | None) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in value.detach().cpu().flatten().tolist()]


if __name__ == "__main__":
    main()
