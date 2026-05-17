from __future__ import annotations

import argparse
from pathlib import Path

from inspectnet_cx import InspectNetCXConfig, InspectNetCXForAnomalyDetection, InspectNetCXProcessor

MODEL_CARD = """---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: image-feature-extraction
tags:
  - anomaly-detection
  - industrial-inspection
  - phase-0
---

# InspectNet-CX Phase 0

This is a Phase 0 scaffold checkpoint. It is not trained for real anomaly detection.

It exists to test the API contract:

- image score
- anomaly heatmap
- binary mask
- threshold
- confidence
- defect regions

Do not use this checkpoint for production inspection or benchmark claims.
"""


def create_phase0_model(output: Path, image_size: int = 224) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    config = InspectNetCXConfig(image_size=image_size)
    processor = InspectNetCXProcessor(image_size=image_size)
    model = InspectNetCXForAnomalyDetection(config)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    (output / "README.md").write_text(MODEL_CARD)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local InspectNet-CX Phase 0 model.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/inspectnet-cx-phase0"))
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = create_phase0_model(args.output, image_size=args.image_size)
    print(f"Wrote Phase 0 model scaffold to {output}")


if __name__ == "__main__":
    main()
