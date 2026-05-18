"""Evaluate a trained Phase 1 reconstruction autoencoder on a MVTec-style test split.

Walks ``--test-root`` (a category root, e.g. ``~/datasets/mvtec_ad/bottle/test``)
and labels each file by its parent folder: ``good``/``normal`` -> 0, else 1.
Writes a per-image score JSON in the same schema as
``scripts/score_anomalib_test.py`` plus an aggregate metrics block.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from inspectnet_cx.training.phase1_recon import load_recon, score_image_paths

NORMAL_NAMES = {"good", "normal"}


def _label_for(path: Path, test_root: Path) -> int:
    rel = path.relative_to(test_root)
    return 0 if rel.parts[0] in NORMAL_NAMES else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--category", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    image_paths = sorted(
        [p for p in Path(args.test_root).rglob("*.png") if p.is_file()]
    )
    if not image_paths:
        raise SystemExit(f"No PNGs under {args.test_root}")

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    model, image_size = load_recon(args.checkpoint, device=device)

    t0 = time.perf_counter()
    scores = score_image_paths(
        model, image_paths, image_size=image_size, device=device, batch_size=args.batch_size
    )
    elapsed = time.perf_counter() - t0

    labels = np.asarray(
        [_label_for(p, Path(args.test_root)) for p in image_paths], dtype=np.int64
    )

    items = [
        {
            "path": str(p.relative_to(Path(args.test_root).parent)),
            "label": int(lab),
            "score": float(sc),
        }
        for p, lab, sc in zip(image_paths, labels, scores, strict=True)
    ]

    payload = {
        "schema": "inspectnet_cx.scores.v1",
        "dataset": args.dataset,
        "category": args.category,
        "method": "phase1_recon_ae",
        "checkpoint": str(args.checkpoint),
        "image_size": int(image_size),
        "device": device,
        "n_images": len(image_paths),
        "elapsed_seconds": float(elapsed),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(
        f"wrote {len(items)} scores ({n_pos} anomaly, {n_neg} normal) to {args.output}"
        f" in {elapsed:.2f}s on {device}"
    )


if __name__ == "__main__":
    main()
