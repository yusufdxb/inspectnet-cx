"""Score the MVTec AD test split with a real Anomalib model and dump per-image scores.

This is a thin wrapper over ``anomalib.engine.Engine.predict`` that mirrors the
``run_anomalib_baseline`` configuration. It exists so that
``scripts/threshold_analysis.py`` can consume a portable per-image score file
without depending on Lightning internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectnet_cx.data.dataset_check import COMMON_IMAGE_EXTENSIONS

NORMAL_NAMES = {"good", "normal"}


def _iter_images(path: Path) -> list[Path]:
    return [
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file() and candidate.suffix.lower() in COMMON_IMAGE_EXTENSIONS
    ]


def _label_for(path: Path, category_root: Path) -> int:
    rel = path.relative_to(category_root / "test")
    return 0 if rel.parts[0] in NORMAL_NAMES else 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--category", required=True)
    parser.add_argument("--method", default="padim", choices=("padim", "patchcore"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/anomalib_scoring"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim, Patchcore

    dataset_root = args.dataset_root.expanduser()
    category_root = dataset_root / args.dataset / args.category

    datamodule = MVTecAD(
        root=dataset_root / args.dataset,
        category=args.category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
    )
    model = (
        Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
        if args.method == "padim"
        else Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"))
    )
    accelerator = "gpu" if args.device == "cuda" else "cpu" if args.device == "cpu" else "auto"
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        default_root_dir=args.work_dir,
        logger=False,
        max_epochs=1,
    )
    engine.fit(model=model, datamodule=datamodule)

    test_images = _iter_images(category_root / "test")
    label_by_path = {p: _label_for(p, category_root) for p in test_images}

    predictions = engine.predict(model=model, datamodule=datamodule)
    items: list[dict[str, object]] = []
    for batch in predictions or []:
        paths = batch.image_path if hasattr(batch, "image_path") else batch.get("image_path")
        scores = batch.pred_score if hasattr(batch, "pred_score") else batch.get("pred_score")
        if paths is None or scores is None:
            continue
        scores_flat = scores.detach().cpu().reshape(-1).tolist()
        if isinstance(paths, str):
            paths = [paths]
        for path_str, score in zip(paths, scores_flat, strict=False):
            absolute = Path(path_str).resolve()
            label = label_by_path.get(absolute)
            if label is None:
                try:
                    rel = absolute.relative_to(category_root / "test")
                    label = 0 if rel.parts[0] in NORMAL_NAMES else 1
                except ValueError:
                    label = -1
            try:
                rel_path = absolute.relative_to(category_root)
            except ValueError:
                rel_path = Path(path_str)
            items.append(
                {
                    "path": str(rel_path),
                    "label": int(label),
                    "score": float(score),
                }
            )

    payload = {
        "schema": "inspectnet_cx.scores.v1",
        "dataset": args.dataset,
        "category": args.category,
        "method": f"anomalib_{args.method}",
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    n_pos = sum(1 for i in items if i["label"] == 1)
    n_neg = sum(1 for i in items if i["label"] == 0)
    print(f"wrote {len(items)} scored items ({n_pos} anomaly, {n_neg} normal) to {args.output}")


if __name__ == "__main__":
    main()
