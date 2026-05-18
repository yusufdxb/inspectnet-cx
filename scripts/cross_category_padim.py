"""PaDiM cross-category transfer: fit on train_cat, score test_cat.

Trains a PaDiM memory bank on one MVTec AD category's train set, then
scores another category's test split using the SAME fitted bank. Writes
a per-image score JSON in the same schema as
``scripts/score_anomalib_test.py``.

The implementation uses two MVTecAD datamodules: ``train_dm`` (for fit)
and ``test_dm`` (for predict). Anomalib's ``Engine.predict`` uses the
``test_dataloader`` of the supplied datamodule, so swapping the
datamodule between fit and predict cleanly redirects the held-out
distribution while keeping the model parameters untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NORMAL_NAMES = {"good", "normal"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--dataset", default="mvtec_ad")
    parser.add_argument("--train-category", required=True)
    parser.add_argument("--test-category", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts/cross_padim"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim

    dataset_root = args.dataset_root.expanduser() / args.dataset

    train_dm = MVTecAD(
        root=dataset_root,
        category=args.train_category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
    )
    test_dm = MVTecAD(
        root=dataset_root,
        category=args.test_category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
    )

    model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    engine = Engine(
        accelerator="cpu",
        devices=1,
        default_root_dir=args.work_dir,
        logger=False,
        max_epochs=1,
    )
    engine.fit(model=model, datamodule=train_dm)

    test_category_root = dataset_root / args.test_category
    predictions = engine.predict(model=model, datamodule=test_dm)

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
            try:
                rel = absolute.relative_to(test_category_root / "test")
                label = 0 if rel.parts[0] in NORMAL_NAMES else 1
            except ValueError:
                label = -1
            try:
                rel_path = absolute.relative_to(test_category_root)
            except ValueError:
                rel_path = Path(path_str)
            items.append(
                {"path": str(rel_path), "label": int(label), "score": float(score)}
            )

    payload = {
        "schema": "inspectnet_cx.scores.v1",
        "dataset": args.dataset,
        "category": args.test_category,
        "train_category": args.train_category,
        "method": "anomalib_padim_cross_category",
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    n_pos = sum(1 for i in items if i["label"] == 1)
    n_neg = sum(1 for i in items if i["label"] == 0)
    print(
        f"train={args.train_category} test={args.test_category}: "
        f"wrote {len(items)} items ({n_pos} anomaly, {n_neg} normal) to {args.output}"
    )


if __name__ == "__main__":
    main()
