from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Investigate PaDiM checkpoint reuse/export.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/agent_b/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("~/datasets").expanduser())
    parser.add_argument("--dataset", default="mvtec_ad", choices=("mvtec_ad",))
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--sample-image", type=Path)
    parser.add_argument(
        "--export-root", type=Path, default=Path("artifacts/agent_b/anomalib_padim_export")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/agent_b/anomalib_padim_export_status.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = investigate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def investigate(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.expanduser()
    sample_image = args.sample_image or (
        args.dataset_root.expanduser()
        / args.dataset
        / args.category
        / "test"
        / "broken_large"
        / "000.png"
    )
    attempts: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "started",
        "dataset": args.dataset,
        "category": args.category,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "sample_image": str(sample_image),
        "attempted_commands": [
            (
                "Engine.predict(model=Padim(...), ckpt_path=<checkpoint>, "
                "data_path=<sample_image>, return_predictions=True)"
            ),
            "Engine.export(model=Padim(...), export_type=ExportType.ONNX, ckpt_path=<checkpoint>)",
            (
                "Engine.export(model=Padim(...), export_type=ExportType.OPENVINO, "
                "ckpt_path=<checkpoint>)"
            ),
        ],
        "attempts": attempts,
        "is_reusable_for_prediction": False,
        "exported_artifacts": [],
        "is_export_real": False,
        "next_precise_action": None,
    }
    if not checkpoint.exists():
        report.update(
            {
                "status": "blocked",
                "blocked_reason": "checkpoint is missing",
                "next_precise_action": (
                    "Run make baseline-anomalib-padim to regenerate the checkpoint."
                ),
            }
        )
        return report

    try:
        from anomalib.deploy import ExportType  # type: ignore[import-not-found]
        from anomalib.engine import Engine  # type: ignore[import-not-found]
        from anomalib.models import Padim  # type: ignore[import-not-found]
    except Exception as exc:
        report.update(
            {
                "status": "blocked",
                "blocked_reason": f"Anomalib import failed: {type(exc).__name__}: {exc}",
                "next_precise_action": (
                    "Install the pinned optional stack from requirements/agent_b_verified.txt."
                ),
            }
        )
        return report

    model = Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    engine = Engine(accelerator="cpu", devices=1, logger=False)

    predict_attempt: dict[str, Any] = {"name": "checkpoint_predict", "success": False}
    try:
        predictions = engine.predict(
            model=model,
            ckpt_path=checkpoint,
            data_path=sample_image,
            return_predictions=True,
        )
        predict_attempt.update(
            {
                "success": bool(predictions),
                "prediction_count": len(predictions or []),
                "output_type": type(predictions[0]).__name__ if predictions else None,
            }
        )
        report["is_reusable_for_prediction"] = bool(predictions)
    except Exception as exc:
        predict_attempt.update(_exception_payload(exc))
    attempts.append(predict_attempt)

    export_root = args.export_root.expanduser()
    for export_type in (ExportType.ONNX, ExportType.OPENVINO):
        export_attempt: dict[str, Any] = {
            "name": f"export_{export_type.value}",
            "success": False,
            "export_type": export_type.value,
            "export_root": str(export_root),
        }
        try:
            exported = engine.export(
                model=model,
                export_type=export_type,
                export_root=export_root,
                ckpt_path=checkpoint,
                input_size=(256, 256),
            )
            exported_path = Path(exported) if exported is not None else None
            export_attempt.update(
                {
                    "success": exported_path is not None and exported_path.exists(),
                    "path": str(exported_path) if exported_path is not None else None,
                    "path_exists": exported_path.exists() if exported_path is not None else False,
                }
            )
            if export_attempt["success"]:
                report["exported_artifacts"].append(export_attempt["path"])
        except Exception as exc:
            export_attempt.update(_exception_payload(exc))
        attempts.append(export_attempt)

    report["is_export_real"] = bool(report["exported_artifacts"])
    if report["is_export_real"]:
        report["status"] = "export_succeeded"
        report["next_precise_action"] = (
            "Validate exported PaDiM inference parity against the Lightning checkpoint before "
            "making deployment claims."
        )
    elif report["is_reusable_for_prediction"]:
        report["status"] = "prediction_reuse_only"
        report["next_precise_action"] = (
            "Use the Lightning checkpoint for demo inference; investigate Anomalib exporter errors "
            "before publishing ONNX/OpenVINO claims."
        )
    else:
        report["status"] = "blocked"
        report["next_precise_action"] = (
            "Regenerate PaDiM with Anomalib and inspect checkpoint compatibility."
        )
    return report


def _exception_payload(exc: Exception) -> dict[str, Any]:
    return {
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-8:],
    }


if __name__ == "__main__":
    main()
