from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from inspectnet_cx import InspectNetCXConfig, InspectNetCXForAnomalyDetection

DEFAULT_ONNX_OUTPUT = Path("artifacts/inspectnet-cx-phase0/model.onnx")
DEFAULT_OPENVINO_OUTPUT = Path("artifacts/inspectnet-cx-phase0/openvino/model.xml")


class _OnnxExportWrapper(nn.Module):
    def __init__(self, model: InspectNetCXForAnomalyDetection) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # The public model output includes Python defect-region postprocessing, which is
        # intentionally not part of the tensor export graph.
        features = self.model.encoder(pixel_values)
        heatmap = torch.sigmoid(self.model.anomaly_head(features))
        heatmap = F.interpolate(
            heatmap,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        image_score = heatmap.flatten(1).amax(dim=1)
        threshold = torch.full_like(image_score, float(self.model.config.threshold))
        binary_mask = (heatmap >= threshold.view(-1, 1, 1, 1)).to(pixel_values.dtype)
        confidence = (image_score - threshold).sigmoid()
        return (
            image_score,
            heatmap,
            binary_mask,
            threshold,
            confidence,
        )


def check_export_readiness(
    model_dir: Path | None = None,
    export_format: str = "onnx",
    onnx_path: Path | None = None,
) -> dict[str, Any]:
    packages = {
        "torch": _package_status("torch"),
        "onnx": _package_status("onnx"),
        "onnxruntime": _package_status("onnxruntime"),
        "onnxscript": _package_status("onnxscript"),
        "openvino": _package_status("openvino"),
    }
    blockers = []
    if export_format == "onnx" and not packages["onnx"]["installed"]:
        blockers.append("onnx is not installed; install with pip install -e '.[export]'")
    if export_format == "onnx" and not packages["onnxscript"]["installed"]:
        blockers.append("onnxscript is not installed; install with pip install -e '.[export]'")
    if export_format == "openvino" and not packages["openvino"]["installed"]:
        blockers.append("openvino is not installed; install openvino before IR conversion")
    if export_format == "openvino" and onnx_path is not None and not onnx_path.exists():
        blockers.append(f"source ONNX file is missing: {onnx_path}")
    if model_dir is not None:
        model_dir = model_dir.expanduser()
        if not (model_dir / "config.json").exists():
            blockers.append(f"model config not found under {model_dir}")

    return {
        "status": "ready" if not blockers else "blocked",
        "export_format": export_format,
        "model_dir": str(model_dir.expanduser()) if model_dir is not None else None,
        "onnx_path": str(onnx_path.expanduser()) if onnx_path is not None else None,
        "packages": packages,
        "blocked_reasons": blockers,
        "proof_note": (
            "Export readiness only proves dependency and file availability. It does not prove "
            "runtime accuracy, OpenVINO parity, TensorRT compatibility, or workstation latency."
        ),
    }


def export_phase0_onnx(
    model_dir: Path | None = None,
    output: Path = DEFAULT_ONNX_OUTPUT,
    image_size: int | None = None,
    opset: int = 18,
    verify: bool = False,
) -> dict[str, Any]:
    readiness = check_export_readiness(model_dir=model_dir, export_format="onnx")
    if readiness["blocked_reasons"]:
        raise RuntimeError("; ".join(readiness["blocked_reasons"]))

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_dir=model_dir, image_size=image_size)
    model.eval()
    resolved_image_size = int(image_size or model.config.image_size)
    dummy = torch.randn(1, 3, resolved_image_size, resolved_image_size)
    wrapper = _OnnxExportWrapper(model).eval()

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            output,
            input_names=["pixel_values"],
            output_names=[
                "image_score",
                "anomaly_heatmap",
                "binary_mask",
                "threshold",
                "confidence",
            ],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "image_score": {0: "batch"},
                "anomaly_heatmap": {0: "batch"},
                "binary_mask": {0: "batch"},
                "threshold": {0: "batch"},
                "confidence": {0: "batch"},
            },
            opset_version=opset,
        )

    result = {
        "status": "exported_phase0_onnx",
        "format": "onnx",
        "path": str(output),
        "image_size": resolved_image_size,
        "opset": opset,
        "verified_with_onnxruntime": False,
        "proof_note": (
            "This is an export of the Phase 0 placeholder model, not a trained production "
            "detector and not a deployment-ready runtime artifact."
        ),
    }
    if verify:
        result["onnxruntime_check"] = _verify_onnx(output, wrapper, dummy)
        result["verified_with_onnxruntime"] = result["onnxruntime_check"]["status"] == "passed"
    return result


def export_openvino_from_onnx(
    onnx_path: Path,
    output: Path = DEFAULT_OPENVINO_OUTPUT,
) -> dict[str, Any]:
    readiness = check_export_readiness(
        export_format="openvino",
        onnx_path=onnx_path,
    )
    if readiness["blocked_reasons"]:
        raise RuntimeError("; ".join(readiness["blocked_reasons"]))

    import openvino as ov  # type: ignore[import-not-found]

    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    ov_model = ov.convert_model(str(onnx_path.expanduser()))
    ov.save_model(ov_model, str(output))
    return {
        "status": "exported_phase0_openvino",
        "format": "openvino_ir",
        "path": str(output),
        "source_onnx": str(onnx_path),
        "proof_note": (
            "OpenVINO conversion succeeded for the Phase 0 placeholder model only. Run parity "
            "and latency on target hardware before making deployment claims."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export or check Phase 0 InspectNet-CX artifacts.")
    parser.add_argument("--model", type=Path, help="Optional save_pretrained model directory.")
    parser.add_argument("--format", choices=("onnx", "openvino"), default="onnx")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-onnx", type=Path, help="Required for OpenVINO conversion.")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.check_only:
        report = check_export_readiness(
            model_dir=args.model,
            export_format=args.format,
            onnx_path=args.source_onnx,
        )
    elif args.format == "onnx":
        report = export_phase0_onnx(
            model_dir=args.model,
            output=args.output or DEFAULT_ONNX_OUTPUT,
            image_size=args.image_size,
            opset=args.opset,
            verify=args.verify,
        )
    else:
        if args.source_onnx is None:
            msg = "--source-onnx is required for OpenVINO conversion"
            raise SystemExit(msg)
        report = export_openvino_from_onnx(
            onnx_path=args.source_onnx,
            output=args.output or DEFAULT_OPENVINO_OUTPUT,
        )
    print(json.dumps(report, indent=2))


def _load_model(
    model_dir: Path | None,
    image_size: int | None,
) -> InspectNetCXForAnomalyDetection:
    if model_dir is None:
        return InspectNetCXForAnomalyDetection(InspectNetCXConfig(image_size=image_size or 224))
    return InspectNetCXForAnomalyDetection.from_pretrained(model_dir.expanduser())


def _verify_onnx(
    output: Path,
    wrapper: _OnnxExportWrapper,
    dummy: torch.Tensor,
) -> dict[str, Any]:
    if importlib.util.find_spec("onnxruntime") is None:
        return {
            "status": "blocked",
            "reason": "onnxruntime is not installed; install with pip install -e '.[export]'",
        }
    import numpy as np
    import onnxruntime as ort  # type: ignore[import-not-found]

    with torch.inference_mode():
        torch_outputs = wrapper(dummy)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"pixel_values": dummy.numpy()})
    max_abs_diffs = [
        float(np.max(np.abs(torch_output.detach().numpy() - ort_output)))
        for torch_output, ort_output in zip(torch_outputs, ort_outputs, strict=True)
    ]
    torch_heatmap = torch_outputs[1].detach().numpy()
    torch_mask = torch_outputs[2].detach().numpy()
    torch_threshold = torch_outputs[3].detach().numpy().reshape(-1, 1, 1, 1)
    ort_mask = ort_outputs[2]
    mask_mismatches = torch_mask != ort_mask
    ambiguous_mask_mismatches = 0
    if mask_mismatches.any():
        margins = np.abs(torch_heatmap - torch_threshold)
        ambiguous_mask_mismatches = int((margins[mask_mismatches] <= 1e-4).sum())
    continuous_diffs = [diff for index, diff in enumerate(max_abs_diffs) if index != 2]
    mask_mismatch_count = int(mask_mismatches.sum())
    mask_status = mask_mismatch_count == ambiguous_mask_mismatches
    return {
        "status": "passed" if max(continuous_diffs) <= 1e-4 and mask_status else "failed",
        "max_abs_diffs": max_abs_diffs,
        "output_names": [
            "image_score",
            "anomaly_heatmap",
            "binary_mask",
            "threshold",
            "confidence",
        ],
        "binary_mask_mismatch_count": mask_mismatch_count,
        "binary_mask_ambiguous_mismatch_count": ambiguous_mask_mismatches,
        "binary_mask_match_ratio": float(1.0 - (mask_mismatch_count / torch_mask.size)),
        "tolerance": 1e-4,
    }


def _package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False, "version": None}
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"installed": True, "version": version}


if __name__ == "__main__":
    main()
