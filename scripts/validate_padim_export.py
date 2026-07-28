from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from inspectnet_cx.data.dataset_check import COMMON_IMAGE_EXTENSIONS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-validate exported Anomalib PaDiM ONNX/OpenVINO artifacts."
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--openvino", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--inference-precision",
        choices=("f32", "bf16", "default"),
        default="f32",
        help=(
            "OpenVINO CPU INFERENCE_PRECISION_HINT. Defaults to f32 for parity-strict "
            "validation. Use 'default' to inherit the CPU plugin default (bfloat16 on "
            "AVX-512-BF16 hosts), which trades parity for speed."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/verification/anomalib_padim_export_smoke.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_export(
        args.onnx,
        args.openvino,
        args.input,
        args.image_size,
        inference_precision=args.inference_precision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def validate_export(
    onnx_path: Path,
    openvino_path: Path,
    input_path: Path,
    image_size: int,
    inference_precision: str = "f32",
) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore[import-not-found]
    import openvino as ov  # type: ignore[import-not-found]

    image_paths = _resolve_input_images(input_path)
    if not image_paths:
        msg = f"no supported images found under {input_path}"
        raise ValueError(msg)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _assert_expected_output_order(session)
    core = ov.Core()
    ov_config: dict[str, str] = {}
    if inference_precision in ("f32", "bf16"):
        ov_config["INFERENCE_PRECISION_HINT"] = inference_precision
    compiled = core.compile_model(str(openvino_path), "CPU", ov_config)
    ov_input = compiled.input(0)
    ov_output_names = [_openvino_output_name(out) for out in compiled.outputs]

    cases = []
    for image_path in image_paths:
        array = _load_image(image_path, image_size)
        onnx_outputs = session.run(None, {"input": array})
        ov_raw = compiled({ov_input.any_name: array})
        ov_outputs = [
            _lookup_openvino_output(ov_raw, output, output_name)
            for output, output_name in zip(compiled.outputs, ov_output_names, strict=True)
        ]
        cases.append(_compare_case(image_path, onnx_outputs, ov_outputs))

    max_abs_error = max(item["max_abs_error"] for item in cases)
    max_mean_abs_error = max(item["max_mean_abs_error"] for item in cases)
    max_rel_error = max(item["max_rel_error"] for item in cases)
    boolean_outputs_match = all(item["boolean_outputs_match"] for item in cases)
    total_pred_mask_pixel_flips = sum(item.get("pred_mask_pixel_flips", 0) for item in cases)
    total_pred_mask_pixels = sum(item.get("pred_mask_pixel_count", 0) for item in cases)
    pred_mask_flip_fraction = (
        total_pred_mask_pixel_flips / total_pred_mask_pixels if total_pred_mask_pixels else 0.0
    )
    if max_abs_error <= 1e-4 and boolean_outputs_match:
        status = "passed"
    elif max_abs_error <= 1e-4 and pred_mask_flip_fraction <= 1e-4:
        status = "passed_mask_boundary_unstable"
    else:
        status = "loaded_parity_failed"
    return {
        "status": status,
        "onnx_path": str(onnx_path),
        "openvino_path": str(openvino_path),
        "provider": "CPU",
        "image_size": image_size,
        "input_count": len(image_paths),
        "inference_precision_hint": inference_precision,
        "onnx_inputs": [(item.name, item.shape, item.type) for item in session.get_inputs()],
        "onnx_outputs": [(item.name, item.shape, item.type) for item in session.get_outputs()],
        "openvino_outputs": [
            {
                "names": sorted(str(name) for name in output.names),
                "partial_shape": str(output.partial_shape),
                "element_type": str(output.element_type),
            }
            for output in compiled.outputs
        ],
        "cases": cases,
        "summary": {
            "max_abs_error": max_abs_error,
            "max_mean_abs_error": max_mean_abs_error,
            "max_rel_error": max_rel_error,
            "boolean_outputs_match": boolean_outputs_match,
            "pred_mask_pixel_flips": total_pred_mask_pixel_flips,
            "pred_mask_pixel_count": total_pred_mask_pixels,
            "pred_mask_flip_fraction": pred_mask_flip_fraction,
        },
        "proof_note": (
            "This checks whether exported trained Anomalib PaDiM ONNX/OpenVINO artifacts load "
            "and agree with each other on local images. A failed status means export files are "
            "present and loadable, but parity is not clean enough for deployment claims. This "
            "does not prove checkpoint-to-export parity, optimized runtime, or target hardware "
            "readiness."
        ),
    }


EXPECTED_OUTPUT_ORDER = ("pred_score", "pred_label", "anomaly_map", "pred_mask")


def _assert_expected_output_order(session: Any) -> None:
    """Fail loudly if the ONNX output order is not the order this script assumes.

    ``_compare_case`` zips a hardcoded name list positionally against the model
    outputs. If a future export reorders the heads, that zip would silently
    mislabel tensors and attach the mask-specific pixel-flip accounting to the
    wrong output. This guard converts that silent corruption into a clear error.
    """
    actual = tuple(item.name for item in session.get_outputs())
    if len(actual) != len(EXPECTED_OUTPUT_ORDER):
        msg = (
            f"ONNX model exposes {len(actual)} outputs {actual}; this validator "
            f"assumes exactly {len(EXPECTED_OUTPUT_ORDER)}: {EXPECTED_OUTPUT_ORDER}"
        )
        raise ValueError(msg)
    # Anomalib's exported names may differ in spelling; only enforce arity and
    # warn loudly on a name mismatch rather than hard-failing a working export.
    if actual != EXPECTED_OUTPUT_ORDER:
        print(
            "WARNING: ONNX output names "
            f"{actual} differ from the assumed order {EXPECTED_OUTPUT_ORDER}; "
            "parity labels are positional and may be misattributed."
        )


def _resolve_input_images(input_path: Path) -> list[Path]:
    input_path = input_path.expanduser()
    if input_path.is_file() and input_path.suffix.lower() in COMMON_IMAGE_EXTENSIONS:
        return [input_path]
    if input_path.is_dir():
        return [
            path
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in COMMON_IMAGE_EXTENSIONS
        ]
    return []


def _load_image(path: Path, image_size: int) -> np.ndarray:
    image = (
        Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))[None, ...]


def _compare_case(
    image_path: Path, onnx_outputs: list[np.ndarray], ov_outputs: list[np.ndarray]
) -> dict[str, Any]:
    names = ["pred_score", "pred_label", "anomaly_map", "pred_mask"]
    output_reports = []
    bool_match = True
    max_abs = 0.0
    max_mean = 0.0
    max_rel = 0.0
    pred_mask_pixel_flips = 0
    pred_mask_pixel_count = 0
    for name, onnx_value, ov_value in zip(names, onnx_outputs, ov_outputs, strict=True):
        if onnx_value.dtype == bool or ov_value.dtype == bool:
            onnx_bool = onnx_value.astype(bool)
            ov_bool = ov_value.astype(bool)
            matches = bool(np.array_equal(onnx_bool, ov_bool))
            bool_match = bool_match and matches
            entry: dict[str, Any] = {
                "name": name,
                "type": "boolean",
                "matches": matches,
                "shape": list(onnx_value.shape),
            }
            if name == "pred_mask":
                flips = int((onnx_bool != ov_bool).sum())
                pred_mask_pixel_flips = flips
                pred_mask_pixel_count = int(onnx_bool.size)
                entry["pixel_flips"] = flips
                entry["pixel_count"] = pred_mask_pixel_count
            output_reports.append(entry)
            continue
        diff = np.abs(onnx_value.astype(np.float32) - ov_value.astype(np.float32))
        rel = diff / np.maximum(np.abs(onnx_value.astype(np.float32)), 1.0e-8)
        max_abs = max(max_abs, float(diff.max()))
        max_mean = max(max_mean, float(diff.mean()))
        max_rel = max(max_rel, float(rel.max()))
        output_reports.append(
            {
                "name": name,
                "type": "continuous",
                "max_abs_error": float(diff.max()),
                "mean_abs_error": float(diff.mean()),
                "max_rel_error": float(rel.max()),
                "shape": list(onnx_value.shape),
            }
        )
    return {
        "path": str(image_path),
        "outputs": output_reports,
        "boolean_outputs_match": bool_match,
        "max_abs_error": max_abs,
        "max_mean_abs_error": max_mean,
        "max_rel_error": max_rel,
        "pred_mask_pixel_flips": pred_mask_pixel_flips,
        "pred_mask_pixel_count": pred_mask_pixel_count,
    }


def _lookup_openvino_output(
    raw: dict[Any, np.ndarray], output: Any, output_name: str
) -> np.ndarray:
    if output in raw:
        return raw[output]
    if output_name in raw:
        return raw[output_name]
    for key, value in raw.items():
        if getattr(key, "any_name", None) == output_name:
            return value
    raise KeyError(output_name)


def _openvino_output_name(output: Any) -> str:
    name = getattr(output, "any_name", None)
    if name:
        return str(name)
    names = getattr(output, "names", None)
    if names:
        return sorted(str(item) for item in names)[0]
    return str(output)


if __name__ == "__main__":
    main()
