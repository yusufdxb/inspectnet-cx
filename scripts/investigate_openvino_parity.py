from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Phase 0 ONNX Runtime and OpenVINO outputs."
    )
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--openvino", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/agent_b/openvino_parity_investigation.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = investigate_parity(args.onnx, args.openvino, image_size=args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def investigate_parity(
    onnx_path: Path, openvino_path: Path, image_size: int = 224
) -> dict[str, Any]:
    import onnxruntime as ort  # type: ignore[import-not-found]
    import openvino as ov  # type: ignore[import-not-found]

    onnx_path = onnx_path.expanduser()
    openvino_path = openvino_path.expanduser()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    core = ov.Core()
    compiled = core.compile_model(str(openvino_path), "CPU")
    ov_input = compiled.input(0)
    output_names = [_openvino_output_name(out) for out in compiled.outputs]

    cases = _input_cases(image_size)
    case_reports = []
    for name, array in cases:
        ort_outputs = session.run(None, {"pixel_values": array})
        ov_outputs_raw = compiled({ov_input.any_name: array})
        ov_outputs = [
            _lookup_openvino_output(ov_outputs_raw, output, output_name)
            for output, output_name in zip(compiled.outputs, output_names, strict=True)
        ]
        case_reports.append(_compare_case(name, ort_outputs, ov_outputs))

    continuous_pass_1e_4 = all(item["continuous_max_abs_error"] <= 1e-4 for item in case_reports)
    continuous_pass_1e_3 = all(item["continuous_max_abs_error"] <= 1e-3 for item in case_reports)
    exact_mask_pass = all(item["binary_mask_mismatch_count"] == 0 for item in case_reports)
    boundary_only_mask_mismatch = all(
        item["binary_mask_mismatch_count"] == item["binary_mask_ambiguous_mismatch_count"]
        for item in case_reports
    )
    if continuous_pass_1e_4 and exact_mask_pass:
        status = "passed_exact"
    elif continuous_pass_1e_4 and boundary_only_mask_mismatch:
        status = "continuous_passed_mask_boundary_unstable"
    elif continuous_pass_1e_3 and boundary_only_mask_mismatch:
        status = "minor_numeric_drift_mask_boundary_unstable"
    else:
        status = "failed"
    return {
        "status": status,
        "onnx_path": str(onnx_path),
        "openvino_path": str(openvino_path),
        "provider": "CPU",
        "image_size": image_size,
        "cases": case_reports,
        "summary": {
            "continuous_pass_1e_4": continuous_pass_1e_4,
            "continuous_pass_1e_3": continuous_pass_1e_3,
            "exact_binary_mask_parity_passed": exact_mask_pass,
            "boundary_only_mask_mismatch": boundary_only_mask_mismatch,
            "max_continuous_abs_error": max(
                item["continuous_max_abs_error"] for item in case_reports
            ),
            "max_continuous_mean_abs_error": max(
                item["continuous_mean_abs_error"] for item in case_reports
            ),
            "max_continuous_rel_error": max(
                item["continuous_max_rel_error"] for item in case_reports
            ),
        },
        "interpretation": (
            "Phase 0 OpenVINO differences are evaluated against ONNX Runtime. Binary mask "
            "differences are treated separately because the placeholder heatmap can lie close "
            "to the hard threshold."
        ),
        "proof_note": (
            "This investigates the Phase 0 placeholder export only. It does not validate a "
            "trained PaDiM or InspectNet-CX anomaly model export."
        ),
    }


def _input_cases(image_size: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    return [
        ("zeros", np.zeros((1, 3, image_size, image_size), dtype=np.float32)),
        ("ones", np.ones((1, 3, image_size, image_size), dtype=np.float32)),
        ("constant_half", np.full((1, 3, image_size, image_size), 0.5, dtype=np.float32)),
        ("seed0_uniform", rng.random((1, 3, image_size, image_size), dtype=np.float32)),
    ]


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


def _compare_case(
    name: str, ort_outputs: list[np.ndarray], ov_outputs: list[np.ndarray]
) -> dict[str, Any]:
    output_names = ["image_score", "anomaly_heatmap", "binary_mask", "threshold", "confidence"]
    per_output = []
    for output_name, onnx_value, ov_value in zip(
        output_names, ort_outputs, ov_outputs, strict=True
    ):
        diff = np.abs(onnx_value - ov_value)
        rel = diff / np.maximum(np.abs(onnx_value), 1.0e-8)
        per_output.append(
            {
                "name": output_name,
                "max_abs_error": float(diff.max()),
                "mean_abs_error": float(diff.mean()),
                "max_rel_error": float(rel.max()),
                "shape": list(onnx_value.shape),
            }
        )

    mask_mismatch = ort_outputs[2] != ov_outputs[2]
    heatmap = ort_outputs[1]
    threshold = ort_outputs[3].reshape(-1, 1, 1, 1)
    ambiguous = 0
    if mask_mismatch.any():
        ambiguous = int((np.abs(heatmap - threshold)[mask_mismatch] <= 1e-3).sum())
    continuous = [row for row in per_output if row["name"] != "binary_mask"]
    return {
        "case": name,
        "outputs": per_output,
        "continuous_max_abs_error": max(row["max_abs_error"] for row in continuous),
        "continuous_mean_abs_error": max(row["mean_abs_error"] for row in continuous),
        "continuous_max_rel_error": max(row["max_rel_error"] for row in continuous),
        "binary_mask_mismatch_count": int(mask_mismatch.sum()),
        "binary_mask_ambiguous_mismatch_count": ambiguous,
        "binary_mask_match_ratio": float(1.0 - (mask_mismatch.sum() / mask_mismatch.size)),
    }


if __name__ == "__main__":
    main()
