from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspectnet_cx.eval.result_schema import validate_result_payload

NON_BENCHMARK_REPORTS = {
    "dataset_check_rerun_mvtec_bottle.json",
    "dataset_check.json",
    "dataset_provenance_mvtec_ad_bottle.json",
    "environment_versions.json",
    "export_readiness_onnx.json",
    "export_readiness_openvino.json",
    "fixture_smoke_report.json",
    "inference.json",
    "inference_phase0_mvtec_good_000.json",
    "anomalib_padim_export_status.json",
    "anomalib_padim_export_smoke.json",
    "jetson_latency.json",
    "local_latency.json",
    "normal_threshold.json",
    "onnx_export_phase0.json",
    "openvino_export_phase0.json",
    "openvino_parity_investigation.json",
    "openvino_parity_phase0.json",
    "predictions_classical_examples.json",
    "predictions_padim_examples.json",
    "proof_readiness.json",
    "proof_readiness_after_agent_b.json",
    "proof_readiness_rerun.json",
}


def validate_results(input_dir: Path) -> dict[str, list[str]]:
    failures = {}
    paths = set(input_dir.glob("*.json"))
    paths.update(input_dir.glob("baseline_*/*/result.json"))
    for path in sorted(paths):
        payload = json.loads(path.read_text())
        if _is_non_benchmark_report(path, payload):
            continue
        errors = validate_result_payload(payload)
        if errors:
            failures[str(path)] = errors
    return failures


def _is_non_benchmark_report(path: Path, payload: dict[str, object]) -> bool:
    if path.name in NON_BENCHMARK_REPORTS:
        return True
    if path.name.startswith("prediction_"):
        return True
    if path.name.startswith(("jetson_latency", "local_latency")):
        return True
    if payload.get("status") == "completed_predictions" and "backend" in payload:
        return True
    if payload.get("status") == "classical_baseline_completed" and "fixture" in path.name:
        return True
    return payload.get("status") in {
        "blocked",
        "fixture_smoke_completed",
        "local_phase0_latency",
        "calibrated_phase0_threshold",
    } and "method" not in payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark result JSON schemas.")
    parser.add_argument("--input", type=Path, default=Path("reports"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    failures = validate_results(args.input)
    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit(1)
    print("All benchmark result JSON files are schema-valid.")


if __name__ == "__main__":
    main()
