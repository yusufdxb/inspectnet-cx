import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "README.md",
    "requirements-verified.txt",
    "requirements.txt",
    "assets/release_visual.svg",
    "artifact_index.json",
    "examples/prediction_padim_good_000.json",
    "examples/prediction_padim_broken_large_000.json",
    "examples/prediction_classical_good_000.json",
    "examples/prediction_classical_broken_large_000.json",
    "reports/anomalib_padim_mvtec_ad_bottle_result.json",
    "reports/classical_patchdiff_rerun_mvtec_ad_bottle_result.json",
    "reports/anomalib_padim_export_status.json",
    "reports/anomalib_padim_export_smoke.json",
    "reports/openvino_parity_investigation.json",
    "reports/dataset_provenance_mvtec_ad_bottle.json",
)

REQUIRED_README_PHRASES = (
    "not production factory-inspection software",
    "not a fully validated edge model",
    "MVTec AD is CC BY-NC-SA 4.0 and is not bundled here",
    "parity is not clean enough for deployment claims",
    "No TensorRT path has been validated",
    "No trained native InspectNet-CX model checkpoint exists yet",
    "Use Python 3.10 for reproduction",
    "CUDA is not required for the published PaDiM evidence",
    "Disk planning",
)

FORBIDDEN_PUBLIC_TEXT_PATTERNS = (
    "/home/yusuf",
    "HF_TOKEN",
    "BEGIN PRIVATE KEY",
)

FORBIDDEN_DATASET_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def iter_index_paths(index: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("prediction_examples", "visual_assets"):
        values = index.get(key, [])
        if isinstance(values, list):
            paths.extend(value for value in values if isinstance(value, str))

    for key in ("benchmark_reports", "export_reports", "parity_reports", "dataset_reports"):
        values = index.get(key, [])
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict) and isinstance(value.get("path"), str):
                    paths.append(value["path"])

    for key in ("dependency_file", "claims_ledger"):
        value = index.get(key)
        if isinstance(value, str):
            paths.append(value)

    return paths


def metric_string(value: float) -> str:
    return f"{value:.4f}"


def check_required_files(package_dir: Path, errors: list[str]) -> None:
    for relative in REQUIRED_FILES:
        if not (package_dir / relative).is_file():
            fail(errors, f"missing required HF package file: {relative}")


def check_json_files(package_dir: Path, errors: list[str]) -> None:
    for path in package_dir.rglob("*.json"):
        try:
            load_json(path)
        except json.JSONDecodeError as exc:
            fail(errors, f"invalid JSON: {path.relative_to(package_dir)}: {exc}")


def check_no_dataset_images(package_dir: Path, errors: list[str]) -> None:
    for path in package_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in FORBIDDEN_DATASET_EXTENSIONS:
            fail(errors, f"forbidden dataset-like image bundled: {path.relative_to(package_dir)}")


def check_no_local_leakage(package_dir: Path, errors: list[str]) -> None:
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in FORBIDDEN_PUBLIC_TEXT_PATTERNS:
            if pattern in text:
                relative = path.relative_to(package_dir)
                fail(
                    errors,
                    f"public package contains forbidden local/private text {pattern!r}: {relative}",
                )


def check_readme(package_dir: Path, errors: list[str]) -> None:
    readme = (package_dir / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())
    for phrase in REQUIRED_README_PHRASES:
        if phrase not in normalized_readme:
            fail(errors, f"README missing required limitation phrase: {phrase}")

    padim = load_json(package_dir / "reports/anomalib_padim_mvtec_ad_bottle_result.json")
    classical = load_json(
        package_dir / "reports/classical_patchdiff_rerun_mvtec_ad_bottle_result.json"
    )
    metric_checks = {
        "PaDiM image AUROC": metric_string(float(padim["image_auroc"])),
        "PaDiM image F1": metric_string(float(padim["image_f1"])),
        "PaDiM pixel AUROC": metric_string(float(padim["pixel_auroc"])),
        "PaDiM pixel F1": metric_string(float(padim["pixel_f1"])),
        "classical image AUROC": metric_string(float(classical["image_auroc"])),
        "classical image F1": metric_string(float(classical["image_f1"])),
        "classical pixel AUROC": metric_string(float(classical["pixel_auroc"])),
        "classical pixel F1": metric_string(float(classical["pixel_f1"])),
    }
    for label, value in metric_checks.items():
        if value not in readme:
            fail(errors, f"README missing {label} source metric {value}")


def check_artifact_index(package_dir: Path, errors: list[str]) -> None:
    index_path = package_dir / "artifact_index.json"
    index = load_json(index_path)
    if index.get("dataset_files_bundled") is not False:
        fail(errors, "artifact_index.json must set dataset_files_bundled to false")

    for relative in iter_index_paths(index):
        if not (package_dir / relative).exists():
            fail(errors, f"artifact_index.json references missing path: {relative}")


def expected_label_from_path(path: str) -> str:
    if re.search(r"/good/|_good_", path):
        return "normal"
    return "anomaly"


def check_prediction_examples(package_dir: Path, errors: list[str]) -> None:
    for path in sorted((package_dir / "examples").glob("prediction_*.json")):
        payload = load_json(path)
        for key in ("status", "backend", "dataset", "category", "input_count", "predictions"):
            if key not in payload:
                fail(errors, f"{path.relative_to(package_dir)} missing top-level key: {key}")
        if payload.get("backend") == "anomalib_padim" and "checkpoint" not in payload:
            fail(errors, f"{path.relative_to(package_dir)} missing checkpoint provenance")

        predictions = payload.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            fail(errors, f"{path.relative_to(package_dir)} has no predictions")
            continue

        prediction = predictions[0]
        for key in ("path", "expected_label", "predicted_label", "predicted_anomaly"):
            if key not in prediction:
                fail(errors, f"{path.relative_to(package_dir)} prediction missing key: {key}")
        image_path = str(prediction.get("path", ""))
        expected = expected_label_from_path(image_path)
        if prediction.get("expected_label") != expected:
            actual = prediction.get("expected_label")
            fail(
                errors,
                f"{path.relative_to(package_dir)} expected_label is {actual!r}, "
                f"expected {expected!r}",
            )


def check_package(package_dir: Path) -> list[str]:
    errors: list[str] = []
    if not package_dir.is_dir():
        return [f"HF package directory does not exist: {package_dir}"]

    check_required_files(package_dir, errors)
    check_json_files(package_dir, errors)
    if errors:
        return errors

    check_no_dataset_images(package_dir, errors)
    check_no_local_leakage(package_dir, errors)
    check_readme(package_dir, errors)
    check_artifact_index(package_dir, errors)
    check_prediction_examples(package_dir, errors)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the InspectNet-CX HF package draft.")
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("hf_package/inspectnet-cx"),
        help="Path to the HF package directory.",
    )
    args = parser.parse_args()

    errors = check_package(args.package_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

    print(f"HF package check passed: {args.package_dir}")


if __name__ == "__main__":
    main()
