from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
from typing import Any

import torch

from inspectnet_cx.data.dataset_check import check_datasets
from inspectnet_cx.export import check_export_readiness

DEFAULT_DATASET_PATHS = {
    "mvtec_ad": Path("~/datasets/mvtec_ad").expanduser(),
    "visa": Path("~/datasets/visa").expanduser(),
    "mvtec_ad2": Path("~/datasets/mvtec_ad2").expanduser(),
    "mvtec_loco": Path("~/datasets/mvtec_loco").expanduser(),
}

REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "transformers",
    "anomalib",
    "timm",
    "onnx",
    "onnxruntime",
    "onnxscript",
)
DEPENDENCY_COMMANDS = {
    "torch": {
        "install_hint": "pip install -e '.[dev]'",
        "enables": ["inspectnet-create-phase0-model", "inspectnet-latency"],
    },
    "torchvision": {
        "install_hint": "pip install -e '.[all]'",
        "enables": ["inspectnet-baseline with Anomalib image transforms"],
    },
    "transformers": {
        "install_hint": "pip install -e '.[dev]'",
        "enables": ["InspectNetCXForAnomalyDetection.from_pretrained"],
    },
    "anomalib": {
        "install_hint": "pip install -e '.[baseline]' or pip install -e '.[all]'",
        "enables": ["inspectnet-baseline --method patchcore|efficientad|padim|simplenet"],
    },
    "timm": {
        "install_hint": "pip install -e '.[baseline]' or pip install -e '.[all]'",
        "enables": ["Anomalib backbone-dependent baselines"],
    },
    "onnx": {
        "install_hint": "pip install -e '.[export]' or pip install -e '.[all]'",
        "enables": ["inspectnet-export --format onnx --verify"],
    },
    "onnxruntime": {
        "install_hint": "pip install -e '.[export]' or pip install -e '.[all]'",
        "enables": ["ONNX export runtime parity checks"],
    },
    "onnxscript": {
        "install_hint": "pip install -e '.[export]' or pip install -e '.[all]'",
        "enables": ["PyTorch 2.11 ONNX graph export"],
    },
    "openvino": {
        "install_hint": "pip install openvino",
        "enables": ["inspectnet-export --format openvino --source-onnx <model.onnx>"],
    },
}
PROOF_GROUPS = {
    "core_api": ("torch", "transformers"),
    "anomalib_baselines": ("anomalib", "torchvision", "timm"),
    "onnx_export": ("onnx", "onnxruntime", "onnxscript"),
    "openvino_export": ("openvino",),
}


def build_readiness_report(dataset_root: Path | None = None) -> dict[str, Any]:
    dataset_paths = _dataset_paths(dataset_root)
    dataset_summary = check_datasets(dataset_root or Path("~/datasets").expanduser())
    packages = {name: _package_status(name) for name in (*REQUIRED_PACKAGES, "openvino")}
    proof_groups = {
        group: {
            "status": "ready"
            if all(packages[package]["installed"] for package in package_names)
            else "blocked",
            "packages": list(package_names),
        }
        for group, package_names in PROOF_GROUPS.items()
    }
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    blocked_reasons = _blocked_reasons(
        packages=packages,
        dataset_summary=dataset_summary,
    )

    return {
        "status": "ready" if not blocked_reasons else "blocked",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "is_jetson_orin_nx": _is_jetson_orin_nx(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "packages": packages,
        "dependency_readiness": {
            name: _dependency_readiness(name, packages[name])
            for name in sorted(packages)
        },
        "proof_groups": proof_groups,
        "datasets": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in dataset_paths.items()
        },
        "dataset_structure": dataset_summary,
        "export_readiness": {
            "onnx": check_export_readiness(export_format="onnx"),
            "openvino": check_export_readiness(export_format="openvino"),
        },
        "blocked_reasons": blocked_reasons,
        "next_commands": [
            "pip install -e '.[all]'",
            "inspectnet-dataset-check --root ~/datasets --output reports/dataset_check.json",
            "inspectnet-export --check-only --format onnx",
            (
                "inspectnet-latency --device auto --image-size 512"
                " --output reports/latency_mewtwo.json"
            ),
        ],
        "proof_requirements": {
            "fixture_smoke": (
                "requires only generated tiny local images; proves command wiring, "
                "not benchmark quality"
            ),
            "real_anomaly_quality": "requires installed Anomalib and real datasets",
            "calibration_quality": (
                "requires normal validation split and threshold-dependent metrics"
            ),
            "benchmark_metrics": "requires MVTec AD, VisA, AD2, and LOCO data",
            "workstation_latency": (
                "measured on mewtwo (AMD Ryzen 9 9900X + RTX 5070); "
                "use --require-jetson opt-in if Jetson Orin NX gating is also needed"
            ),
            "deployability": (
                "requires trained model, export parity, target-runtime latency, monitoring, "
                "operator runbook, and failure-mode validation"
            ),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check proof readiness for InspectNet-CX claims.")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/proof_readiness.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = build_readiness_report(args.dataset_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def _dataset_paths(dataset_root: Path | None) -> dict[str, Path]:
    if dataset_root is None:
        return DEFAULT_DATASET_PATHS
    root = dataset_root.expanduser()
    return {name: root / name for name in DEFAULT_DATASET_PATHS}


def _all_ready(dataset_paths: dict[str, Path]) -> bool:
    packages_ready = all(_package_status(name)["installed"] for name in REQUIRED_PACKAGES)
    datasets_ready = all(path.exists() for path in dataset_paths.values())
    return packages_ready and datasets_ready and _is_workstation_class()


def _package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False, "version": None}
    try:
        version = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"installed": True, "version": version}


def _dependency_readiness(name: str, status: dict[str, Any]) -> dict[str, Any]:
    detail = DEPENDENCY_COMMANDS.get(
        name,
        {"install_hint": "install the package in the active environment", "enables": []},
    )
    return {
        "installed": bool(status["installed"]),
        "version": status["version"],
        "status": "ready" if status["installed"] else "blocked",
        "install_hint": detail["install_hint"],
        "commands_enabled": detail["enables"] if status["installed"] else [],
        "commands_blocked": [] if status["installed"] else detail["enables"],
    }


def _blocked_reasons(
    packages: dict[str, dict[str, Any]],
    dataset_summary: dict[str, Any],
) -> list[str]:
    reasons = []
    for package in ("anomalib", "torchvision", "timm"):
        if not packages[package]["installed"]:
            reasons.append(f"missing Anomalib baseline dependency: {package}")
    for package in ("onnx", "onnxruntime", "onnxscript"):
        if not packages[package]["installed"]:
            reasons.append(f"missing ONNX export/check dependency: {package}")
    if dataset_summary["status"] != "ready":
        reasons.append("one or more benchmark datasets are missing or structurally unverified")
    if not _is_workstation_class():
        reasons.append(
            "no CUDA device detected and CPU does not report AVX2; "
            "workstation-class hardware (CUDA GPU or AVX2 CPU) is required for deployment proofs"
        )
    return reasons


def _is_workstation_class() -> bool:
    """Return True if the host has CUDA available or an AVX2-capable CPU."""
    if torch.cuda.is_available():
        return True
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        try:
            text = cpuinfo_path.read_text()
            if "avx2" in text.lower():
                return True
        except Exception:  # broad catch is intentional: /proc/cpuinfo read may fail on any OS
            pass
    return False


def _is_jetson_orin_nx() -> bool:
    model_paths = [
        Path("/proc/device-tree/model"),
        Path("/sys/firmware/devicetree/base/model"),
    ]
    for path in model_paths:
        if path.exists():
            model = path.read_text(errors="ignore").lower()
            return "orin" in model and "nx" in model
    return False


if __name__ == "__main__":
    main()
