from __future__ import annotations

import argparse
import contextlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch

from inspectnet_cx import InspectNetCXConfig, InspectNetCXForAnomalyDetection


def benchmark_latency(
    image_size: int = 512,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 50,
    device: str = "cpu",
    target_hardware: str = "local",
    require_jetson: bool = False,
) -> dict[str, Any]:
    is_jetson = _is_jetson_orin_nx()
    if require_jetson and not is_jetson:
        return {
            "status": "blocked",
            "blocked_reasons": ["Jetson Orin NX 16GB latency requires running on Jetson hardware"],
            "target_hardware": target_hardware,
            "is_jetson_orin_nx": is_jetson,
            "platform": platform.platform(),
            "hardware_note": "No latency was measured because the target hardware gate failed.",
        }

    hardware_info = _get_hardware_fingerprint()
    runtime_device = _resolve_device(device)
    model = InspectNetCXForAnomalyDetection(InspectNetCXConfig(image_size=image_size))
    model.to(runtime_device)
    model.eval()

    timings = []
    pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=runtime_device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(pixel_values=pixel_values)
        _sync(runtime_device)

        for _ in range(iterations):
            start = time.perf_counter()
            model(pixel_values=pixel_values)
            _sync(runtime_device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms)

    timings_sorted = sorted(timings)
    median_ms = timings_sorted[len(timings) // 2]
    p95_ms = timings_sorted[int(len(timings) * 0.95)] if len(timings) > 1 else median_ms
    mean_ms = sum(timings) / len(timings)

    return {
        "status": "local_phase0_latency",
        "device": str(runtime_device),
        "target_hardware": target_hardware,
        "is_jetson_orin_nx": is_jetson,
        "platform": platform.platform(),
        "image_size": image_size,
        "batch_size": batch_size,
        "warmup": warmup,
        "iterations": iterations,
        "latency_ms_per_batch": {
            "mean": mean_ms,
            "median": median_ms,
            "p95": p95_ms,
        },
        "latency_ms_per_image": {
            "mean": mean_ms / batch_size,
            "median": median_ms / batch_size,
            "p95": p95_ms / batch_size,
        },
        "hardware": hardware_info,
        "hardware_note": (
            "Workstation latency measurement on the current host; opt in to --require-jetson "
            "if you specifically need Jetson Orin NX hardware gating."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Phase 0 InspectNet-CX local latency.")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Number of timing runs for latency measurement (default: 50).",
    )
    parser.add_argument(
        "--image-size", type=int, default=512, help="Input image size (default: 512)."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1).")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs (default: 10).")
    parser.add_argument("--iterations", type=int, default=50, help="(Deprecated: use --n-runs).")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "auto"),
        help="Device to run on: cpu, cuda, or auto (default: cpu).",
    )
    parser.add_argument("--target-hardware", default="local", help="Target hardware label.")
    parser.add_argument(
        "--require-jetson", action="store_true", help="Gate benchmark to Jetson hardware only."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/local_latency.json"), help="Output JSON path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    iterations = args.n_runs if args.n_runs != 50 else args.iterations
    result = benchmark_latency(
        image_size=args.image_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=iterations,
        device=args.device,
        target_hardware=args.target_hardware,
        require_jetson=args.require_jetson,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA requested but torch.cuda.is_available() is false."
        raise RuntimeError(msg)
    return torch.device(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def _get_hardware_fingerprint() -> dict[str, Any]:
    fingerprint = {
        "jetson": _is_jetson_orin_nx(),
        "cpu_model": _get_cpu_model(),
        "gpu_device": None,
    }

    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            fingerprint["gpu_device"] = torch.cuda.get_device_name(0)

    tegra_path = Path("/etc/nv_tegra_release")
    if tegra_path.exists():
        fingerprint["jetson"] = True

    return fingerprint


def _get_cpu_model() -> str | None:
    cpuinfo_path = Path("/proc/cpuinfo")
    if not cpuinfo_path.exists():
        return None

    with contextlib.suppress(Exception):
        for line in cpuinfo_path.read_text().split("\n"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()

    return None


if __name__ == "__main__":
    main()
