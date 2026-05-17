from __future__ import annotations

import argparse
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

    runtime_device = _resolve_device(device)
    model = InspectNetCXForAnomalyDetection(InspectNetCXConfig(image_size=image_size))
    model.to(runtime_device)
    model.eval()
    pixel_values = torch.randn(batch_size, 3, image_size, image_size, device=runtime_device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(pixel_values=pixel_values)
        _sync(runtime_device)
        start = time.perf_counter()
        for _ in range(iterations):
            model(pixel_values=pixel_values)
        _sync(runtime_device)
        elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / iterations
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
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_image": latency_ms / batch_size,
        "hardware_note": (
            "This is not Jetson proof unless run on Jetson Orin NX 16GB with "
            "--target-hardware jetson-orin-nx-16gb --require-jetson."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Phase 0 InspectNet-CX local latency.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--target-hardware", default="local")
    parser.add_argument("--require-jetson", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/local_latency.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = benchmark_latency(
        image_size=args.image_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
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


if __name__ == "__main__":
    main()
