"""Latency benchmark for PatchCore checkpoints (CPU + GPU, batch=1).

For each checkpoint we:
  - load the model from .ckpt via anomalib's Patchcore.load_from_checkpoint,
  - move to the target device,
  - feed N>=50 test/good images sized to the training resolution (256x256 default),
  - warm up >=10 iterations (not counted),
  - time per-image forward with time.perf_counter, plus torch.cuda.synchronize for GPU,
  - write reports/eval_harness/patchcore_<cat>_latency.json with hardware
    fingerprint, library versions, and min/median/p95/mean/std.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _cpu_model_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _gpu_fingerprint() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip().splitlines()[0]
    except Exception as e:
        return f"ERR:{e!r}"


def _lib_versions() -> dict[str, str]:
    import numpy
    import torchvision

    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    try:
        import anomalib

        out["anomalib"] = anomalib.__version__
    except Exception as e:
        out["anomalib"] = f"ERR:{e!r}"
    return out


def _load_test_images(category_root: Path, n: int, image_size: int) -> torch.Tensor:
    good = category_root / "test" / "good"
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(
        p for p in good.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        raise FileNotFoundError(f"no test/good images under {good}")
    # If we have fewer than n, loop with replacement to hit n.
    picked = []
    i = 0
    while len(picked) < n:
        picked.append(paths[i % len(paths)])
        i += 1
    tensors = []
    for p in picked:
        img = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # HWC -> CHW
        t = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(t)
    return torch.stack(tensors, dim=0)  # (N,3,H,W)


def _load_patchcore(ckpt: Path):
    from anomalib.models import Patchcore

    # Reconstruct config from sidecar if available, otherwise defaults.
    sidecar_path = ckpt.parent / "train_config.json"
    backbone = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset = 0.1
    if sidecar_path.is_file():
        d = json.loads(sidecar_path.read_text())
        backbone = d.get("backbone", backbone)
        ls = d.get("layers", layers)
        layers = tuple(ls)
        coreset = float(d.get("coreset_sampling_ratio", coreset))
    model = Patchcore(
        backbone=backbone,
        layers=layers,
        pre_trained=True,
        coreset_sampling_ratio=coreset,
    )
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def benchmark(
    *,
    category: str,
    checkpoint: Path,
    dataset_root: Path,
    device: str,
    n_images: int = 50,
    n_warmup: int = 10,
    image_size: int = 256,
) -> dict[str, Any]:
    model = _load_patchcore(checkpoint)
    dev = torch.device("cuda" if device == "cuda" else "cpu")
    model = model.to(dev)

    imgs = _load_test_images(
        dataset_root / category, n=n_images + n_warmup, image_size=image_size
    ).to(dev)

    timings_ms: list[float] = []
    with torch.no_grad():
        # Warmup
        for i in range(n_warmup):
            x = imgs[i:i + 1]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
        # Timed
        for i in range(n_warmup, n_warmup + n_images):
            x = imgs[i:i + 1]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings_ms.append((t1 - t0) * 1000.0)

    arr = np.array(timings_ms, dtype=np.float64)
    return {
        "device": device,
        "n_images": n_images,
        "n_warmup": n_warmup,
        "image_size": image_size,
        "batch_size": 1,
        "min_ms": float(arr.min()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "raw_ms": timings_ms,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("~/datasets/mvtec_ad").expanduser())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-images", type=int, default=50)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--devices", nargs="+", default=["cpu", "cuda"], choices=("cpu", "cuda"))
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    ckpt = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()

    results: dict[str, Any] = {
        "schema": "inspectnet_cx.bench_latency.v1",
        "category": args.category,
        "checkpoint": str(ckpt),
        "dataset_root": str(dataset_root),
        "hardware": {
            "cpu": _cpu_model_name(),
            "gpu": _gpu_fingerprint(),
            "platform": platform.platform(),
        },
        "library_versions": _lib_versions(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "per_device": {},
    }
    for d in args.devices:
        if d == "cuda" and not torch.cuda.is_available():
            results["per_device"][d] = {"error": "cuda not available"}
            continue
        results["per_device"][d] = benchmark(
            category=args.category,
            checkpoint=ckpt,
            dataset_root=dataset_root,
            device=d,
            n_images=args.n_images,
            n_warmup=args.n_warmup,
            image_size=args.image_size,
        )
        # Don't pollute the JSON file with 50 raw floats per device; keep summary only.
        results["per_device"][d].pop("raw_ms", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for d, r in results["per_device"].items():
        if "error" in r:
            print(f"[{args.category}/{d}] {r['error']}")
        else:
            print(
                f"[{args.category}/{d}] median={r['median_ms']:.2f}ms "
                f"p95={r['p95_ms']:.2f}ms mean={r['mean_ms']:.2f}ms (n={r['n_images']})"
            )


if __name__ == "__main__":
    main()
