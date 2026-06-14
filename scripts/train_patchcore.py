"""Phase B trainer for PatchCore on MVTec AD.

Mirrors the calling pattern of ``run_anomalib_baseline.py`` but is dedicated to
PatchCore so the coreset subsampling ratio is a first-class CLI hyperparameter
and gets recorded in a sidecar JSON next to the Lightning checkpoint.

The produced checkpoint is consumed by ``scripts/eval_harness.py
--method patchcore`` (Phase A frozen harness).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any


def _seed_everything(seed: int) -> None:
    """Set all RNGs we can reach. Anomalib runs on Lightning, so we also call
    lightning.seed_everything if importable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        from lightning.pytorch import seed_everything as _se

        _se(seed, workers=True)
    except Exception:
        try:
            from pytorch_lightning import seed_everything as _se  # type: ignore

            _se(seed, workers=True)
        except Exception:
            pass


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _lib_versions() -> dict[str, str]:
    import numpy
    import sklearn
    import torch
    import torchvision

    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    try:
        import anomalib

        out["anomalib"] = anomalib.__version__
    except Exception as e:  # pragma: no cover
        out["anomalib"] = f"ERR:{e!r}"
    return out


def _git_head(repo_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except Exception as e:  # pragma: no cover
        return f"ERR:{e!r}"


def _find_ckpt(work_dir: Path) -> Path:
    """Locate the lightning .ckpt produced by anomalib's Engine.fit."""
    candidates = sorted(work_dir.rglob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"no .ckpt found under {work_dir}")
    # Prefer the canonical anomalib path .../weights/lightning/model.ckpt
    for c in candidates:
        if c.name == "model.ckpt" and "lightning" in c.parts:
            return c
    return candidates[-1]


def train(
    *,
    category: str,
    dataset_root: Path,
    backbone: str,
    coreset_ratio: float,
    output_dir: Path,
    device: str,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_workers: int = 0,
    layers: tuple[str, ...] = ("layer2", "layer3"),
    seed: int | None = None,
) -> dict[str, Any]:
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    if seed is not None:
        _seed_everything(seed)

    dataset_root = dataset_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = "gpu" if device == "cuda" else "cpu"

    datamodule = MVTecAD(
        root=dataset_root,
        category=category,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    model = Patchcore(
        backbone=backbone,
        layers=layers,
        pre_trained=True,
        coreset_sampling_ratio=coreset_ratio,
        num_neighbors=9,
    )
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        default_root_dir=output_dir,
        logger=False,
        max_epochs=1,
    )

    started = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_elapsed_s = time.perf_counter() - started

    ckpt_path = _find_ckpt(output_dir)
    ckpt_sha = _file_sha256(ckpt_path)
    ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    repo_root = Path(__file__).resolve().parents[1]
    reproduce_cmd = (
        f"python scripts/train_patchcore.py "
        f"--category {category} "
        f"--dataset-root {dataset_root} "
        f"--backbone {backbone} "
        f"--coreset-ratio {coreset_ratio} "
        f"--output-dir {output_dir} "
        f"--device {device}"
    )

    sidecar = {
        "schema": "inspectnet_cx.train_patchcore.v1",
        "category": category,
        "method": "patchcore",
        "backbone": backbone,
        "layers": list(layers),
        "coreset_sampling_ratio": coreset_ratio,
        "num_neighbors": 9,
        "pre_trained": True,
        "device": device,
        "accelerator": accelerator,
        "seed": seed,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_size_mb": ckpt_size_mb,
        "fit_elapsed_s": fit_elapsed_s,
        "library_versions": _lib_versions(),
        "git_commit": _git_head(repo_root),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "reproduce_command": reproduce_cmd,
        "anomalib_config": {
            "model": "Patchcore",
            "backbone": backbone,
            "layers": list(layers),
            "coreset_sampling_ratio": coreset_ratio,
            "num_neighbors": 9,
            "pre_trained": True,
        },
    }
    sidecar_path = ckpt_path.parent / "train_config.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    sidecar["sidecar_path"] = str(sidecar_path)
    return sidecar


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", required=True)
    p.add_argument("--dataset-root", type=Path, default=Path("~/datasets/mvtec_ad").expanduser())
    p.add_argument("--backbone", default="wide_resnet50_2")
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed all RNGs (random, numpy, torch, lightning). "
             "None means no explicit seeding (legacy Phase B behavior).",
    )
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    out = train(
        category=args.category,
        dataset_root=args.dataset_root,
        backbone=args.backbone,
        coreset_ratio=args.coreset_ratio,
        output_dir=args.output_dir,
        device=args.device,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(
        f"[{out['category']}] trained PatchCore coreset={out['coreset_sampling_ratio']} "
        f"-> {out['checkpoint']} (sha256={out['checkpoint_sha256'][:12]}, "
        f"{out['checkpoint_size_mb']:.1f} MB, fit={out['fit_elapsed_s']:.1f}s)"
    )


if __name__ == "__main__":
    main()
