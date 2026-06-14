"""Phase A eval harness for InspectNet-CX.

Loads an existing PaDiM (or PatchCore) Anomalib checkpoint, runs prediction on
the MVTec AD test split for a single category, and writes a self-contained
result JSON with image AUROC, pixel AUROC, AUPRO, threshold (selected on the
TRAIN split only), dataset hash, checkpoint hash, and library versions.

Hard rule: threshold selection MUST NOT see test labels. See
``select_threshold_from_train`` and ``_threshold_guard``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

NORMAL_NAMES = {"good", "normal"}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------- threshold-leakage guard ----------------------------------------


class ThresholdLeakageError(RuntimeError):
    """Raised if test data is detected on the threshold-selection code path."""


def select_threshold_from_train(train_scores: np.ndarray, *, quantile: float = 0.995) -> float:
    """Pick a threshold using ONLY train (normal) anomaly scores.

    Signature is deliberately single-positional ``train_scores``. There is no
    code path here that accepts test labels or test scores. The runtime guard
    additionally rejects inputs that look like a (scores, labels) pair.
    """
    _threshold_guard(train_scores)
    arr = np.asarray(train_scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("train_scores is empty")
    return float(np.quantile(arr, quantile))


def _threshold_guard(obj: Any) -> None:
    """Refuse anything that smells like test labels reaching threshold code."""
    # Reject tuples/dicts that look like (scores, labels) or {"test_*": ...}.
    if isinstance(obj, tuple) and len(obj) == 2:
        raise ThresholdLeakageError(
            "test data must not reach threshold computation: "
            "select_threshold_from_train received a 2-tuple that looks like "
            "(scores, labels). Pass ONLY train normal scores."
        )
    if isinstance(obj, dict):
        bad = [k for k in obj if "test" in str(k).lower() or "label" in str(k).lower()]
        if bad:
            raise ThresholdLeakageError(
                f"test data must not reach threshold computation: dict keys {bad}"
            )


# ---------- dataset hash ---------------------------------------------------


def category_dataset_hash(category_root: Path) -> str:
    """SHA256 over sorted (relative_path, file_size) for train/test/ground_truth."""
    items: list[tuple[str, int]] = []
    for sub in ("train", "test", "ground_truth"):
        d = category_root / sub
        if not d.is_dir():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMG_EXTS:
                rel = f.relative_to(category_root).as_posix()
                items.append((rel, f.stat().st_size))
    items.sort()
    h = hashlib.sha256()
    for rel, size in items:
        h.update(f"{rel}\t{size}\n".encode())
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- metrics --------------------------------------------------------


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if len(np.unique(labels)) < 2:
        raise ValueError("image_auroc needs both classes present")
    return float(roc_auc_score(labels, scores))


def pixel_auroc(anomaly_maps: np.ndarray, gt_masks: np.ndarray) -> float:
    """anomaly_maps: (N,H,W) float, gt_masks: (N,H,W) {0,1}. Flattened sklearn AUROC."""
    s = np.asarray(anomaly_maps, dtype=np.float64).reshape(-1)
    y = np.asarray(gt_masks, dtype=np.int64).reshape(-1)
    if len(np.unique(y)) < 2:
        raise ValueError("pixel_auroc needs both classes present in flattened masks")
    return float(roc_auc_score(y, s))


def compute_aupro(anomaly_maps_t: torch.Tensor, gt_masks_t: torch.Tensor, fpr_limit: float = 0.3) -> float:
    """anomalib's per-region AUPRO at fpr_limit (standard MVTec setting = 0.3)."""
    from anomalib.metrics.aupro import AUPRO as _Wrapper  # noqa: N811

    raw = [c for c in _Wrapper.__mro__ if c.__name__ == "_AUPRO"][0]
    metric = raw(fpr_limit=fpr_limit)
    # Only anomaly-containing images contribute regions; passing the full set is
    # also fine, since fully-normal masks produce zero connected components.
    metric.update(anomaly_maps_t.float(), gt_masks_t.to(torch.uint8))
    return float(metric.compute().item())


# ---------- versions / env -------------------------------------------------


def _lib_versions(openvino_loaded: bool) -> dict[str, str]:
    import numpy
    import sklearn
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
    if openvino_loaded:
        try:
            import openvino

            out["openvino"] = openvino.__version__
        except Exception as e:  # pragma: no cover
            out["openvino"] = f"ERR:{e!r}"
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


# ---------- main evaluation -----------------------------------------------


def _build_model(method: str):
    from anomalib.models import Padim, Patchcore

    if method == "padim":
        return Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])
    if method == "patchcore":
        return Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"))
    raise ValueError(f"unknown method {method!r}")


def _read_train_sidecar(checkpoint: Path) -> dict[str, Any] | None:
    """Optional sidecar JSON written by scripts/train_patchcore.py.

    Lives at ``<ckpt_dir>/train_config.json``. We surface ``coreset_sampling_ratio``
    in the eval JSON output when present so PatchCore runs at different coreset
    ratios are distinguishable downstream. Absent for PaDiM (Phase A) checkpoints.
    """
    sidecar = checkpoint.parent / "train_config.json"
    if not sidecar.is_file():
        return None
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return None


def _resize_mask(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Load a GT mask and resize to (H,W). Returns {0,1} uint8."""
    from PIL import Image

    img = Image.open(mask_path).convert("L")
    img = img.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    arr = np.array(img, dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def run_eval(
    *,
    checkpoint: Path,
    category: str,
    method: str,
    dataset_root: Path,
    output: Path,
    work_dir: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    category_root = dataset_root / category
    if not category_root.is_dir():
        raise FileNotFoundError(f"missing category root {category_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint {checkpoint}")

    datamodule = MVTecAD(
        root=dataset_root,
        category=category,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=0,
    )
    model = _build_model(method)
    accelerator = "gpu" if device == "cuda" else "cpu"
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        default_root_dir=work_dir,
        logger=False,
        max_epochs=1,
    )

    # --- TEST prediction ---
    test_batches = engine.predict(model=model, datamodule=datamodule, ckpt_path=str(checkpoint))
    if not test_batches:
        raise RuntimeError("no test predictions returned")

    image_scores: list[float] = []
    image_labels: list[int] = []
    image_paths: list[str] = []
    anomaly_maps_list: list[torch.Tensor] = []
    gt_masks_list: list[torch.Tensor] = []

    for b in test_batches:
        paths = list(b.image_path)
        ps = b.pred_score.detach().cpu().reshape(-1).numpy()
        gl = b.gt_label.detach().cpu().reshape(-1).to(torch.int64).numpy()
        am = b.anomaly_map.detach().cpu().float()  # (B,H,W)
        gm = b.gt_mask.detach().cpu().to(torch.uint8)  # (B,H,W) bool->uint8

        for i, p in enumerate(paths):
            image_paths.append(p)
            image_scores.append(float(ps[i]))
            image_labels.append(int(gl[i]))
        anomaly_maps_list.append(am)
        gt_masks_list.append(gm)

    anomaly_maps_t = torch.cat(anomaly_maps_list, dim=0)  # (N,H,W)
    gt_masks_t = torch.cat(gt_masks_list, dim=0)  # (N,H,W) uint8

    scores_np = np.array(image_scores)
    labels_np = np.array(image_labels, dtype=np.int64)

    img_auroc = image_auroc(scores_np, labels_np)
    pix_auroc = pixel_auroc(anomaly_maps_t.numpy(), gt_masks_t.numpy())
    aupro = compute_aupro(anomaly_maps_t, gt_masks_t, fpr_limit=0.3)

    # --- threshold selection: TRAIN split ONLY ---
    # Rerun predict on the train dataloader. The model checkpoint already
    # encodes the PaDiM Gaussian fitted on train, so this is the canonical
    # train-side score distribution. Test labels are never used.
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    train_batches = engine.predict(model=model, dataloaders=train_loader, ckpt_path=str(checkpoint))
    train_scores = np.concatenate(
        [b.pred_score.detach().cpu().reshape(-1).numpy() for b in train_batches]
    )

    # SAFETY GUARD: structurally impossible for test labels to reach here
    # because select_threshold_from_train takes a single positional argument.
    threshold = select_threshold_from_train(train_scores, quantile=0.995)

    # --- assemble result ---
    openvino_loaded = "openvino" in sys.modules
    repo_root = Path(__file__).resolve().parents[1]
    sidecar = _read_train_sidecar(checkpoint)
    coreset_ratio: float | None = None
    if sidecar is not None:
        v = sidecar.get("coreset_sampling_ratio")
        if isinstance(v, (int, float)):
            coreset_ratio = float(v)
    result = {
        "schema": "inspectnet_cx.eval_harness.v1",
        "category": category,
        "method": method,
        "dataset": "mvtec_ad",
        "split": "mvtec_ad_default_train_test",
        "split_hash": category_dataset_hash(category_root),
        "checkpoint": str(checkpoint),
        "checkpoint_hash": file_sha256(checkpoint),
        "image_auroc": img_auroc,
        "pixel_auroc": pix_auroc,
        "aupro": aupro,
        "aupro_fpr_limit": 0.3,
        "threshold": float(threshold),
        "threshold_selection": {
            "rule": "quantile",
            "quantile": 0.995,
            "split": "train",
            "n_train_scores": int(train_scores.size),
        },
        "n_test_images": int(scores_np.size),
        "n_test_anomaly": int((labels_np == 1).sum()),
        "n_test_normal": int((labels_np == 0).sum()),
        "library_versions": _lib_versions(openvino_loaded),
        "git_commit": _git_head(repo_root),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "device": device,
        "coreset_sampling_ratio": coreset_ratio,
        "train_sidecar": (
            {
                "backbone": sidecar.get("backbone"),
                "layers": sidecar.get("layers"),
                "coreset_sampling_ratio": sidecar.get("coreset_sampling_ratio"),
                "checkpoint_sha256": sidecar.get("checkpoint_sha256"),
                "git_commit": sidecar.get("git_commit"),
            }
            if sidecar is not None
            else None
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--category", required=True)
    p.add_argument("--method", default="padim", choices=("padim", "patchcore"))
    p.add_argument("--dataset-root", type=Path, default=Path("~/datasets/mvtec_ad").expanduser())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("artifacts/eval_harness"))
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")
    res = run_eval(
        checkpoint=args.checkpoint.expanduser().resolve(),
        category=args.category,
        method=args.method,
        dataset_root=args.dataset_root.expanduser().resolve(),
        output=args.output,
        work_dir=args.work_dir,
        device=args.device,
    )
    print(
        f"[{res['category']}] image_auroc={res['image_auroc']:.4f} "
        f"pixel_auroc={res['pixel_auroc']:.4f} aupro={res['aupro']:.4f} "
        f"threshold(train q=0.995)={res['threshold']:.4f} -> {args.output}"
    )


if __name__ == "__main__":
    main()
