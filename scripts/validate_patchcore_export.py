"""Export trained PatchCore checkpoint to ONNX + OpenVINO and measure parity.

Writes a self-contained JSON to reports/eval_harness/openvino_parity_patchcore_{cat}.json
with per-output error stats, pred_label flips, pred_mask pixel flips, library
versions, git commit, and checkpoint SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def load_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(
        (image_size, image_size), Image.Resampling.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None, ...]


def pick_images(category_root: Path, n: int = 20) -> list[Path]:
    """Pick a mix of normal and anomalous test images."""
    test = category_root / "test"
    normals = sorted((test / "good").glob("*.png"))
    abnormals: list[Path] = []
    for sub in sorted(test.iterdir()):
        if sub.name == "good" or not sub.is_dir():
            continue
        abnormals.extend(sorted(sub.glob("*.png")))
    n_norm = n // 2
    n_ab = n - n_norm
    picked = normals[:n_norm] + abnormals[:n_ab]
    if len(picked) < n:
        picked = (normals + abnormals)[:n]
    return picked[:n]


def export_patchcore(ckpt: Path, out_dir: Path) -> tuple[Path, Path]:
    """Export to ONNX and OpenVINO. Returns (onnx_path, openvino_xml_path)."""
    import torch

    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    # Anomalib checkpoints contain pickled enums/objects (PrecisionType, etc.)
    # that PyTorch >=2.6 blocks under the default weights_only=True. Allowlist
    # them since this is a local trusted checkpoint, and also monkey-patch
    # torch.load to weights_only=False as a belt-and-suspenders.
    try:
        import anomalib as _anomalib_mod
        safe = []
        for attr in ("PrecisionType", "TaskType", "LearningType"):
            obj = getattr(_anomalib_mod, attr, None)
            if obj is not None:
                safe.append(obj)
        if safe:
            torch.serialization.add_safe_globals(safe)
    except Exception:
        pass

    _orig_torch_load = torch.load

    def _patched_load(*args: Any, **kwargs: Any) -> Any:
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore[assignment]
    try:
        model = Patchcore.load_from_checkpoint(str(ckpt))
    finally:
        torch.load = _orig_torch_load  # type: ignore[assignment]
    engine = Engine()

    onnx_dir = out_dir / "onnx"
    ov_dir = out_dir / "openvino"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    ov_dir.mkdir(parents=True, exist_ok=True)

    onnx_path_returned = engine.export(
        model=model,
        export_type=ExportType.ONNX,
        export_root=str(onnx_dir),
    )
    ov_path_returned = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=str(ov_dir),
    )

    # The engine returns a Path to either the file or a directory; locate the actual artifacts.
    onnx_file = None
    if onnx_path_returned is not None:
        p = Path(onnx_path_returned)
        if p.is_file() and p.suffix == ".onnx":
            onnx_file = p
    if onnx_file is None:
        cands = list(onnx_dir.rglob("*.onnx"))
        if not cands:
            raise FileNotFoundError(f"No .onnx file produced under {onnx_dir}")
        onnx_file = cands[0]

    ov_file = None
    if ov_path_returned is not None:
        p = Path(ov_path_returned)
        if p.is_file() and p.suffix == ".xml":
            ov_file = p
    if ov_file is None:
        cands = list(ov_dir.rglob("*.xml"))
        if not cands:
            raise FileNotFoundError(f"No OpenVINO .xml produced under {ov_dir}")
        ov_file = cands[0]

    return onnx_file, ov_file


def compare(onnx_path: Path, ov_path: Path, images: list[Path], image_size: int) -> dict[str, Any]:
    import onnxruntime as ort
    import openvino as ov

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_output_names = [o.name for o in sess.get_outputs()]

    core = ov.Core()
    compiled = core.compile_model(str(ov_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    ov_input = compiled.input(0)
    ov_outputs = compiled.outputs

    per_output_stats: dict[str, dict[str, float]] = {
        name: {"max_abs_error": 0.0, "sum_abs_error": 0.0, "count": 0, "max_rel_error": 0.0}
        for name in onnx_output_names
    }
    pred_label_flips = 0
    pred_label_total = 0
    pred_mask_pixel_flips = 0
    pred_mask_pixel_total = 0

    for img_path in images:
        x = load_image(img_path, image_size)
        onnx_out = sess.run(None, {input_name: x})
        ov_out_raw = compiled({ov_input.any_name: x})
        # Map OV outputs positionally (same order as ONNX inside Engine.export)
        ov_out_list = [ov_out_raw[ov_outputs[i]] for i in range(len(ov_outputs))]

        for name, o_val, v_val in zip(onnx_output_names, onnx_out, ov_out_list):
            o_f = np.asarray(o_val)
            v_f = np.asarray(v_val)
            if o_f.dtype == bool or v_f.dtype == bool:
                ob = o_f.astype(bool)
                vb = v_f.astype(bool)
                flips = int((ob != vb).sum())
                total = int(ob.size)
                if "label" in name.lower():
                    pred_label_flips += flips
                    pred_label_total += total
                elif "mask" in name.lower():
                    pred_mask_pixel_flips += flips
                    pred_mask_pixel_total += total
                continue
            o_fp = o_f.astype(np.float32)
            v_fp = v_f.astype(np.float32)
            diff = np.abs(o_fp - v_fp)
            denom = np.maximum(np.abs(o_fp), 1e-8)
            rel = diff / denom
            s = per_output_stats[name]
            s["max_abs_error"] = max(s["max_abs_error"], float(diff.max()))
            s["sum_abs_error"] += float(diff.sum())
            s["count"] += int(diff.size)
            s["max_rel_error"] = max(s["max_rel_error"], float(rel.max()))

    per_output: dict[str, dict[str, float]] = {}
    for name, s in per_output_stats.items():
        per_output[name] = {
            "max_abs_error": s["max_abs_error"],
            "mean_abs_error": (s["sum_abs_error"] / s["count"]) if s["count"] else 0.0,
            "max_rel_error": s["max_rel_error"],
        }
    return {
        "per_output": per_output,
        "pred_label_flips": pred_label_flips,
        "pred_label_total": pred_label_total,
        "pred_mask_pixel_flips": pred_mask_pixel_flips,
        "pred_mask_pixel_total": pred_mask_pixel_total,
        "pred_mask_pixel_flip_fraction": (
            pred_mask_pixel_flips / pred_mask_pixel_total if pred_mask_pixel_total else 0.0
        ),
        "onnx_output_names": onnx_output_names,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument(
        "--dataset-root", type=Path, default=Path("/home/yusuf/datasets/mvtec_ad")
    )
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import onnxruntime
    import openvino

    ckpt = args.checkpoint
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}", file=sys.stderr)
        return 2
    ckpt_sha = sha256_file(ckpt)
    images = pick_images(args.dataset_root / args.category, n=args.n_images)
    if len(images) < args.n_images:
        print(f"WARNING: only {len(images)} images found", file=sys.stderr)

    result: dict[str, Any] = {
        "schema": "inspectnet_cx.openvino_parity_patchcore.v1",
        "category": args.category,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": ckpt_sha,
        "image_count": len(images),
        "image_size": args.image_size,
        "inference_precision_hint": "f32",
        "device": "CPU",
        "library_versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "onnxruntime": onnxruntime.__version__,
            "openvino": openvino.__version__,
            "anomalib": __import__("anomalib").__version__,
            "torch": __import__("torch").__version__,
        },
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with tempfile.TemporaryDirectory(prefix=f"patchcore_export_{args.category}_") as tmp:
            tmp_dir = Path(tmp)
            onnx_path, ov_path = export_patchcore(ckpt, tmp_dir)
            result["onnx_path"] = str(onnx_path)
            result["openvino_path"] = str(ov_path)
            cmp_stats = compare(onnx_path, ov_path, images, args.image_size)
            result["comparison"] = cmp_stats
            # Categorize parity verdict
            max_abs = max(
                (v["max_abs_error"] for v in cmp_stats["per_output"].values()), default=0.0
            )
            result["summary"] = {
                "max_abs_error_any_output": max_abs,
                "pred_label_flips": cmp_stats["pred_label_flips"],
                "pred_label_total": cmp_stats["pred_label_total"],
                "pred_mask_pixel_flips": cmp_stats["pred_mask_pixel_flips"],
                "pred_mask_pixel_total": cmp_stats["pred_mask_pixel_total"],
                "pred_mask_pixel_flip_fraction": cmp_stats["pred_mask_pixel_flip_fraction"],
            }
            result["status"] = (
                "parity_clean"
                if max_abs <= 1e-3
                and cmp_stats["pred_label_flips"] == 0
                and cmp_stats["pred_mask_pixel_flip_fraction"] <= 1e-4
                else "parity_imperfect"
            )
    except Exception as e:  # noqa: BLE001
        result["status"] = "export_failed"
        result["error"] = f"{type(e).__name__}: {e}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"] if "summary" in result else {"status": result["status"], "error": result.get("error")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
