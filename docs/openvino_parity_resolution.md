# OpenVINO PaDiM Parity Resolution

Date: 2026-05-17
Host: the dev workstation (Intel CPU with native AVX-512 BF16)
onnxruntime: 1.23.2
openvino: 2026.1.0-21367-63e31528c62-releases/2026/1

## Status: RESOLVED

The OpenVINO PaDiM export parity gap closes once OpenVINO's CPU plugin is told to use FP32 instead of its default inference precision (BF16 on this host).

## Hypothesis

The continuous parity gap (max abs error 1.88e-2 on `anomaly_map`, 7.93e-4 on `pred_score`) between ONNX Runtime and OpenVINO on the exported Anomalib PaDiM model was caused by OpenVINO CPU plugin's default `INFERENCE_PRECISION_HINT` selecting BF16 on AVX-512-BF16 hosts. ONNX Runtime stays in FP32 by default. The mismatch is precision policy, not preprocessing, dynamic shape, dtype, or normalization.

## Evidence (single image, MVTec AD `bottle/test/good/000.png`)

Reproducer:

```bash
PYTHONPATH=src python3 scripts/validate_padim_export.py \
  --onnx artifacts/verification/anomalib_padim_export/weights/onnx/model.onnx \
  --openvino artifacts/verification/anomalib_padim_export/weights/openvino/model.xml \
  --input ~/datasets/mvtec_ad/bottle/test/good/000.png \
  --inference-precision <f32|default>
```

| Output       | Default (BF16) max_abs | f32 hint max_abs |
| ------------ | ---------------------- | ---------------- |
| pred_score   | 7.933974e-04           | 2.980232e-08     |
| anomaly_map  | 1.880911e-02           | 7.450581e-07     |
| pred_label   | matches (bool)         | matches (bool)   |
| pred_mask    | matches (bool)         | matches (bool)   |

Effective device default at runtime: `core.get_property("CPU", "INFERENCE_PRECISION_HINT")` returns `<Type: 'bfloat16'>` on this host.

## Evidence (full MVTec AD bottle test split, 83 images)

Reproducer:

```bash
PYTHONPATH=src python3 scripts/validate_padim_export.py \
  --onnx artifacts/verification/anomalib_padim_export/weights/onnx/model.onnx \
  --openvino artifacts/verification/anomalib_padim_export/weights/openvino/model.xml \
  --input ~/datasets/mvtec_ad/bottle/test \
  --inference-precision f32 \
  --output reports/verification/anomalib_padim_export_smoke_f32_bottle_test.json
```

Result with `--inference-precision f32`:

- status: `passed_mask_boundary_unstable`
- input_count: 83
- max_abs_error: 1.7881e-06
- max_mean_abs_error: 1.5497e-06
- max_rel_error: 4.2871e-06
- boolean_outputs_match: false (1 case)
- pred_mask_pixel_flips: 1 / 5,439,488 (1.84e-7 fraction)
- pred_label_flips: 0 / 83 (per-image label parity holds)

The single pred_mask flip is on `bottle/test/broken_large/002.png` at one pixel where `|anomaly_map_ort - anomaly_map_ov| = 2.09e-7`, i.e., the heatmap sits within 2.1e-7 of the binarization threshold. This is intrinsic float boundary noise, not a model defect.

For comparison, with the OpenVINO default (BF16) on a single representative image set (10 images across `good` and `broken_large`):

- map_max: 2.33e-2
- map_mean: 3.39e-3
- score_max: 7.06e-3
- pred_mask flips: 610

## Phase 0 placeholder export

Same fix applies to the Phase 0 InspectNet-CX placeholder export. Re-running the parity comparison with `--inference-precision f32` (recorded in `reports/verification/openvino_parity_investigation.json`):

| Quantity                        | Before (default BF16) | After (f32 hint) |
| ------------------------------- | --------------------- | ---------------- |
| max continuous abs error        | 4.65e-05              | 1.19e-07         |
| max continuous mean abs error   | 4.65e-05              | 5.96e-08         |
| max continuous rel error        | 9.30e-05              | 2.38e-07         |
| binary mask mismatch (seed0)    | 7391 / 50176 px       | 36 / 50176 px    |
| boundary-only mask mismatch     | yes                   | yes              |

All remaining binary mask mismatches are within 1e-3 of the hard threshold (boundary-only).

## Root cause

OpenVINO 2026.1 CPU plugin defaults `INFERENCE_PRECISION_HINT` to `bf16` on hosts with AVX-512 BF16. ONNX Runtime defaults its CPU EP to FP32. The two backends were therefore not running the same numerical computation; the divergence was a precision-policy mismatch, not an export bug.

This is consistent across both exports we validated: the Phase 0 placeholder (Conv + sigmoid + bilinear interpolate) and the real Anomalib PaDiM model (ResNet18 features + Mahalanobis distance + Gaussian blur + bilinear upsample). The Gaussian blur and bilinear upsample steps amplify per-element BF16 rounding to roughly 1e-2 in the final heatmap; reductions over the heatmap then concentrate that error into pred_score.

## Fix

`scripts/validate_padim_export.py` now exposes `--inference-precision {f32, bf16, default}` and defaults to `f32` for parity-strict validation.

The new flag is recorded in the report payload as `inference_precision_hint`. `validate_padim_export.py` also now records per-case `pred_mask_pixel_flips` / `pred_mask_pixel_count` and recognizes a `passed_mask_boundary_unstable` status when continuous parity is at 1e-4 and the total mask flip fraction is at or below 1e-4.

## Deployment guidance

- For accuracy parity validation, run with `--inference-precision f32`. This is the new default.
- For latency / throughput benchmarking on real hardware, leave the hint as `default` (or set `bf16` explicitly) to use the optimized BF16 path. Expect anomaly_map drift up to about 2.3e-2 vs the ORT FP32 reference, and accept that the binary mask boundary will not be byte-identical.
- Per-image `pred_label` parity holds in both modes on the bottle test split; per-pixel `pred_mask` parity holds only in `f32` (mod 1 boundary pixel out of 5.4M).

## Files updated

- `scripts/validate_padim_export.py`: added `--inference-precision`, `pred_mask` pixel-flip accounting, `passed_mask_boundary_unstable` status.
- `reports/verification/openvino_parity_investigation.json`: regenerated with `--inference-precision f32`.
- `reports/verification/anomalib_padim_export_smoke.json`: regenerated with `--inference-precision f32` on a single fixture image.
- `reports/verification/anomalib_padim_export_smoke_f32_bottle_test.json`: full 83-image bottle test sweep.
- `hf_package/inspectnet-cx/reports/openvino_parity_investigation.json`: mirror of the regenerated phase 0 report.

## What this resolution does NOT prove

- It does not prove checkpoint-to-export parity (PyTorch -> ONNX). That is a separate check.
- It does not prove future hardware (Jetson Orin NX 16GB, TensorRT) latency or accuracy; those are untested.
- It does not modify the exported ONNX or OpenVINO graph; only the OpenVINO runtime precision hint changes.
