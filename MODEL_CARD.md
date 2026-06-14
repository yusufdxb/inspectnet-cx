---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: image-classification
tags:
  - anomaly-detection
  - industrial-inspection
  - computer-vision
  - mvtec-ad
  - padim
  - patchcore
  - pytorch
---

# InspectNet-CX

InspectNet-CX is a reproducible industrial anomaly-inspection harness on MVTec AD. It ships a
natively-trained student-teacher detector, two verified classical baselines (PaDiM and
PatchCore), a cross-category transfer study, and an ONNX/OpenVINO export-parity investigation
with a root-caused fix. The native detector (`src/inspectnet_cx/models/student_teacher.py`) is
trained here; the separate Hugging Face-style `InspectNetCX` class
(`modeling_inspectnet_cx.py`) is a packaging/API scaffold.

## Verified Results

Image-level AUROC, matched train/test, four MVTec AD categories:

| category | PaDiM (ResNet-18) | PatchCore | InspectNet-CX (student-teacher, ours) |
| -------- | ----------------: | --------: | ------------------------------------: |
| bottle   | 0.998 | 1.000 | 1.000 |
| cable    | 0.872 | 0.991 | 0.751 |
| capsule  | 0.881 | 0.994 | 0.888 |
| leather  | 0.993 | 1.000 | 0.913 |

The native student-teacher detector ties PatchCore on `bottle` and edges PaDiM on `capsule`, but
honestly trails PatchCore on `cable` and `leather`. PaDiM/PatchCore are strong references, not
the author's results.

Cross-category transfer: a PaDiM bank fit on one category and scored on another drops AUROC by
0.431 (95% bootstrap CI [0.403, 0.458]); off-diagonal cells collapse to ~0.50. PaDiM is
category-specific.

ONNX/OpenVINO parity: the ORT-vs-OpenVINO gap on the exported PaDiM model is OpenVINO's CPU
plugin defaulting to BF16 on AVX-512-BF16 hosts. Forcing FP32 drops `pred_score` max-abs error
from 7.9e-4 to 3.0e-8.

## Intended Use

- Reproduce MVTec AD anomaly-detection baselines (PaDiM, PatchCore) from a clean harness.
- Study cross-category transfer and export parity.
- Prototype against the InspectNet-CX Python API (`save_pretrained` / `from_pretrained`).

## Out of Scope

- Production or safety-critical inspection.
- Benchmark claims beyond the four verified MVTec AD categories.
- Jetson / TensorRT latency claims (unmeasured).

## Limitations

- The `InspectNetCX` model class is a placeholder CNN; masks and regions it emits are not
  defect-quality evidence.
- All evidence is on local MVTec AD; the OpenVINO parity fix is verified on CPU only.
- `support_images` are accepted by the processor but not yet used by the model.

## License

Code is Apache-2.0. MVTec-derived artifacts (scores, thresholds, result JSONs) inherit MVTec
AD's CC BY-NC-SA 4.0 non-commercial terms. No MVTec images are redistributed here.

## Reproduction

See the repository README and `docs/`: `padim_cross_category_transfer.md`,
`openvino_parity_resolution.md`, `claims_ledger.md`, `latency_baseline.md`.
