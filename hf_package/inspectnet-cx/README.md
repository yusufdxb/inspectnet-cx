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

InspectNet-CX is a reproducible industrial anomaly-inspection harness on MVTec AD. This
Hugging Face package bundles the verified evidence (PaDiM and a classical pixel-difference
baseline on MVTec AD `bottle`), a cross-category transfer study, an ONNX/OpenVINO
export-parity investigation with a root-caused fix, prediction examples, and an API scaffold.
It is a research and reproduction artifact, not production factory-inspection software and not
a fully validated edge model.

## Verified Results (bundled)

The two result JSONs in `reports/` are the numeric evidence shipped here. Both were computed
on the local MVTec AD `bottle` category (209 normal-train, 20 normal-test, 63 anomaly-test).

**PaDiM (Anomalib, ResNet-18 backbone), MVTec AD `bottle`:**

| metric      | value  |
| ----------- | -----: |
| image AUROC | 0.9960 |
| image F1    | 0.9756 |
| pixel AUROC | 0.9794 |
| pixel F1    | 0.6808 |

**Classical pixel-difference baseline (normal-only), MVTec AD `bottle`:**

| metric      | value  |
| ----------- | -----: |
| image AUROC | 0.9151 |
| image F1    | 0.7692 |
| pixel AUROC | 0.8765 |
| pixel F1    | 0.4750 |

PaDiM is a strong published reference, not the author's result. The classical baseline is a
real numeric local reference, not a state-of-the-art detector, and must not be compared as one.

Image-level AUROC across four MVTec AD categories, for context (PaDiM and PatchCore are
references, the native reverse-distillation detector is the repo's own work and is described
in the repository, not bundled as a checkpoint here):

| category | PaDiM (ResNet-18) | PatchCore | InspectNet-CX (reverse distillation, repo) |
| -------- | ----------------: | --------: | -----------------------------------------: |
| bottle   | 0.998 | 1.000 | 1.000 |
| cable    | 0.872 | 0.991 | 0.885 |
| capsule  | 0.881 | 0.994 | 0.901 |
| leather  | 0.993 | 1.000 | 1.000 |

The repo's native reverse-distillation detector ties PatchCore on `bottle` and `leather` and
beats PaDiM on all four categories, but still trails PatchCore on `cable` and `capsule`; it
does not beat PatchCore overall. No trained native InspectNet-CX model checkpoint exists yet
in this package: the bundled evidence is the PaDiM and classical baselines above, and the
`InspectNetCX` class shipped here is a packaging/API scaffold.

**Cross-category transfer.** A PaDiM bank fit on one category and scored on another drops image
AUROC by 0.431 (95% bootstrap CI [0.403, 0.458]); the off-diagonal cells collapse to chance
(~0.50). PaDiM is category-specific.

**ONNX/OpenVINO parity.** The ONNX Runtime vs OpenVINO gap on the exported PaDiM model is
OpenVINO's CPU plugin defaulting to BF16 on AVX-512-BF16 hosts. Forcing FP32 drops `pred_score`
max-abs error from 7.9e-4 to 3.0e-8. This parity is not clean enough for deployment claims
without re-verification on the target hardware, and is verified on CPU only.

## Intended Use

- Reproduce MVTec AD anomaly-detection baselines (PaDiM, classical, PatchCore) from a clean harness.
- Study cross-category transfer and export parity.
- Prototype against the InspectNet-CX Python API (`save_pretrained` / `from_pretrained`).

## Out of Scope

- Production or safety-critical inspection.
- Benchmark claims beyond the verified MVTec AD categories.
- Jetson / TensorRT latency claims. No TensorRT path has been validated.

## Limitations

- This package is not production factory-inspection software and not a fully validated edge model.
- The `InspectNetCX` model class is a packaging/API scaffold; masks and regions it emits are not
  defect-quality evidence.
- All evidence is on local MVTec AD; the OpenVINO parity fix is verified on CPU only and that
  parity is not clean enough for deployment claims.
- No trained native InspectNet-CX model checkpoint exists yet in this package.
- No TensorRT path has been validated.

## Reproduction

- Use Python 3.10 for reproduction (verified on Ubuntu 22.04 / Linux 6.8, x86_64, Python 3.10.12).
- CUDA is not required for the published PaDiM evidence: it ran on CPU. A GPU only speeds up
  optional native training.
- MVTec AD is CC BY-NC-SA 4.0 and is not bundled here. Fetch it into a local data root and point
  the scripts at it (the repository ships `scripts/download_mvtec.py`).
- Disk planning: the local `bottle` subset used for the verified evidence is ~151 MB; budget a
  few hundred MB more if you fetch additional categories.

See the repository README and `docs/`: `padim_cross_category_transfer.md`,
`openvino_parity_resolution.md`, `claims_ledger.md`, `latency_baseline.md`.

## License

Code is Apache-2.0. MVTec-derived artifacts (scores, thresholds, result JSONs) inherit MVTec
AD's CC BY-NC-SA 4.0 non-commercial terms. MVTec AD is CC BY-NC-SA 4.0 and is not bundled here;
no MVTec images are redistributed.
