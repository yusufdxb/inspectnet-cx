---
language: en
license: apache-2.0
library_name: anomalib
pipeline_tag: image-classification
tags:
  - anomaly-detection
  - industrial-inspection
  - mvtec-ad
  - padim
  - openvino
  - onnx
---

# InspectNet-CX

![InspectNet-CX release visual](assets/release_visual.svg)

InspectNet-CX is a reproducible industrial anomaly-inspection scaffold with real MVTec AD
bottle PaDiM baseline evidence, reusable checkpoint inference, and early export-path
diagnostics. It is not production-ready or edge-validated.

This is not production factory-inspection software and it is not a fully validated edge model.

## Verified Evidence

| artifact | dataset | method | image AUROC | image F1 | pixel AUROC | pixel F1 | boundary |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `reports/anomalib_padim_mvtec_ad_bottle_result.json` | MVTec AD bottle | Anomalib PaDiM, ResNet-18 | 0.9960 | 0.9756 | 0.9794 | 0.6808 | real local fit/test |
| `reports/classical_patchdiff_rerun_mvtec_ad_bottle_result.json` | MVTec AD bottle | CPU classical patch-difference | 0.9151 | 0.7692 | 0.8765 | 0.4750 | weaker sanity baseline |

Dataset counts for the PaDiM run: 209 normal training images, 20 normal test images, 63 anomalous
test images. The source dataset is MVTec AD `bottle`; MVTec AD is CC BY-NC-SA 4.0 and is not
bundled here.

Every metric in this table is copied from the named JSON report. See `artifact_index.json` and
`claims_ledger.md` for the package-level map from public claim to source artifact.

## Tested Environment And Disk Requirements

The verified Agent B evidence was produced on Ubuntu 22.04 / Linux 6.8, x86_64, Python 3.10.12.
Use Python 3.10 for reproduction. The optional verified stack is pinned in
`requirements-agent_b_verified.txt`.

CUDA is not required for the published PaDiM evidence. The verified baseline and prediction
examples ran on CPU. The pinned optional stack includes Torch `2.11.0+cu128` and Torchvision
`0.26.0+cu128`; that records the tested dependency environment and does not imply Jetson,
TensorRT, or CUDA deployment validation.

Disk planning:

- Compact HF package: under 1 MB.
- Local MVTec AD `bottle` subset used for the verified evidence: about 151 MB.
- Python environment with Anomalib, Torch, ONNX Runtime, and OpenVINO: budget 5-10 GB.
- MVTec AD images, checkpoints, ONNX files, and OpenVINO files are not bundled in this package.

## Real Inference Demo

From the repository checkout:

```bash
PYTHONPATH=src python3 scripts/predict_anomaly.py \
  --backend anomalib_padim \
  --input ~/datasets/mvtec_ad/bottle/test/good/000.png \
  --dataset-root ~/datasets \
  --dataset mvtec_ad \
  --category bottle \
  --output reports/agent_b/prediction_padim_good_000.json

PYTHONPATH=src python3 scripts/predict_anomaly.py \
  --backend anomalib_padim \
  --input ~/datasets/mvtec_ad/bottle/test/broken_large/000.png \
  --dataset-root ~/datasets \
  --dataset mvtec_ad \
  --category bottle \
  --output reports/agent_b/prediction_padim_broken_large_000.json
```

Example outputs are included under `examples/`. They show reusable checkpoint inference on one
normal and one anomalous real MVTec bottle image, with image-level score, predicted label, mask
path, anomaly-map path, checkpoint metadata, and proof boundary.

## Export Status

The trained Anomalib PaDiM checkpoint exists locally at:

```text
artifacts/agent_b/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt
```

`reports/anomalib_padim_export_status.json` records successful exports to:

```text
artifacts/agent_b/anomalib_padim_export/weights/onnx/model.onnx
artifacts/agent_b/anomalib_padim_export/weights/openvino/model.xml
```

The export smoke report, `reports/anomalib_padim_export_smoke.json`, loads both trained export
artifacts and compares ONNX Runtime vs OpenVINO on 83 real MVTec bottle test images. It is marked
`loaded_parity_failed`: the files are real and loadable, but ONNX/OpenVINO parity is not clean
enough for deployment claims.

The Phase 0 placeholder OpenVINO investigation is separate:
`reports/openvino_parity_investigation.json` shows continuous outputs within `4.65e-05` max
absolute error, but binary mask differences occur at hard-threshold boundaries. That does not
validate the trained PaDiM export.

## Reproducible Commands

```bash
python3 -m pip install -e '.[all]'
python3 scripts/check_datasets.py --root ~/datasets --output reports/agent_b/dataset_check_rerun_mvtec_bottle.json
PYTHONPATH=src python3 scripts/run_anomalib_baseline.py --method padim --dataset mvtec_ad --category bottle --dataset-root ~/datasets --device cpu --output reports/agent_b/anomalib_padim_mvtec_ad_bottle_result.json --work-dir artifacts/agent_b/anomalib
PYTHONPATH=src python3 scripts/run_classical_baseline.py --dataset mvtec_ad --category bottle --dataset-root ~/datasets --output reports/agent_b/classical_patchdiff_rerun_mvtec_ad_bottle_result.json
PYTHONPATH=src python3 scripts/predict_anomaly.py --backend anomalib_padim --input ~/datasets/mvtec_ad/bottle/test/good/000.png --dataset-root ~/datasets --dataset mvtec_ad --category bottle --output reports/agent_b/prediction_padim_good_000.json
PYTHONPATH=src python3 scripts/predict_anomaly.py --backend anomalib_padim --input ~/datasets/mvtec_ad/bottle/test/broken_large/000.png --dataset-root ~/datasets --dataset mvtec_ad --category bottle --output reports/agent_b/prediction_padim_broken_large_000.json
PYTHONPATH=src python3 scripts/investigate_anomalib_export.py --checkpoint artifacts/agent_b/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt --dataset-root ~/datasets --dataset mvtec_ad --category bottle --output reports/agent_b/anomalib_padim_export_status.json
PYTHONPATH=src python3 scripts/validate_padim_export.py --onnx artifacts/agent_b/anomalib_padim_export/weights/onnx/model.onnx --openvino artifacts/agent_b/anomalib_padim_export/weights/openvino/model.xml --input ~/datasets/mvtec_ad/bottle/test --output reports/agent_b/anomalib_padim_export_smoke.json
PYTHONPATH=src python3 scripts/investigate_openvino_parity.py --onnx artifacts/agent_b/inspectnet-cx-phase0-rerun/model.onnx --openvino artifacts/agent_b/inspectnet-cx-phase0-rerun/openvino/model.xml --output reports/agent_b/openvino_parity_investigation.json
PYTHONPATH=src pytest -q
ruff check src tests scripts
PYTHONPATH=src python3 scripts/validate_results.py --input reports/agent_b
PYTHONPATH=src python3 scripts/check_hf_package.py
```

## Verified Claims

- Real MVTec AD `bottle` data is configured locally and used for benchmark metrics.
- Anomalib PaDiM fit/test completed with strong image-level and pixel-level metrics.
- A reusable PaDiM Lightning checkpoint can be loaded for prediction through Anomalib.
- InspectNet-CX provides a JSON prediction CLI for real image files and directories.
- Trained PaDiM ONNX and OpenVINO export files were generated.
- Trained export parity is not clean enough for deployment claims.

## Unverified Claims

- No Jetson Orin NX latency has been measured.
- No TensorRT path has been validated.
- No production thresholding or operator workflow has been validated.
- No cross-category MVTec AD, VisA, AD2, or LOCO results are included.
- No trained native InspectNet-CX model checkpoint exists yet.

## Package Contents

- `README.md`: this project card and model-card text.
- `assets/release_visual.svg`: compact pipeline, heatmap, and benchmark-summary visual.
- `requirements.txt`: minimal clean-venv install path for package validation.
- `artifact_index.json`: machine-readable package index with artifact paths, claims, commands,
  and limitations.
- `claims_ledger.md`: human-readable claim-to-artifact ledger.
- `requirements-agent_b_verified.txt`: optional stack pinned from the verified Agent B
  environment.
- `examples/*.json`: prediction CLI output examples for PaDiM and classical backends.
- `reports/*.json`: copied benchmark, export, dataset provenance, and parity reports.

The package intentionally excludes MVTec AD images, generated masks, generated anomaly maps,
model checkpoints, ONNX files, and OpenVINO files.

## Dependencies

See `requirements-agent_b_verified.txt` for the verified optional stack. Key versions include
Anomalib 2.4.1, ONNX Runtime 1.23.2, OpenVINO 2026.1.0, Torch 2.11.0+cu128, Torchvision
0.26.0+cu128, and Timm 1.0.26.
