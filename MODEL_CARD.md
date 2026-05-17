---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: image-feature-extraction
tags:
  - anomaly-detection
  - industrial-inspection
  - computer-vision
  - pytorch
  - transformers
  - phase-0
---

# InspectNet-CX Phase 0

InspectNet-CX Phase 0 is a Hugging Face-style scaffold for industrial visual anomaly
detection. It defines a model contract for image score, anomaly heatmap, binary mask,
threshold, confidence, and defect-region outputs.

This is not a trained native InspectNet-CX anomaly detection checkpoint. It is a public
scaffold for the API, packaging, proof-readiness checks, baseline reproduction, and honest
deployment-path investigation.

Current repository evidence includes a real Anomalib PaDiM CPU fit/test on local MVTec AD
`bottle`. The checkpoint is reusable through Anomalib and has been exported to ONNX/OpenVINO,
but it is an Anomalib baseline artifact, not a native InspectNet-CX checkpoint.

The public metric table should use the preserved verified PaDiM report:
`reports/agent_b/anomalib_padim_mvtec_ad_bottle_result.json` (`image_auroc=0.9960`,
`image_f1=0.9756`, `pixel_auroc=0.9794`, `pixel_f1=0.6808`). A separate rerun exists at
`reports/agent_b/padim_rerun_mvtec_ad_bottle_result.json` (`image_auroc=0.9984`,
`image_f1=0.9841`, `pixel_auroc=0.9786`, `pixel_f1=0.6747`). The CPU classical
patch-difference baseline report is `reports/agent_b/classical_patchdiff_rerun_mvtec_ad_bottle_result.json`.
These reports are baseline evidence for the exact local dataset path and environment, not
evidence that this Phase 0 checkpoint is a trained detector.

## Intended Use

- Prototype the InspectNet-CX Python API.
- Verify save and load behavior with `save_pretrained` and `from_pretrained`.
- Test downstream integration code before real training exists.
- Run local latency and proof-readiness checks.

## Out of Scope

- Production inspection.
- Safety-critical reject decisions.
- Benchmark claims on MVTec AD, VisA, AD2, or LOCO.
- Jetson latency claims unless measured on Jetson Orin NX 16GB.

## Example

```python
from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor

processor = InspectNetCXProcessor.from_pretrained("yusufdxb/inspectnet-cx-phase0")
model = InspectNetCXForAnomalyDetection.from_pretrained("yusufdxb/inspectnet-cx-phase0")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

## Limitations

- The Phase 0 model uses a tiny placeholder CNN.
- `support_images` are accepted by the processor but not used by the model yet.
- Calibration utilities are present, but calibration quality is not proven.
- Generated masks and regions are placeholder outputs, not defect-quality evidence.
- ONNX/OpenVINO exports exist for the Phase 0 placeholder and for the trained Anomalib PaDiM
  baseline, but trained export parity is not clean.
- OpenVINO parity for the Phase 0 placeholder is numerically close for continuous outputs, but
  thresholded binary masks can differ.
- MVTec AD evidence is currently limited to the `bottle` category on this local machine.
- ONNX/OpenVINO export mechanics are verified for the trained Anomalib PaDiM artifact, but
  `reports/agent_b/anomalib_padim_export_smoke.json` is marked `loaded_parity_failed`.
- Do not claim OpenVINO deployment readiness until trained checkpoint-to-export parity and
  target-hardware measurements are clean.
- Jetson Orin NX latency and TensorRT compatibility are unproven.

## Proof Boundary

The repository test suite proves importability, CPU forward pass, save and load roundtrip,
processor behavior, CLI surfaces, and proof-readiness reporting.

Real local baseline evidence as of 2026-05-13:

- Dataset: local `~/datasets/mvtec_ad/bottle`, 209 normal train images, 20 normal test
  images, 63 anomalous test images.
- Primary baseline: `reports/agent_b/anomalib_padim_mvtec_ad_bottle_result.json`.
- Metrics: image AUROC `0.9960`, image F1 `0.9756`, pixel AUROC `0.9794`, pixel F1 `0.6808`.

Prediction demo evidence:

- `reports/agent_b/prediction_padim_good_000.json` loads the Anomalib PaDiM checkpoint and
  predicts normal for a real MVTec AD bottle good image.
- `reports/agent_b/prediction_padim_broken_large_000.json` loads the same checkpoint and
  predicts anomaly for a real MVTec AD bottle defective image.

Export evidence:

- `reports/agent_b/anomalib_padim_export_status.json` records successful trained PaDiM ONNX
  and OpenVINO artifact creation.
- `reports/agent_b/anomalib_padim_export_smoke.json` records that those artifacts load, but
  ONNX/OpenVINO parity fails over the 83-image bottle test folder.
- `reports/agent_b/openvino_parity_investigation.json` characterizes Phase 0 placeholder
  OpenVINO drift as mostly threshold-boundary mask instability.

This proves a local anomaly-baseline path, dataset wiring, reusable Anomalib checkpoint
inference, and a real but not deployment-ready export path. It does not prove a trained native
InspectNet-CX checkpoint, Jetson latency, factory deployment readiness, or cross-category
generalization.

Additional local reports under `reports/agent_b/` prove that the optional Anomalib/export stack
is installed and that MVTec AD `bottle` is locally evaluable. Proof readiness remains blocked
because VisA, MVTec AD 2, MVTec LOCO, broader MVTec AD categories, and Jetson latency are still
missing.
