# Proof Status

This file records what the local machine can and cannot prove.

## Proven Locally

- The Phase 0 package imports and passes the CPU test suite.
- The Phase 0 model runs inference on synthetic tensors.
- The local latency harness runs at 512 px on the available CUDA GPU.
- The proof-readiness harness detects missing dependencies, datasets, and Jetson hardware.
- The tiny fixture smoke command can create local MVTec-style images, calibrate a Phase 0
  threshold, run inference, embed the dataset-check output, and validate the generated proof
  report. This is command wiring evidence, not benchmark evidence.
- Dependency readiness is reported package-by-package, including which commands each optional
  dependency blocks or unlocks.
- Export readiness can report blocked ONNX/OpenVINO dependency or file states without creating
  false deployment claims.
- Dataset readiness can identify conservative MVTec/VisA-style normal training image structure.
- Normal-only threshold calibration has a CLI and blocked-state path, but only proves threshold
  computation when local normal images exist.
- CPU-only classical baselines can produce numeric local metrics from normal training images
  and labeled MVTec-style test folders without Anomalib. Fixture results prove executable
  evaluation plumbing, not official benchmark performance.
- The active Agent B environment has Anomalib and export dependencies installed and pinned in
  `pyproject.toml` plus `requirements/agent_b_verified.txt`.
- A real Anomalib PaDiM runner exists for local MVTec AD categories; completed reports are valid
  only for the exact local dataset path and package stack recorded in the report.
- MVTec AD `bottle` exists under local `~/datasets/mvtec_ad/bottle` and passes the local
  structure check for train/test/ground-truth folders.
- Agent B reran real Anomalib PaDiM on MVTec AD `bottle`:
  `reports/agent_b/padim_rerun_mvtec_ad_bottle_result.json` reports image AUROC `0.9984`,
  image F1 `0.9841`, pixel AUROC `0.9786`, and pixel F1 `0.6747` on `209` normal training,
  `20` normal test, and `63` anomaly test images.
- The earlier verified PaDiM report remains preserved at
  `reports/agent_b/anomalib_padim_mvtec_ad_bottle_result.json` with image AUROC `0.9960`,
  image F1 `0.9756`, pixel AUROC `0.9794`, and pixel F1 `0.6808`.
- Agent B reran the CPU classical patch-difference baseline:
  `reports/agent_b/classical_patchdiff_rerun_mvtec_ad_bottle_result.json` reports image AUROC
  `0.9151`, image F1 `0.7692`, pixel AUROC `0.8765`, pixel F1 `0.4750`, and latency
  `15.9225` ms/image on this x86_64 machine.
- `scripts/predict_anomaly.py` provides real-image JSON inference for both
  `classical_patchdiff` and `anomalib_padim`. The PaDiM path loaded
  `artifacts/agent_b/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt` and
  produced correct normal/anomaly example predictions in
  `reports/agent_b/prediction_padim_good_000.json` and
  `reports/agent_b/prediction_padim_broken_large_000.json`.
- Phase 0 ONNX export exists at `artifacts/agent_b/inspectnet-cx-phase0/model.onnx`; ONNX Runtime
  verification passed with binary mask match ratio `0.9997209821`.
- Phase 0 OpenVINO conversion exists at `artifacts/agent_b/inspectnet-cx-phase0/openvino/model.xml`.
  Conversion succeeded, but parity is not proven because `reports/agent_b/openvino_parity_phase0.json`
  is marked `failed` on a threshold-boundary random case.
- `reports/agent_b/openvino_parity_investigation.json` narrows the Phase 0 OpenVINO issue:
  continuous outputs pass at `1e-4` max absolute tolerance, while binary mask mismatches occur
  at hard-threshold boundaries.
- Trained Anomalib PaDiM export was attempted and produced ONNX/OpenVINO artifacts under
  `artifacts/agent_b/anomalib_padim_export/weights/`. `reports/agent_b/anomalib_padim_export_status.json`
  records export success.
- `reports/agent_b/anomalib_padim_export_smoke.json` proves the trained export artifacts load
  in ONNX Runtime and OpenVINO, but marks parity `loaded_parity_failed` over 83 real MVTec
  bottle test images. This blocks trained export deployment claims.
- `hf_package/inspectnet-cx/` contains an honest Hugging Face package draft with model-card
  text, report copies, prediction examples, dependency pins, and a dataset license note.

## Not Proven Locally

- Cross-category or cross-dataset anomaly detection quality.
- Normal-only calibration quality for production thresholds.
- Full benchmark metrics on all of MVTec AD, VisA, AD2, or LOCO.
- State-of-the-art or Anomalib-equivalent baseline quality from the classical CPU baselines.
- Jetson Orin NX 16GB latency.
- Clean ONNX/OpenVINO parity for a trained model.
- Checkpoint-to-export parity for trained PaDiM.
- TensorRT compatibility.
- Factory deployment readiness.

## Current Blockers

- Only the MVTec AD `bottle` category is present under `~/datasets`; VisA, AD2, LOCO, and the
  rest of MVTec AD are still missing.
- This machine is x86_64 with an NVIDIA GeForce RTX GPU, not Jetson Orin NX 16GB.
- Trained PaDiM ONNX/OpenVINO export artifacts exist, but parity is not clean enough for
  deployment claims.
- No trained native InspectNet-CX model has been exported.
- The proof-readiness reports remain `blocked` because benchmark dataset coverage and target
  hardware latency are incomplete.

## Readiness Estimate

- HF-style anomaly scaffold release: ready with strict baseline/deployment-scaffold wording.
- Local MVTec AD `bottle` PaDiM baseline note: ready as exact-path local evidence.
- Full benchmark claim: not ready.
- Deployment/edge claim: not ready.

## Repeatable Commands

```bash
make test
make lint
make proof-readiness
make export-check
make dataset-check
make baseline-anomalib-padim
python scripts/run_anomalib_baseline.py --method padim --dataset mvtec_ad --category bottle --dataset-root ~/datasets --device cpu --output reports/agent_b/padim_rerun_mvtec_ad_bottle_result.json --work-dir reports/agent_b/padim_rerun
python scripts/run_classical_baseline.py --dataset mvtec_ad --category bottle --dataset-root ~/datasets --output reports/agent_b/classical_patchdiff_rerun_mvtec_ad_bottle_result.json
python scripts/export_phase0.py --format onnx --model artifacts/agent_b/inspectnet-cx-phase0 --output artifacts/agent_b/inspectnet-cx-phase0/model.onnx --verify
python scripts/export_phase0.py --format openvino --source-onnx artifacts/agent_b/inspectnet-cx-phase0/model.onnx --output artifacts/agent_b/inspectnet-cx-phase0/openvino/model.xml
PYTHONPATH=src python3 scripts/predict_anomaly.py --backend anomalib_padim --input ~/datasets/mvtec_ad/bottle/test/good/000.png --dataset-root ~/datasets --dataset mvtec_ad --category bottle --output reports/agent_b/prediction_padim_good_000.json
PYTHONPATH=src python3 scripts/predict_anomaly.py --backend anomalib_padim --input ~/datasets/mvtec_ad/bottle/test/broken_large/000.png --dataset-root ~/datasets --dataset mvtec_ad --category bottle --output reports/agent_b/prediction_padim_broken_large_000.json
PYTHONPATH=src python3 scripts/investigate_anomalib_export.py --checkpoint artifacts/agent_b/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt --dataset-root ~/datasets --dataset mvtec_ad --category bottle --output reports/agent_b/anomalib_padim_export_status.json
PYTHONPATH=src python3 scripts/validate_padim_export.py --onnx artifacts/agent_b/anomalib_padim_export/weights/onnx/model.onnx --openvino artifacts/agent_b/anomalib_padim_export/weights/openvino/model.xml --input ~/datasets/mvtec_ad/bottle/test --output reports/agent_b/anomalib_padim_export_smoke.json
PYTHONPATH=src python3 scripts/investigate_openvino_parity.py --onnx artifacts/agent_b/inspectnet-cx-phase0-rerun/model.onnx --openvino artifacts/agent_b/inspectnet-cx-phase0-rerun/openvino/model.xml --output reports/agent_b/openvino_parity_investigation.json
make latency
make jetson-latency
```
