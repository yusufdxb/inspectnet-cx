# Export Verification Evidence - 2026-05-13

Scope: local workstation evidence for InspectNet-CX using `/home/yusuf/datasets` and this
checkout only. Fixture-only results are not presented as real metrics.

## Environment

- Python: 3.10.12
- Platform: Linux x86_64, CUDA available on NVIDIA GeForce RTX 5070
- Pinned optional stack: see `../../requirements/agent_b_verified.txt`
- Installed package versions: `environment_versions.json`

## Dataset

- Dataset: MVTec AD, `bottle` category
- Local path: `/home/yusuf/datasets/mvtec_ad/bottle`
- Source archive: Hugging Face mirror `micguida1/mvtech_anomaly_detection`
- Official dataset page: `https://www.mvtec.com/research-teaching/datasets/mvtec-ad`
- Archive SHA256: `cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d`
- License note: CC BY-NC-SA 4.0, non-commercial use only
- Structure: 209 train good, 20 test good, 63 anomalous test images, 63 masks

## Real Baseline Results

`anomalib_padim_mvtec_ad_bottle_result.json` is the preserved verified PaDiM result from the
current evidence handoff:

- Method: Anomalib PaDiM, ResNet-18 default stack, CPU
- Image AUROC: 0.9960317611694336
- Image F1: 0.9756097793579102
- Pixel AUROC: 0.9793554544448853
- Pixel F1: 0.6807529926300049
- Train/test counts: 209 normal train, 20 normal test, 63 anomaly test

`padim_rerun_mvtec_ad_bottle_result.json` is an additional real anomaly baseline rerun:

- Method: Anomalib PaDiM, ResNet-18 default stack, CPU
- Image AUROC: 0.9984127283096313
- Image F1: 0.9841269850730896
- Pixel AUROC: 0.9785765409469604
- Pixel F1: 0.6746558547019958
- AU-PRO: TBD
- Elapsed: 18.18325551500004 s
- Train/test counts: 209 normal train, 20 normal test, 63 anomaly test

`classical_patchdiff_rerun_mvtec_ad_bottle_result.json` is the current weaker CPU sanity baseline:

- Image AUROC: 0.915079365079365
- Image F1: 0.7692307692307693
- Pixel AUROC: 0.8764983959395012
- Pixel F1: 0.4749988465974625
- Latency: 15.922462180723063 ms/image on this workstation
- Model size: 0.393216 MB

Older verification files such as `anomalib_padim_mvtec_ad_bottle_result.json` and
`classical_range_mvtec_ad_bottle_result.json` remain useful historical runs, but use the
`*_rerun_*` reports above for current release notes.

## Export And Runtime

- ONNX Phase 0 placeholder export: `artifacts/agent_b/inspectnet-cx-phase0/model.onnx`
- ONNX Runtime verification: passed for the Phase 0 placeholder graph
- OpenVINO IR conversion: `artifacts/agent_b/inspectnet-cx-phase0/openvino/model.xml`
- OpenVINO parity investigation: `openvino_parity_investigation.json` shows continuous outputs
  pass at `1e-4` max absolute tolerance, but binary masks can flip at threshold boundaries.

- Trained PaDiM checkpoint:
  `artifacts/agent_b/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt`
- Reusable checkpoint prediction examples:
  `prediction_padim_good_000.json` and `prediction_padim_broken_large_000.json`
- Trained PaDiM export artifacts:
  `artifacts/agent_b/anomalib_padim_export/weights/onnx/model.onnx` and
  `artifacts/agent_b/anomalib_padim_export/weights/openvino/model.xml`
- Trained PaDiM export status: `anomalib_padim_export_status.json` records successful ONNX and
  OpenVINO artifact creation.
- Trained PaDiM export smoke: `anomalib_padim_export_smoke.json` loads both artifacts across 83
  real MVTec bottle test images but is marked `loaded_parity_failed` with max absolute drift
  `0.0238316059` and non-matching boolean masks.

These export artifacts are real trained Anomalib PaDiM export artifacts, but parity is not clean
and they are not factory deployment evidence.

## Commands Rerun

```bash
python3 -m pip install -e '.[all]'
python3 scripts/check_datasets.py --root /home/yusuf/datasets --output reports/verification/dataset_check_rerun_mvtec_bottle.json
python3 scripts/check_proof_readiness.py --output reports/verification/proof_readiness_after_verification.json
make baseline-anomalib-padim
python3 scripts/run_baseline.py --method classical-range --dataset mvtec_ad --category bottle --data-root /home/yusuf/datasets --output reports/verification/classical_range_mvtec_ad_bottle_result.json
python3 scripts/create_phase0_model.py --output artifacts/agent_b/inspectnet-cx-phase0 --image-size 224
python3 scripts/export_phase0.py --check-only --format onnx --model artifacts/agent_b/inspectnet-cx-phase0
python3 scripts/export_phase0.py --format onnx --model artifacts/agent_b/inspectnet-cx-phase0 --output artifacts/agent_b/inspectnet-cx-phase0/model.onnx --verify
python3 scripts/export_phase0.py --check-only --format openvino --source-onnx artifacts/agent_b/inspectnet-cx-phase0/model.onnx
python3 scripts/export_phase0.py --format openvino --source-onnx artifacts/agent_b/inspectnet-cx-phase0/model.onnx --output artifacts/agent_b/inspectnet-cx-phase0/openvino/model.xml
PYTHONPATH=src python3 scripts/predict_anomaly.py --backend anomalib_padim --input /home/yusuf/datasets/mvtec_ad/bottle/test/good/000.png --dataset-root /home/yusuf/datasets --dataset mvtec_ad --category bottle --output reports/verification/prediction_padim_good_000.json
PYTHONPATH=src python3 scripts/predict_anomaly.py --backend anomalib_padim --input /home/yusuf/datasets/mvtec_ad/bottle/test/broken_large/000.png --dataset-root /home/yusuf/datasets --dataset mvtec_ad --category bottle --output reports/verification/prediction_padim_broken_large_000.json
PYTHONPATH=src python3 scripts/investigate_anomalib_export.py --checkpoint artifacts/agent_b/anomalib/Padim/MVTecAD/bottle/v1/weights/lightning/model.ckpt --dataset-root /home/yusuf/datasets --dataset mvtec_ad --category bottle --output reports/verification/anomalib_padim_export_status.json
PYTHONPATH=src python3 scripts/validate_padim_export.py --onnx artifacts/agent_b/anomalib_padim_export/weights/onnx/model.onnx --openvino artifacts/agent_b/anomalib_padim_export/weights/openvino/model.xml --input /home/yusuf/datasets/mvtec_ad/bottle/test --output reports/verification/anomalib_padim_export_smoke.json
PYTHONPATH=src python3 scripts/investigate_openvino_parity.py --onnx artifacts/agent_b/inspectnet-cx-phase0-rerun/model.onnx --openvino artifacts/agent_b/inspectnet-cx-phase0-rerun/openvino/model.xml --output reports/verification/openvino_parity_investigation.json
python3 scripts/validate_results.py --input reports/verification
ruff check src tests scripts
pytest -q
python3 -m build
```

## Remaining Blockers

- Only MVTec AD `bottle` is present locally. Full MVTec AD, VisA, AD2, and LOCO are still absent.
- Jetson Orin NX latency is unmeasured; this workstation is not Jetson hardware.
- Trained PaDiM ONNX/OpenVINO parity is not clean enough for deployment claims.
- No trained native InspectNet-CX model has been exported.
