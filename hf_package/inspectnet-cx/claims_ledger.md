# InspectNet-CX Public Claims Ledger

This package is scoped to local MVTec AD `bottle` evidence and release diagnostics. Every
verified public claim below points to an artifact included in this package.

## Verified Claims

| claim | package artifact |
| --- | --- |
| Real MVTec AD `bottle` data was configured locally with 209 normal train images, 20 normal test images, 63 anomalous test images, and 63 masks. | `reports/dataset_provenance_mvtec_ad_bottle.json`; `reports/anomalib_padim_mvtec_ad_bottle_result.json` |
| Anomalib PaDiM, ResNet-18, CPU fit/test completed on local MVTec AD `bottle` with image AUROC `0.9960`, image F1 `0.9756`, pixel AUROC `0.9794`, and pixel F1 `0.6808`. | `reports/anomalib_padim_mvtec_ad_bottle_result.json` |
| CPU classical patch-difference sanity baseline completed with image AUROC `0.9151`, image F1 `0.7692`, pixel AUROC `0.8765`, and pixel F1 `0.4750`. | `reports/classical_patchdiff_rerun_mvtec_ad_bottle_result.json` |
| Reusable Anomalib PaDiM checkpoint inference predicts `normal` for a real MVTec AD `bottle/test/good/000.png` example. | `examples/prediction_padim_good_000.json` |
| Reusable Anomalib PaDiM checkpoint inference predicts `anomaly` for a real MVTec AD `bottle/test/broken_large/000.png` example. | `examples/prediction_padim_broken_large_000.json` |
| Prediction JSONs include expected labels, predicted labels, backend metadata, and checkpoint provenance for the PaDiM backend. | `examples/prediction_padim_good_000.json`; `examples/prediction_padim_broken_large_000.json` |
| Trained Anomalib PaDiM ONNX and OpenVINO export artifacts were created locally. | `reports/anomalib_padim_export_status.json` |
| Trained exported PaDiM ONNX/OpenVINO artifacts load, but parity fails over the 83-image local bottle test folder. | `reports/anomalib_padim_export_smoke.json` |
| Phase 0 placeholder OpenVINO continuous-output parity is close, but binary mask parity is unstable at hard-threshold boundaries. | `reports/openvino_parity_investigation.json` |

## Partially Verified Claims

| claim | current boundary |
| --- | --- |
| ONNX/OpenVINO export path exists. | Verified as artifact creation and smoke loading, but trained export parity is not clean and checkpoint-to-export parity is not established. |
| Prediction CLI can be rerun by another user. | Commands are documented and examples are included; users must provide their own MVTec AD copy and local checkpoint artifacts because dataset files and checkpoints are not bundled. |

## Blocked Claims

| blocked claim | blocker |
| --- | --- |
| Production or factory deployment readiness. | No production calibration, monitoring, operator workflow, target hardware latency, or clean trained export parity. |
| Jetson Orin NX latency or edge validation. | No Jetson measurement exists. |
| TensorRT compatibility. | No TensorRT validation artifact exists. |
| Cross-category or cross-dataset benchmark quality. | Only MVTec AD `bottle` is included. |
| Clean trained PaDiM ONNX/OpenVINO parity. | `reports/anomalib_padim_export_smoke.json` is `loaded_parity_failed`. |
| Trained native InspectNet-CX detector checkpoint. | No trained native InspectNet-CX checkpoint exists. |

## Claims To Remove Or Soften

| wording risk | required wording |
| --- | --- |
| Standalone benchmark-performance claims. | Say "local MVTec AD `bottle` PaDiM baseline evidence" and cite the report. |
| "Deployment-ready", "edge-ready", "Jetson-ready", or "OpenVINO-ready". | Say export artifacts exist and load, but trained export parity is not clean enough for deployment claims. |
| Any implication that MVTec AD images are included. | Say MVTec AD is CC BY-NC-SA 4.0 and is not bundled. |
