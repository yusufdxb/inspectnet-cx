# InspectNet-CX Public Claims Ledger

This ledger is scoped to the public README, model card, proof status, Agent B report, and
`hf_package/inspectnet-cx` package draft. Every verified claim below points to a local source
artifact. These are local-machine evidence claims only, not production or edge-deployment claims.

## Verified Claims

| claim | source artifact |
| --- | --- |
| Real MVTec AD `bottle` data is configured locally with 209 normal train images, 20 normal test images, 63 anomalous test images, and 63 masks. | `reports/agent_b/dataset_provenance_mvtec_ad_bottle.json`; `reports/agent_b/anomalib_padim_mvtec_ad_bottle_result.json` |
| Preserved Anomalib PaDiM, ResNet-18, CPU fit/test completed on local MVTec AD `bottle` with image AUROC `0.9960`, image F1 `0.9756`, pixel AUROC `0.9794`, and pixel F1 `0.6808`. | `reports/agent_b/anomalib_padim_mvtec_ad_bottle_result.json` |
| Additional PaDiM rerun completed on the same local category with image AUROC `0.9984`, image F1 `0.9841`, pixel AUROC `0.9786`, and pixel F1 `0.6747`. | `reports/agent_b/padim_rerun_mvtec_ad_bottle_result.json` |
| CPU classical patch-difference sanity baseline completed with image AUROC `0.9151`, image F1 `0.7692`, pixel AUROC `0.8765`, and pixel F1 `0.4750`. | `reports/agent_b/classical_patchdiff_rerun_mvtec_ad_bottle_result.json` |
| Reusable Anomalib PaDiM checkpoint inference predicts `normal` for MVTec AD `bottle/test/good/000.png`. | `reports/agent_b/prediction_padim_good_000.json` |
| Reusable Anomalib PaDiM checkpoint inference predicts `anomaly` for MVTec AD `bottle/test/broken_large/000.png`. | `reports/agent_b/prediction_padim_broken_large_000.json` |
| Prediction JSONs include expected labels, predicted labels, backend metadata, and checkpoint provenance for the PaDiM backend. | `reports/agent_b/prediction_padim_good_000.json`; `reports/agent_b/prediction_padim_broken_large_000.json` |
| Phase 0 ONNX export exists and ONNX Runtime verification passed for the placeholder model. | `reports/agent_b/onnx_export_phase0.json` |
| Phase 0 OpenVINO conversion exists, while binary mask parity remains unstable around threshold boundaries. | `reports/agent_b/openvino_export_phase0.json`; `reports/agent_b/openvino_parity_phase0.json`; `reports/agent_b/openvino_parity_investigation.json` |
| Trained Anomalib PaDiM ONNX and OpenVINO export artifacts were created. | `reports/agent_b/anomalib_padim_export_status.json` |
| Trained exported PaDiM ONNX/OpenVINO artifacts load, but parity fails over the 83-image local bottle test folder. | `reports/agent_b/anomalib_padim_export_smoke.json` |
| The HF package draft includes report copies, example prediction JSONs, dependency pins, an artifact index, and a dataset not-bundled note. | `hf_package/inspectnet-cx/artifact_index.json`; `hf_package/inspectnet-cx/README.md` |

## Partially Verified Claims

| claim | current boundary |
| --- | --- |
| ONNX/OpenVINO export path exists. | Verified as artifact creation and smoke loading, but trained export parity is not clean and checkpoint-to-export parity is not established. |
| OpenVINO Phase 0 parity is close. | Continuous outputs are close in `reports/agent_b/openvino_parity_investigation.json`, but binary masks are unstable at hard-threshold boundaries. |
| The prediction CLI can be run by another user. | Commands are documented and local examples work; another user must provide their own MVTec AD copy and local checkpoint artifacts because dataset files are not bundled. |

## Blocked Claims

| blocked claim | blocker artifact or note |
| --- | --- |
| Production or factory deployment readiness. | Explicitly not proven in `docs/proof_status.md`; requires trained checkpoint parity, target-hardware latency, monitoring, calibration, and operator workflow evidence. |
| Jetson Orin NX latency or edge validation. | `docs/proof_status.md` states this machine is not Jetson hardware and Jetson latency is unmeasured. |
| TensorRT compatibility. | No TensorRT validation artifact exists. |
| Cross-category or cross-dataset benchmark quality. | Only MVTec AD `bottle` is locally evaluated. |
| Clean trained PaDiM ONNX/OpenVINO parity. | `reports/agent_b/anomalib_padim_export_smoke.json` is `loaded_parity_failed`. |
| Trained native InspectNet-CX detector checkpoint. | No trained native InspectNet-CX checkpoint artifact exists. |

## Claims To Remove Or Soften

| wording risk | required wording |
| --- | --- |
| Any standalone "benchmark performance" claim without `bottle`, local path, method, and report path. | Say "local MVTec AD `bottle` PaDiM baseline evidence" and cite the report. |
| Any "export-ready", "deployment-ready", "OpenVINO-ready", "edge-ready", or "Jetson-ready" phrasing. | Say "export artifacts exist and load, but trained export parity is not clean enough for deployment claims." |
| Any implication that the Phase 0 placeholder model is the trained anomaly detector behind the PaDiM metrics. | Say the metrics belong to an Anomalib PaDiM baseline artifact, not a trained native InspectNet-CX checkpoint. |
| Any implication that MVTec AD images are included in the HF package. | Say MVTec AD is CC BY-NC-SA 4.0 and is not bundled. |
