# Reports

This directory is for generated local reports. JSON and Markdown reports are ignored by git
except this file.

Generate current local reports with:

```bash
make proof-readiness
make latency
```

Generated reports are machine-specific evidence, not portable benchmark claims.

Current verification evidence files:

- `verification/padim_rerun_mvtec_ad_bottle_result.json`: real local Anomalib PaDiM fit/test on
  MVTec AD `bottle`.
- `verification/classical_patchdiff_rerun_mvtec_ad_bottle_result.json`: CPU classical baseline on
  the same local category.
- `verification/dataset_check_rerun_mvtec_bottle.json`: local dataset readiness check.
- `verification/dataset_provenance_mvtec_ad_bottle.json`: archive source, checksum, license, and
  extracted file counts.
- `verification/onnx_export_phase0.json`, `verification/openvino_export_phase0.json`, and
  `verification/openvino_parity_phase0.json`: Phase 0 export evidence. OpenVINO parity is currently
  failed, so do not treat conversion as deployment proof.
