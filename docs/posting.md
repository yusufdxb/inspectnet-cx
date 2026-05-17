# Posting Guide

Use this framing for public release:

> InspectNet-CX Phase 0 is an HF-style scaffold for industrial visual anomaly detection
> APIs, proof-readiness checks, and future Anomalib-backed baselines.

Do not claim:

- state-of-the-art performance
- trained anomaly detection quality
- benchmark superiority
- production inspection readiness
- Jetson Orin NX latency (untested; future hardware)
- OpenVINO parity or TensorRT compatibility

Allowed evidence framing:

- "A real local Anomalib PaDiM rerun on MVTec AD bottle reports image AUROC 0.9984 and pixel
  AUROC 0.9786 for the exact local dataset/environment."
- "A CPU classical patch-difference baseline reports image AUROC 0.9151 on the same category."
- "Phase 0 ONNX export mechanics pass ONNX Runtime verification, while OpenVINO parity is
  explicitly not passed yet."

## GitHub Post Checklist

```bash
make clean-generated
make test
make lint
python -m build
uvx --from twine twine check dist/*
```

Then inspect:

- `README.md`
- `MODEL_CARD.md`
- `docs/audit.md`
- `docs/proof_status.md`
- `docs/benchmark_protocol.md`
- `docs/release_checklist.md`

## Hugging Face Phase 0 Checkpoint

```bash
make create-phase0-model
```

Upload `artifacts/inspectnet-cx-phase0` only as a Phase 0 scaffold checkpoint. Use the
model card wording from `MODEL_CARD.md`.

Do not upload `artifacts/agent_b/inspectnet-cx-phase0` as a trained detector. It is still a
Phase 0 scaffold with additional ONNX/OpenVINO export files; the OpenVINO parity report is
failed.
