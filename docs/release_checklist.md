# Release Checklist

Use this checklist before posting the repository or a Hugging Face model repo.

## Repository

- [ ] `pytest -q` passes.
- [ ] `ruff check src tests scripts` passes.
- [ ] `python -m build` succeeds.
- [ ] `LICENSE` is present.
- [ ] `MODEL_CARD.md` is present and does not claim trained performance.
- [ ] `README.md` states the proof boundary clearly.
- [ ] Generated local reports are not committed as benchmark proof.
- [ ] `inspectnet-proof-readiness` reports blockers honestly.
- [ ] `inspectnet-export --check-only --format onnx` has been run.
- [ ] `inspectnet-dataset-check --root ~/datasets` has been run or explicitly blocked.
- [ ] Any PaDiM/classical numbers in docs cite their exact JSON report path.
- [ ] OpenVINO claims mention the current parity status; no deployment claim is made while
      `reports/agent_b/openvino_parity_phase0.json` is `failed`.
- [x] Workstation latency is measured on mewtwo (AMD Ryzen 9 9900X + RTX 5070): CUDA median 0.474 ms/img (p95 0.622 ms) at 512 px; CPU median 2.956 ms/img (p95 3.217 ms) at 512 px.
- [ ] Jetson Orin NX 16GB latency is explicitly marked as future hardware (unmeasured); use `--require-jetson` opt-in if Jetson gating is needed.

## Hugging Face Phase 0 Model

- [ ] Run `make create-phase0-model`.
- [ ] Run `make export-check`.
- [ ] Inspect `artifacts/inspectnet-cx-phase0/README.md`.
- [ ] Verify `InspectNetCXForAnomalyDetection.from_pretrained(...)`.
- [ ] Push only as a Phase 0 scaffold checkpoint.
- [ ] Do not describe Agent B export artifacts as a trained or deployable model.

## Claims Not Allowed Yet

- [ ] Do not claim MVTec AD, VisA, AD2, or LOCO performance.
- [ ] Do not claim calibrated threshold quality.
- [x] Workstation latency is claimed only for mewtwo (AMD Ryzen 9 9900X + RTX 5070) with measured numbers.
- [ ] Do not claim Jetson Orin NX latency (untested; future hardware).
- [ ] Do not claim production defect detection.
- [ ] Do not claim OpenVINO parity until the parity report passes.
