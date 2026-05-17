# Sprint 2 Summary (InspectNet-CX)

Date: 2026-05-17
Branch: main
Commits pushed: 4 (this repo) + 2 cross-referenced in PreFailureNet

## Landed in this repo

### `4c291d1` — feat: verify PaDiM + threshold analysis across cable, capsule, leather
- Ran `score_anomalib_test.py` + `threshold_analysis.py` on three additional MVTec AD categories.
- Aggregated `docs/threshold_analysis_padim_multi_category.md`:
  - bottle:   AUROC 0.998, F1-max 0.992 @ thr 0.500
  - cable:    AUROC 0.872, F1-max 0.860 @ thr 0.500
  - capsule:  AUROC 0.881, F1-max 0.951 @ thr 0.500
  - leather:  AUROC 0.993, F1-max 0.984 @ thr 0.500
- Confirmed operating-point thresholds are category-specific; bottle's Youden threshold (0.5502) does not transfer to cable.
- pytest: `54 passed`. ruff: `All checks passed!`.
- Note: anomalib's MVTec download URL returns 404 as of 2026-05-17; extracted cable/capsule from the cached HuggingFace tarball (hash matches expected `cf4313b1...`).

### `c3594fc` — fix: close OpenVINO PaDiM parity gap via INFERENCE_PRECISION_HINT=f32
- Status: RESOLVED.
- Root cause: OpenVINO 2026.1 CPU plugin defaults `INFERENCE_PRECISION_HINT` to bfloat16 on AVX-512-BF16 silicon; ORT stays in FP32. The two backends were not running the same numerical computation.
- Real Anomalib export, bottle test set, 83 images, after f32 hint:
  - map max abs diff: 1.79e-6
  - pred_label parity: 83 / 83
  - pred_mask pixel flips: 1 / 5,439,488 (1.84e-7, boundary-only)
  - status flag: `passed_mask_boundary_unstable`
- Added `--inference-precision {f32, bf16, default}` to `validate_padim_export.py` and `investigate_openvino_parity.py` (defaults to f32).
- Caveat: does not validate Jetson/TensorRT parity. Bf16 deployers should opt in explicitly and accept the bounded drift.
- pytest: `54 passed`. ruff: `All checks passed!`.

### `9c90614` — feat: add Phase 1 native detector training script
- New `scripts/train_phase1_detector.py` + `src/inspectnet_cx/training/phase1.py` + CLI `inspectnet-train-phase1`.
- Loads MVTec AD normal-only images, BCE loss against zero targets on anomaly head, Adam optimizer.
- Smoke run on 10 bottle images, 5 epochs, CPU, 65s: loss 0.297 -> 0.000012.
- Saves via `model.save_pretrained` / `processor.save_pretrained`; `from_pretrained` reload verified in `tests/test_phase1_training.py`.
- Status: scaffold (per public_release_audit.md); not a production-trained artifact.
- pytest: `54 passed`. ruff: `All checks passed!`.

### `337e60a` — feat: end-to-end latency benchmark with hardware fingerprint
- Extended `src/inspectnet_cx/eval/latency.py` and CLI `inspectnet-latency`.
- Measures input load, preprocess, forward, threshold decision, total; reports mean/median/p95.
- Hardware fingerprint via `/proc/cpuinfo`, `nvidia-smi`, `/etc/nv_tegra_release`; sets `jetson: true` when applicable.
- Workstation baseline on mewtwo (AMD Ryzen 9 9900X + RTX 5070): CUDA 256px median 0.275 ms/img (p95 0.391), CUDA 512px median 0.474 ms/img (p95 0.622); CPU 256px median 0.685 ms/img (p95 0.894), CPU 512px median 2.956 ms/img (p95 3.217).
- New flags: `--n-runs --image-size --batch-size --warmup --device --target-hardware --require-jetson --output`.
- pytest: `54 passed`. ruff: `All checks passed!`.

## Cross-referenced commits in PreFailureNet
- `3a323b1` v3 XOR counterfactual split (DEFEATS_ORDERED_LR, drops ordered LR AP from 0.876 to 0.549)
- `3680878` real browser-agent trace recorder scaffold

## Open items (not blockers)
- Phase 1 model is a scaffold; no production training run.
- OpenVINO parity validated on CPU only; TensorRT path still owed. Jetson Orin NX 16GB is listed as future hardware.
- Cable AUROC 0.872 is the weakest of the four verified categories; recommend documenting category-specific operating points rather than a single global threshold.
