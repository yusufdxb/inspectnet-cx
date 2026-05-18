# InspectNet-CX Public Release Audit

Date: 2026-05-17

## Public URL

- Hugging Face repo: `https://huggingface.co/yusufdxb/inspectnet-cx`
- Visibility: public model repo.
- Publication status: live.
- Upload commit (HF): `6a048ac1fa4ddcb9538d3cdc008b0de0fbe17a9a`.
- Upload timestamp (UTC): `2026-05-17T17:53:09Z`.
- Authenticated as: `yusufdxb` (write scope).
- Namespace note: the brief named `yusufguenena/inspectnet-cx`, but the active HF account is `yusufdxb`. Published under `yusufdxb/inspectnet-cx` with explicit user confirmation. No `yusufguenena` namespace was registered.

## Release Tag

- Local tag: `v0.1.0-inspectnetcx-hf-release`
- Note: the source directory was not a Git repository at release time, so local Git metadata was initialized before tagging.

## Clean Validation

Validation was run from a temporary Git snapshot cloned into `/tmp/inspectnet-cx-release-clone`, with a fresh `.venv` and a sanitized shell environment to avoid ROS/PYTHONPATH leakage.

Commands validated:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
ruff check src tests scripts
python -m build
inspectnet-fixture-smoke --output-dir reports/fixture_smoke --image-size 32
inspectnet-validate-results --input reports/fixture_smoke
inspectnet-create-phase0-model --output artifacts/inspectnet-cx-phase0-clean
inspectnet-infer --model artifacts/inspectnet-cx-phase0-clean --image reports/fixture_smoke/datasets/mvtec_ad/bottle/test/good/000.png --output reports/release_clean_inference.json
PYTHONPATH=src python3 scripts/predict_anomaly.py --backend classical_patchdiff --input reports/fixture_smoke/datasets/mvtec_ad/bottle/test/good/000.png --dataset-root reports/fixture_smoke/datasets --dataset mvtec_ad --category bottle --output reports/release_prediction_classical_good.json
PYTHONPATH=src python3 scripts/validate_results.py --input reports/agent_b
PYTHONPATH=src python3 scripts/check_hf_package.py
python scripts/release_smoke.py
```

Result: passed. Pytest reported `38 passed`; ruff passed; package build passed; HF package checker passed; SVG parsing and artifact index checks passed.

## Fixes Required

- Replaced the README inference example that referenced missing untracked `part.png` with a reproducible fixture-backed image path.
- Replaced absolute local dataset examples in root public docs with portable `~/datasets` examples.
- Reran clean validation after fixes.

## Package Safety

- HF package size: about 224 KB.
- No executable files in `hf_package/inspectnet-cx`.
- No raw MVTec AD images, masks, anomaly maps, checkpoints, ONNX files, OpenVINO files, NPZ arrays, caches, or secrets in the HF package.
- Package checker enforces absence of local home paths, Hugging Face token markers, and private-key markers.

## Multi-Category PaDiM Verification (Sprint 2)

PaDiM (ResNet-18, layers layer1/layer2/layer3) has been verified on four MVTec AD
categories. Commands and per-category detail are in
`docs/threshold_analysis_padim_multi_category.md`.

Results (image-level AUROC, Youden F1, F1-max F1):

| category | AUROC  | Youden F1 | F1-max F1 |
| -------- | -----: | --------: | --------: |
| bottle   | 0.9976 |    0.9756 |    0.9921 |
| cable    | 0.8720 |    0.8601 |    0.8601 |
| capsule  | 0.8807 |    0.9439 |    0.9507 |
| leather  | 0.9925 |    0.9834 |    0.9836 |

Key finding: operating-point thresholds are category-specific and must not be
reused across categories. Leather and bottle are strong fits for this PaDiM
variant; cable and capsule are harder (AUROC 0.87-0.88).

## Sprint 3 Rigor (2026-05-17)

Sprint 3 turned Sprint 2's per-category point estimates into CI-aware
evidence, replaced the degenerate Phase 1 BCE-against-zero stub with an
honest learned baseline, and quantified PaDiM's category-specificity
under the anomalib post-processor.

### Proven

- **Bootstrap 95% CIs on PaDiM AUROC, Youden-F1, F1-max-F1** for all four
  cached categories (n=1000 stratified percentile bootstrap, seed=0).
  See `docs/threshold_analysis_padim_multi_category.md`.
  Headlines:
  - bottle AUROC 0.9976 [0.9905, 1.0000]
  - cable AUROC 0.8720 [0.8156, 0.9264] (widest informative CI)
  - capsule AUROC 0.8807 [0.7786, 0.9713] (CI lower bound below 0.80)
  - leather AUROC 0.9925 [0.9769, 1.0000]
- **Honest Phase 1 native baseline trained on RTX 5070.** Small
  reconstruction autoencoder (855,619 params, 0.188 ms/img CUDA at
  128 px), 30 epochs Adam lr=1e-3, 80/20 train/val split on MVTec AD
  `bottle` and `cable` train/good. Per-image score = mean squared
  reconstruction error.
- **Head-to-head native vs PaDiM** (same bootstrap methodology, see
  `docs/native_vs_padim_bottle.md`):
  - bottle: PaDiM 0.998 [0.991, 1.000] vs ReconAE 0.824 [0.730, 0.910]
  - cable:  PaDiM 0.872 [0.816, 0.926] vs ReconAE 0.697 [0.602, 0.781]
  - Native CIs do not overlap PaDiM CIs on either category; native
    bottle CI lower bound 0.730 is below the 0.85 acceptance line.
    Documented as a **negative result**; the scaffold did not become
    production.
- **PaDiM cross-category transfer matrix** (4x4 over bottle/cable/
  capsule/leather; pill substituted by capsule due to local data
  availability). See `docs/padim_cross_category_transfer.md`.
  Headline: mean off-diagonal AUROC drop is **0.431 +- 0.014**
  (95% bootstrap CI [0.403, 0.458], n=1000). Many cross-cells land
  at exactly 0.500 due to the anomalib `OneClassPostProcessor`
  saturating; worst cells (`bottle -> capsule` 0.458,
  `leather -> capsule` 0.482) score below chance. PaDiM is
  empirically category-specific under this configuration.

### Skipped, with reasons recorded

- **Extending PaDiM to pill/screw/tile/wood (Deliverable B).** Only
  bottle/cable/capsule/leather are present under `~/datasets/mvtec_ad/`.
  The anomalib bundled download URL returned HTTP 404 on 2026-05-17. No
  alternative mirror was attempted within the budget. Coverage stays at
  4/15 of MVTec AD. Recorded in
  `docs/threshold_analysis_padim_multi_category.md`.
- **TensorRT FP32 vs ORT FP32 parity on RTX 5070 (Deliverable E).**
  TensorRT, polygraphy, and `trtexec` are all absent from mewtwo; the
  Sprint 3 hard constraint forbids new system installs. Documented in
  `docs/tensorrt_parity_rtx5070.md`.

### Residual gaps (still scaffold)

- 11 of 15 MVTec AD categories untested.
- VisA, AD2, LOCO untested.
- Native InspectNet-CX detector remains **below PaDiM** on both tested
  categories; no production claim.
- No TensorRT parity. No Jetson Orin NX 16GB latency.
- The released `InspectNetCXForAnomalyDetection` model surface in the
  v0.1.0 HF tag is intentionally untouched; the reconstruction AE
  lives in `inspectnet_cx.training.phase1_recon` and is not part of
  the public model card. Promoting it requires its own release cycle.

### Verification (this sprint, trailing lines)

- `PYTHONPATH=src python3 -m pytest -q`: 61 passed, 1 warning.
- `ruff check src tests scripts`: All checks passed.
- `PYTHONPATH=src python3 scripts/check_hf_package.py`: HF package
  check passed.

## Public Positioning

InspectNet-CX is a reproducible industrial anomaly-inspection scaffold with real
MVTec AD PaDiM baseline evidence across bottle, cable, capsule, and leather
categories, reusable checkpoint inference examples, and early export-path
diagnostics.

It is not production factory-inspection software, not a fully validated edge model, and not a trained native InspectNet-CX detector checkpoint. The model IS validated on mewtwo (AMD Ryzen 9 9900X + NVIDIA RTX 5070): CUDA median 0.474 ms/img (p95 0.622 ms) and CPU median 2.956 ms/img (p95 3.217 ms) at 512 px. Jetson Orin NX 16GB is untested.

## Known Limitations

- MVTec AD data is not bundled.
- The native InspectNet-CX model is a Phase 0 scaffold, not the trained detector behind PaDiM metrics.
- Trained PaDiM ONNX/OpenVINO artifacts load and reach FP32 parity (max abs error 1.79e-6, 1 boundary mask pixel out of 5.44M) on the MVTec AD bottle test split when OpenVINO is compiled with `INFERENCE_PRECISION_HINT=f32`. See `docs/openvino_parity_resolution.md`. With the BF16 default on AVX-512-BF16 CPUs, anomaly_map drifts up to about 2.3e-2 vs the ORT FP32 reference. No target-hardware (Jetson, TensorRT) parity claim is made.
- No TensorRT validation, operator workflow, monitoring, or production threshold validation exists. Jetson Orin NX 16GB is untested.

## Post-Publication Verification

- README rendered on the public page; positioning sentence ("reproducible industrial anomaly-inspection scaffold ... It is not production-ready or edge-validated") confirmed via HTML fetch (10 in-page hits across body + meta).
- All 16 content files plus `.gitattributes` listed by the HF API (`/api/models/yusufdxb/inspectnet-cx`); file tree matches `hf_package/inspectnet-cx/` byte-for-byte.
- Asset URLs verified by HEAD request (all return HTTP 307 to LFS/CDN, which is the standard HF resolve flow):
  - `resolve/main/README.md`
  - `resolve/main/assets/release_visual.svg`
  - `resolve/main/claims_ledger.md`
  - `resolve/main/artifact_index.json`
  - `resolve/main/reports/anomalib_padim_mvtec_ad_bottle_result.json`
- Repo metadata tags as published: `anomalib`, `anomaly-detection`, `industrial-inspection`, `mvtec-ad`, `padim`, `openvino`, `onnx`, `image-classification`, `en`, `license:apache-2.0`, `region:us`. Pipeline tag: `image-classification`. Matches README YAML front-matter.
- No dataset leakage: MVTec AD images, masks, anomaly maps, ONNX, OpenVINO, NPZ caches, and secrets are all absent from the uploaded tree (confirmed before upload by file inventory and post-upload by API file list).

## Remaining Blockers

- None outstanding for the v0.1.0 public release.

## Honest Roadmap

- Publish compact HF package after authentication is available.
- Add trained native InspectNet-CX checkpoint only after training and held-out validation.
- Validate export parity before any OpenVINO deployment claim.
- Measure Jetson Orin NX 16GB latency before any Jetson edge claim.
