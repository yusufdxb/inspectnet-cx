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

## Public Positioning

InspectNet-CX is a reproducible industrial anomaly-inspection scaffold with real MVTec AD bottle PaDiM baseline evidence, reusable checkpoint inference examples, and early export-path diagnostics.

It is not production factory-inspection software, not a fully validated edge model, not Jetson validated, and not a trained native InspectNet-CX detector checkpoint.

## Known Limitations

- MVTec AD data is not bundled.
- The native InspectNet-CX model is a Phase 0 scaffold, not the trained detector behind PaDiM metrics.
- Trained PaDiM ONNX/OpenVINO artifacts load, but parity is not clean enough for deployment claims.
- No TensorRT validation, Jetson latency, operator workflow, monitoring, or production threshold validation exists.

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
- Measure target-hardware latency before any edge or Jetson claim.
