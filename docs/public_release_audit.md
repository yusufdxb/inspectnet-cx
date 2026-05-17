# InspectNet-CX Public Release Audit

Date: 2026-05-17

## Public URL

- Intended Hugging Face repo: `https://huggingface.co/yusufguenena/inspectnet-cx`
- Publication status: blocked. The local `hf` CLI reports `Not logged in`, and no Hugging Face token is available in the environment.

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

## Remaining Blockers

- Hugging Face publication requires a valid token with write access to `yusufguenena/inspectnet-cx`.
- Public page render, public artifact downloads, public SVG render, and repo metadata verification remain blocked until upload succeeds.

## Honest Roadmap

- Publish compact HF package after authentication is available.
- Add trained native InspectNet-CX checkpoint only after training and held-out validation.
- Validate export parity before any OpenVINO deployment claim.
- Measure target-hardware latency before any edge or Jetson claim.
