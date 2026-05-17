# Public Release Audit

Date: 2026-05-16

## Executive Status

InspectNet-CX is clean-validation ready for Hugging Face publication, but it has not been uploaded from this environment because `hf` CLI authentication failed with `401 Unauthorized`.

## Intended Public URL

- Target: `https://huggingface.co/yusufguenena/inspectnet-cx`
- Status: not published from this session

## Intended Release Tag

- `v0.1.0-inspectnetcx-hf-release`
- Status: not created because this directory is not a Git repository.

## Validation Commands

Clean validation was run from `/tmp/release_validation_inspectnet_final/inspectnet-cx` using a fresh venv:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
inspectnet-create-phase0-model --output artifacts/clean-phase0
inspectnet-infer --model artifacts/clean-phase0 --image artifacts/part.png --output reports/clean_inference.json
PYTHONPATH=src python3 scripts/validate_results.py --input reports/agent_b
PYTHONPATH=src python3 scripts/check_hf_package.py
PYTHONPATH=src pytest -q
ruff check src tests scripts
python3 -m json.tool reports/clean_inference.json
python3 -m json.tool hf_package/inspectnet-cx/artifact_index.json
```

Result: passed. `pytest` reported 38 passing tests and one CUDA driver warning from Torch in the clean venv.

## Public Package Safety

- No raw MVTec AD images are included in `hf_package/inspectnet-cx`.
- No checkpoints, ONNX files, OpenVINO binaries, or safetensors are included.
- No `/home/yusuf` absolute paths remain in the HF package.
- No obvious secrets or token strings were found in the HF package scan.
- The package checker enforces the absence of `/home/yusuf`, `HF_TOKEN`, and private-key markers.

## Exact Public Positioning

InspectNet-CX is a reproducible industrial anomaly-inspection scaffold with real MVTec AD bottle PaDiM baseline evidence, reusable checkpoint inference, and early export-path diagnostics. It is not production-ready or edge-validated.

## Known Limitations

- Not production-ready.
- Not Jetson validated.
- Not TensorRT validated.
- Not cross-dataset benchmarked.
- Not a factory-certified detector.
- Trained export parity is still incomplete.
- MVTec AD images are not bundled.

## Remaining Blockers

- Provide a valid Hugging Face CLI token or expose an authenticated upload tool.
- Convert the directory into a Git repository or attach it to the intended Git remote before creating the release tag.
- After upload, manually verify README rendering, SVG rendering, JSON downloads, tags, and accidental artifact size.
