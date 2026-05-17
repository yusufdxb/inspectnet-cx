# InspectNet-CX

![InspectNet-CX release visual](hf_package/inspectnet-cx/assets/release_visual.svg)

InspectNet-CX is a reproducible industrial anomaly-inspection scaffold with real MVTec AD
bottle PaDiM baseline evidence, reusable checkpoint inference, and early export-path
diagnostics. It is not production-ready or edge-validated.

The project targets factory inspection, robotics QA, surface defects, assembly verification,
and rare-defect discovery as a research scaffold.

The positioning is specific: HF-style API, calibrated reject decisions, and few-shot
normal-prototype adaptation. The CNN backbone is an implementation choice, not the moat.

## What This Is

- A runnable Python package scaffold for anomaly score, heatmap, mask, threshold, confidence,
  and defect-region outputs.
- A CPU-safe placeholder model that supports `save_pretrained` and `from_pretrained`.
- A real Anomalib PaDiM baseline path on local MVTec AD `bottle`, plus a JSON prediction CLI
  for reusable checkpoint inference.

## What This Is Not

- Not a trained native InspectNet-CX anomaly detection checkpoint.
- Not a broad benchmark claim or a trained native InspectNet-CX detector performance claim.
- Not an upstream Transformers task implementation.
- Not a dataset downloader.
- Not deployable factory-inspection software. Deployment still requires a trained checkpoint,
  held-out dataset metrics, export parity, target-hardware latency, monitoring, and an operator
  runbook.

## Tested Environment And Disk Requirements

This release evidence was produced on Ubuntu 22.04 / Linux 6.8, x86_64, Python 3.10.12.
Use Python 3.10 for reproduction; `pyproject.toml` allows Python `>=3.10`, but the verified
Agent B optional stack was pinned on Python 3.10.12.

CUDA is optional for the published MVTec AD bottle evidence. The verified PaDiM baseline and
prediction examples ran on CPU. The optional verified environment includes Torch
`2.11.0+cu128` and Torchvision `0.26.0+cu128`; this is dependency provenance, not a CUDA or
Jetson validation claim.

Disk planning:

- Compact HF package: under 1 MB.
- Local MVTec AD `bottle` subset used for the verified evidence: about 151 MB.
- Python environment with Anomalib, Torch, ONNX Runtime, and OpenVINO: budget 5-10 GB.
- Full external datasets are not bundled; keep them outside the repository under a local data
  root such as `~/datasets`.

## Setup

Use `uv` if available:

```bash
make setup
```

Fresh-clone validation path:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Install optional baseline, export, and Space dependencies only when needed:

```bash
pip install -e ".[all]"
```

## Validation

```bash
pytest -q
ruff check src tests scripts
python -m build
```

## Example Usage

```python
from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor

processor = InspectNetCXProcessor.from_pretrained(
    "yusufdxb/inspectnet-cx-small",
    trust_remote_code=True,
)
model = InspectNetCXForAnomalyDetection.from_pretrained(
    "yusufdxb/inspectnet-cx-small",
    trust_remote_code=True,
)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
```

For Phase 0, use the project classes directly. A true
`AutoModelForAnomalyDetection.from_pretrained(...)` call would require upstream Transformers
support or a project-specific wrapper.

## Local Inference CLI

Create a tiny synthetic fixture and a local Phase 0 model:

```bash
inspectnet-fixture-smoke --output-dir reports/fixture_smoke --image-size 32
```

Or create only the local Phase 0 model and processor:

```bash
inspectnet-create-phase0-model --output artifacts/inspectnet-cx-phase0
```

Then run:

```bash
inspectnet-infer \
  --model artifacts/inspectnet-cx-phase0 \
  --image reports/fixture_smoke/datasets/mvtec_ad/bottle/test/good/000.png \
  --output reports/inference.json
```

From an uninstalled checkout, use:

```bash
python scripts/run_inference.py \
  --model artifacts/inspectnet-cx-phase0 \
  --image reports/fixture_smoke/datasets/mvtec_ad/bottle/test/good/000.png \
  --output reports/inference.json
```

The JSON output includes image score, threshold, confidence, defect regions, heatmap shape,
and mask shape. It does not claim real defect quality in Phase 0.

## Baseline Harness

The baseline scripts have three evidence levels. `run_baseline.py` records command plans for
unrun Anomalib methods, `run_anomalib_baseline.py` performs a real Anomalib fit/test for supported
local datasets, and the CPU-only `classical_patchdiff` runner trains and evaluates a small
normal-only baseline directly on local MVTec-style folders without Anomalib.

```bash
python scripts/run_baseline.py \
  --method patchcore \
  --dataset mvtec_ad \
  --category bottle \
  --data-root ~/datasets \
  --device cpu \
  --output reports/patchcore_mvtec_ad_bottle.json

python scripts/run_baseline.py \
  --plan-only \
  --method patchcore \
  --dataset visa \
  --category candle \
  --data-root ~/datasets

python scripts/run_anomalib_baseline.py \
  --method padim \
  --dataset mvtec_ad \
  --category bottle \
  --dataset-root ~/datasets \
  --device cpu \
  --output reports/anomalib_padim_mvtec_ad_bottle.json

python scripts/run_classical_baseline.py \
  --dataset mvtec_ad \
  --category bottle \
  --dataset-root ~/datasets \
  --output reports/classical_patchdiff_mvtec_ad_bottle.json

python scripts/aggregate_results.py --input reports --output reports/baselines.md
```

After installation, the same commands are available as console scripts:

```bash
inspectnet-baseline --method patchcore --dataset mvtec_ad --category bottle --device cpu
inspectnet-anomalib-baseline --method padim --dataset mvtec_ad --category bottle --dataset-root ~/datasets --device cpu
inspectnet-classical-baseline --dataset mvtec_ad --category bottle --dataset-root ~/datasets
inspectnet-aggregate --input reports --output reports/baselines.md
```

The generated `anomalib_command` field is a reproducibility scaffold, not proof that Anomalib
has run. `run_anomalib_baseline.py` writes numeric metrics only after Anomalib completes a real
fit/test. The classical baseline fits a normal-image pixel-difference model, calibrates its
threshold from normal training scores, and reports image-level AUROC/F1 for the exact local path.
This is not factory readiness or target-hardware latency evidence.

Current local Agent B evidence on MVTec AD `bottle`:

| report | method | image AUROC | image F1 | pixel AUROC | pixel F1 | boundary |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `reports/agent_b/anomalib_padim_mvtec_ad_bottle_result.json` | Anomalib PaDiM, ResNet-18, CPU | `0.9960` | `0.9756` | `0.9794` | `0.6808` | real local fit/test on `~/datasets/mvtec_ad/bottle` only |
| `reports/agent_b/padim_rerun_mvtec_ad_bottle_result.json` | Anomalib PaDiM, ResNet-18, CPU | `0.9984` | `0.9841` | `0.9786` | `0.6747` | real local fit/test on `~/datasets/mvtec_ad/bottle` only |
| `reports/agent_b/classical_patchdiff_rerun_mvtec_ad_bottle_result.json` | CPU classical patch-difference | `0.9151` | `0.7692` | `0.8765` | `0.4750` | simple local baseline, not Anomalib-equivalent |

The local bottle subset has `209` normal training images, `20` normal test images, and `63`
anomaly test images. The provenance report records the source archive and checksum in
`reports/agent_b/dataset_provenance_mvtec_ad_bottle.json`; the dataset license is
CC BY-NC-SA 4.0, so treat it as non-commercial research evidence.

## Real Prediction CLI

Run a reusable Anomalib PaDiM checkpoint on real MVTec AD bottle images:

```bash
PYTHONPATH=src python3 scripts/predict_anomaly.py \
  --backend anomalib_padim \
  --input ~/datasets/mvtec_ad/bottle/test/good/000.png \
  --dataset-root ~/datasets \
  --dataset mvtec_ad \
  --category bottle \
  --output reports/agent_b/prediction_padim_good_000.json

PYTHONPATH=src python3 scripts/predict_anomaly.py \
  --backend anomalib_padim \
  --input ~/datasets/mvtec_ad/bottle/test/broken_large/000.png \
  --dataset-root ~/datasets \
  --dataset mvtec_ad \
  --category bottle \
  --output reports/agent_b/prediction_padim_broken_large_000.json
```

The same CLI supports `--backend classical_patchdiff` as a dependency-light sanity baseline.
The PaDiM path loads the Anomalib Lightning checkpoint and writes JSON with image score,
predicted label, mask path, anomaly-map path, backend metadata, and a proof note. This is real
local inference, not optimized deployment evidence.

## Proof Checks

Run proof readiness before claiming benchmark or deployment results:

```bash
inspectnet-proof-readiness --output reports/proof_readiness.json
```

Measure local Phase 0 latency:

```bash
inspectnet-latency --device auto --image-size 512 --iterations 50 --output reports/local_latency.json
```

Local latency is not Jetson proof unless the command is run on Jetson Orin NX 16GB.

For target-hardware latency claims, use the hardware gate:

```bash
inspectnet-latency \
  --target-hardware jetson-orin-nx-16gb \
  --require-jetson \
  --device auto \
  --image-size 512 \
  --iterations 50 \
  --output reports/jetson_latency.json
```

On non-Jetson machines this reports a blocked state instead of a latency claim.

Run the tiny fixture-backed smoke path:

```bash
inspectnet-fixture-smoke --output-dir reports/fixture_smoke --image-size 32
inspectnet-validate-results --input reports/fixture_smoke
```

This creates a tiny synthetic MVTec-style directory, runs the dataset checker, calibrates from
normal images, runs Phase 0 inference on held-out fixture images, runs the classical
pixel-difference baseline, writes a proof report, and validates that the proof report is not
misclassified as benchmark JSON. It is command evidence only, not benchmark evidence.

The proof-readiness report includes `dependency_readiness` entries for `anomalib`, `onnx`,
`onnxruntime`, `openvino`, `torch`, `torchvision`, `transformers`, and `timm`. Each entry states
which commands are blocked or unlocked by that package.

## Export Checks

Check export dependencies without creating artifacts:

```bash
inspectnet-export --check-only --format onnx
inspectnet-export --check-only --format openvino --source-onnx artifacts/inspectnet-cx-phase0/model.onnx
```

Export the Phase 0 placeholder model to ONNX only after creating a local Phase 0 model:

```bash
inspectnet-create-phase0-model --output artifacts/inspectnet-cx-phase0
inspectnet-export \
  --format onnx \
  --model artifacts/inspectnet-cx-phase0 \
  --output artifacts/inspectnet-cx-phase0/model.onnx \
  --verify
```

This validates export mechanics for the placeholder model only. It does not prove TensorRT,
Jetson, or production deployment readiness.

Agent B Phase 0 export artifacts:

- `artifacts/agent_b/inspectnet-cx-phase0/model.onnx`
- `artifacts/agent_b/inspectnet-cx-phase0/model.onnx.data`
- `artifacts/agent_b/inspectnet-cx-phase0/openvino/model.xml`
- `reports/agent_b/onnx_export_phase0.json`
- `reports/agent_b/openvino_export_phase0.json`
- `reports/agent_b/openvino_parity_phase0.json`

ONNX Runtime verification passed for the Phase 0 placeholder model. The newer parity
investigation at `reports/agent_b/openvino_parity_investigation.json` shows continuous outputs
within `4.6492e-05` max absolute error across deterministic inputs, but binary mask values can
flip at hard-threshold boundaries. Do not claim OpenVINO deployment parity from Phase 0 alone.

Trained Anomalib PaDiM export was also attempted and is now real:

- `artifacts/agent_b/anomalib_padim_export/weights/onnx/model.onnx`
- `artifacts/agent_b/anomalib_padim_export/weights/openvino/model.xml`
- `reports/agent_b/anomalib_padim_export_status.json`
- `reports/agent_b/anomalib_padim_export_smoke.json`

The trained export files load in ONNX Runtime and OpenVINO, but the smoke report over 83 real
MVTec bottle test images is marked `loaded_parity_failed` with max absolute drift `0.0238316`
and non-matching boolean masks. Treat this as export-path evidence, not deployment readiness.

## Threshold Calibration

If local MVTec AD or VisA normal images exist, calibrate the Phase 0 threshold from normal-only
images:

```bash
inspectnet-calibrate-normal-threshold \
  --model artifacts/inspectnet-cx-phase0 \
  --dataset mvtec_ad \
  --category bottle \
  --dataset-root ~/datasets \
  --output reports/normal_threshold.json
```

This records the threshold source as `normal_only_local_split`. It does not prove anomaly
quality or threshold-dependent benchmark metrics.

## Dataset Instructions

Do not download datasets during Phase 0. For Phase 1, download datasets from their official
sources and place them under a local data root outside the repository, for example:

- MVTec AD: `~/datasets/mvtec_ad`
- VisA: `~/datasets/visa`
- MVTec AD 2: `~/datasets/mvtec_ad2`
- MVTec LOCO AD: `~/datasets/mvtec_loco`

Record exact download URLs, checksums where available, and dataset versions in the run report.

Check local dataset readiness with:

```bash
inspectnet-dataset-check --root ~/datasets --output reports/dataset_check.json
```

The checker looks for conservative Anomalib-style category structure such as
`~/datasets/mvtec_ad/bottle/train/good`. For VisA it also recognizes the common converted
`~/datasets/visa/visa_pytorch/<category>/train/good` form.

See `docs/benchmark_protocol.md` before publishing any result table.

## Roadmap

Phase 0:

- Create the package scaffold.
- Define the HF-style config, processor, output contract, and model stub.
- Add CPU tests and baseline JSON schema.

Phase 1:

- Wire real Anomalib PatchCore and EfficientAD runs.
- Add real feature extraction, memory bank logic, and normal-only calibration.
- Produce honest baseline tables on MVTec AD and VisA.

Phase 2:

- Evaluate AD2 and LOCO.
- Add few-shot normal-prototype adaptation.
- Promote ONNX and OpenVINO export scaffolds into parity-tested trained-model export.
- Measure Jetson Orin NX 16GB latency at 512 px.

## Proof Boundary

This repository proves the Phase 0 API scaffold, local MVTec AD `bottle` dataset readiness,
real local Anomalib PaDiM baselines, one CPU classical baseline, reusable PaDiM checkpoint
prediction, Phase 0 ONNX/OpenVINO export mechanics, and trained PaDiM export artifact creation.
It does not prove cross-category benchmark quality, production calibration, Jetson latency,
clean trained-model export parity, TensorRT compatibility, or factory deployment readiness.

Readiness estimate:

- Repository / HF anomaly scaffold: ready to share with strict baseline/deployment-scaffold wording.
- Local MVTec AD `bottle` PaDiM baseline evidence: ready to cite with exact report paths.
- Broader benchmark release across MVTec AD, VisA, AD2, or LOCO: not ready.
- Edge deployment: not ready until trained-model export parity, clean OpenVINO/TensorRT checks,
  and Jetson Orin NX latency are measured.

See `docs/audit.md`, `docs/proof_status.md`, `docs/benchmark_protocol.md`,
`docs/posting.md`, and `docs/release_checklist.md` before posting or pushing a model
checkpoint.
