# Benchmark Protocol

This protocol is required before any benchmark table is posted.

## Datasets

- MVTec AD
- VisA
- MVTec AD2
- MVTec LOCO AD

Datasets must live outside the repository. Use:

```bash
inspectnet-dataset-check --root ~/datasets --output reports/dataset_check.json
```

Expected local structures include:

```text
~/datasets/mvtec_ad/<category>/train/good
~/datasets/mvtec_ad/<category>/test
~/datasets/mvtec_ad/<category>/ground_truth
~/datasets/visa/visa_pytorch/<category>/train/good
```

For MVTec AD and VisA, Anomalib can also prepare/download datasets when configured to do so.
Record whether data came from Anomalib preparation or a manual official download.

## Baselines

Minimum baseline set:

- PatchCore
- EfficientAD
- PaDiM
- SimpleNet

Use Anomalib implementations directly. Do not reimplement baselines unless a paper-specific
difference is being studied.

Generate a command/result scaffold before running a real baseline:

```bash
inspectnet-baseline --plan-only --method patchcore --dataset mvtec_ad --category bottle --data-root ~/datasets
inspectnet-baseline --plan-only --method patchcore --dataset visa --category candle --data-root ~/datasets
```

The emitted `anomalib_command` follows the current Anomalib CLI shape, but must be validated
against the exact installed Anomalib version before benchmark publication.

## Metrics

Required metrics:

- image AUROC
- pixel AUROC
- AU-PRO
- pixel F1
- threshold source
- latency in ms per image
- peak VRAM in MB
- model size in MB

All threshold-dependent metrics must state whether the threshold was calibrated from normal
data only, validation anomalies, or test labels.

Normal-only threshold scaffold:

```bash
inspectnet-calibrate-normal-threshold --model artifacts/inspectnet-cx-phase0 --dataset mvtec_ad --category bottle --dataset-root ~/datasets
```

This command is acceptable as proof of calibration mechanics only. It is not evidence of
calibration quality unless paired with held-out labeled evaluation.

## Reporting Rules

- No fabricated numbers.
- `TBD` is allowed only in Phase 0 placeholder files.
- Every posted table must include exact commands, hardware, package versions, dataset root,
  and commit hash.
- AD2 and LOCO results must be separated from saturated MVTec AD results.
- ONNX/OpenVINO export claims require an export report and numerical parity check.
- Workstation latency claims must be produced on mewtwo (AMD Ryzen 9 9900X + RTX 5070) or an equivalent x86_64 host with CUDA; cite device, image size, and the output JSON.
- Jetson Orin NX 16GB latency claims require `--require-jetson` output produced on that hardware (future hardware; not yet measured).
