# Phase A: Eval Harness Reproduction Report

**Date (UTC):** 2026-05-25
**Repo HEAD at run:** `e8b13f0163d67d14f3e3803fab58a834b002d5ea`
**Harness:** `scripts/eval_harness.py`
**Tests:** `tests/test_eval_harness.py` (10 passed)

## What got built

- `scripts/eval_harness.py`: single-entry CLI that loads an existing Anomalib
  PaDiM/PatchCore checkpoint, runs `engine.predict` on the standard MVTec AD
  test split, and writes a self-contained JSON with image AUROC, pixel AUROC,
  AUPRO, train-derived threshold, dataset hash, checkpoint SHA256, library
  versions, git commit, and UTC timestamp.
- Threshold-leakage guard: `select_threshold_from_train(train_scores, *, quantile)`
  has exactly one positional parameter named `train_scores`. The runtime guard
  rejects 2-tuples (looks like `(scores, labels)`) and dicts whose keys mention
  `test` or `label`, raising `ThresholdLeakageError`.
- Metrics: image AUROC and pixel AUROC use sklearn `roc_auc_score`. AUPRO uses
  anomalib 2.4.1's `_AUPRO` torchmetric (per-region overlap, `fpr_limit=0.3`,
  standard MVTec setting).
- Dataset hash: SHA256 over sorted `(relative_path, file_size_bytes)` for
  `train/`, `test/`, `ground_truth/` under the category root.
- Checkpoint hash: SHA256 of the `.ckpt` file bytes.

## Checkpoint inventory

All four PaDiM checkpoints already existed under `artifacts/`. No fits were
re-run in this phase.

| category | checkpoint path | SHA256 (prefix) |
|----------|-----------------|-----------------|
| bottle   | `artifacts/anomalib_scoring/Padim/MVTecAD/bottle/v2/weights/lightning/model.ckpt` | `de7ea7c54f...` |
| cable    | `artifacts/anomalib_scoring_cable/Padim/MVTecAD/cable/v1/weights/lightning/model.ckpt` | `2db928d249...` |
| capsule  | `artifacts/anomalib_scoring_capsule/Padim/MVTecAD/capsule/v0/weights/lightning/model.ckpt` | `0fcba06e0c...` |
| leather  | `artifacts/anomalib_scoring_leather/Padim/MVTecAD/leather/v0/weights/lightning/model.ckpt` | `986cf8871a...` |

## Reproduction table (image-level AUROC)

Reference numbers are from `docs/threshold_analysis_padim_multi_category.md`.
Tolerance: ±0.005.

| category | n_test | reference image AUROC | measured image AUROC | delta   | pass |
|----------|-------:|----------------------:|---------------------:|--------:|:----:|
| bottle   |     83 | 0.9976                | 0.9976               | 0.0000  | YES  |
| cable    |    150 | 0.8720                | 0.8720               | 0.0000  | YES  |
| capsule  |    132 | 0.8807                | 0.8807               | 0.0000  | YES  |
| leather  |    124 | 0.9925                | 0.9925               | 0.0000  | YES  |

All four land at the 4th decimal of the canonical numbers. The bottle reference
0.9976 (and not the 0.9960 artifact_index value) is what reproduces against the
v2 bottle checkpoint with the existing scoring path; the 0.9960 figure
corresponds to a different prior bottle artifact and is not reproduced here.

## Newly measured pixel-level numbers (no published reference)

These are first-time measurements from this harness, no reference to compare to.

| category | pixel AUROC | AUPRO (fpr_limit=0.3) | threshold (train q=0.995) |
|----------|------------:|----------------------:|--------------------------:|
| bottle   | 0.9816      | 0.9406                | 0.3784                    |
| cable    | 0.9551      | 0.8519                | 0.3756                    |
| capsule  | 0.9849      | 0.9149                | 0.4080                    |
| leather  | 0.9882      | 0.9682                | 0.4027                    |

Threshold is the 99.5th percentile of train-set per-image anomaly scores
predicted by the fitted PaDiM model. Test labels never enter the
threshold-selection path (enforced structurally by the function signature and
by `_threshold_guard`).

## Library versions used

```
python       3.10.12
platform     Linux-6.8.0-117-generic-x86_64-with-glibc2.35
numpy        2.2.6
sklearn      1.7.2
torch        2.11.0+cu128
torchvision  0.26.0+cu128
anomalib     2.4.1
openvino     2026.1.0  (installed but not loaded by the harness)
```

## Pytest output

```
$ python3 -m pytest tests/test_eval_harness.py -q
..........                                                               [100%]
10 passed, 6 warnings in 2.38s
```

Coverage:
- 4 tests for the threshold leakage guard (tuple, dict, signature, happy path)
- 2 tests for image AUROC against hand-computed values (perfect and 0.75 case)
- 1 test for pixel AUROC against a hand-computed flattened example
- 2 tests for AUPRO (perfect segmentation = 1.0, random in [0,1])
- 2 tests for hashing (`file_sha256` stability, dataset hash sensitivity)

## Result JSONs

- `reports/eval_harness/padim_bottle_repro.json`
- `reports/eval_harness/padim_cable_repro.json`
- `reports/eval_harness/padim_capsule_repro.json`
- `reports/eval_harness/padim_leather_repro.json`

Each contains: category, image_auroc, pixel_auroc, aupro (separate fields),
threshold + selection rule + split it was selected on, split_hash,
checkpoint_hash, library_versions, git_commit, timestamp_utc.

## Deviations from the prompt

None of substance. Notes:
- No fits were re-run because all four PaDiM checkpoints already existed on
  disk. The 30 min/category time-box was not approached.
- AUPRO is computed via anomalib's bundled `_AUPRO` (per-region, fpr_limit=0.3)
  rather than a hand-rolled PRO-curve integration, as the harness verifies the
  same library that produced the canonical numbers.
- Threshold-leakage guard combines two layers: (1) function signature has only
  one positional parameter `train_scores`, structurally preventing test labels
  from being passed; (2) runtime `_threshold_guard` rejects suspicious shapes
  (2-tuples and dicts with test/label keys). Both are tested.

## Riskiest unverified assumption

The published reference numbers in `docs/threshold_analysis_padim_multi_category.md`
were themselves produced by `scripts/score_anomalib_test.py` (the existing
fit+predict path). The harness reuses the same PaDiM (`backbone=resnet18`,
`layers=layer1/2/3`) configuration and the same Anomalib version, so a match
is essentially a tautology check, not an independent validation of the
underlying PaDiM correctness. Independent validation (e.g., against MVTec
AD's published PaDiM table or a fresh fit on a different machine) is not part
of Phase A and would be needed before claiming the numbers are correct in an
absolute sense.

Pixel AUROC and AUPRO have NO published reference in this repo; they are
reported here as new measurements and should be sanity-checked against the
MVTec AD literature in Phase B before being used as comparison targets.

---

**STOPPED after Phase A reproduction check. Awaiting approval to proceed to Phase B (PatchCore training).**
