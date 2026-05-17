# PaDiM MVTec AD Bottle: Operating-Point Analysis

This document records the threshold-analysis result for the published Anomalib
PaDiM baseline on MVTec AD `bottle`. The numbers below are produced by

```bash
PYTHONPATH=src python scripts/score_anomalib_test.py \
  --category bottle \
  --output reports/scores_padim_bottle.json
PYTHONPATH=src python scripts/threshold_analysis.py \
  --input reports/scores_padim_bottle.json \
  --output reports/threshold_analysis_padim_bottle.json
```

The score file is per-image, with one row per test image, labels of 0 (normal)
or 1 (anomalous), and the PaDiM image-level anomaly score. The analysis script
emits ROC, AUROC, and operating-point recommendations.

`reports/threshold_analysis_padim_bottle.json` is regenerable and is therefore
not checked into git. The summary below is the canonical reference; the
generator command is reproducible from a fresh venv as documented in the
release audit (`docs/public_release_audit.md`).

## Test Split

- `n_samples`: 83
- `n_positive` (anomaly): 63
- `n_negative` (normal): 20
- score range: 0.381 to 1.000

## Headline

- **AUROC**: 0.9976. Consistent with the published image-AUROC of 0.9960 in
  `reports/anomalib_padim_mvtec_ad_bottle_result.json`. The small delta comes
  from the fact that AUROC computed from `engine.predict` per-image scores can
  differ from the metric Anomalib reports through `engine.test` when image
  scores are rescaled by its post-processor.

## Recommended Operating Points

| operating point | threshold | TPR    | FPR    | precision | recall | F1     | TP | FP | TN | FN |
| --------------- | --------: | -----: | -----: | --------: | -----: | -----: | -: | -: | -: | -: |
| Youden (max TPR-FPR) |  0.5502 | 0.9524 | 0.0000 |    1.0000 | 0.9524 | 0.9756 | 60 |  0 | 20 |  3 |
| F1-max               |  0.5000 | 1.0000 | 0.0500 |    0.9844 | 1.0000 | 0.9921 | 63 |  1 | 19 |  0 |
| FPR <= 0.01          |  0.5502 | 0.9524 | 0.0000 |    1.0000 | 0.9524 | 0.9756 | 60 |  0 | 20 |  3 |
| FPR <= 0.05          |  0.5000 | 1.0000 | 0.0500 |    0.9844 | 1.0000 | 0.9921 | 63 |  1 | 19 |  0 |
| FPR <= 0.10          |  0.5000 | 1.0000 | 0.0500 |    0.9844 | 1.0000 | 0.9921 | 63 |  1 | 19 |  0 |

## How to Read These

- The Youden point is the natural detector-side threshold: it maximizes the
  margin between true positives and false positives. At 0.5502 the model
  catches 60 of 63 anomalies with zero false alarms on the 20 normal test
  images. Three anomaly images slip through.
- The F1-maximizing point is the natural operator-side threshold when the
  goal is balanced precision and recall. At 0.5000 the model catches all 63
  anomalies at the cost of one false alarm.
- The FPR-ceiling points show what the model can offer if the operator
  publishes a tolerated false-alarm rate. At a 1% FPR ceiling, recall is
  0.9524. At 5% it is 1.0. The bottle category is small enough that
  FPR <= 0.05 and FPR <= 0.10 collapse to the same point.

## Honest Bounds

- Twenty normal test images is a small denominator. A single false positive
  moves the empirical FPR by 0.05. Bottle-category thresholds should not be
  generalized to other MVTec categories or to production inspection without
  re-running this analysis on the relevant category and recalculating
  operating points from a larger labeled set.
- These numbers are **per-test-split**, not per-deployment. Deployment
  thresholds require a separate held-out calibration split, the operator's
  tolerated FPR, and the prior class balance in the deployment line, which
  almost certainly is not 63 anomaly : 20 normal.
- The PaDiM score distribution is rescaled by Anomalib's post-processor.
  Thresholds reported here are in the rescaled score space; raw PaDiM
  patch-distance scores produce a different distribution and a different
  operating-point table.
