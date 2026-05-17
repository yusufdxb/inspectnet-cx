# PaDiM MVTec AD Multi-Category Threshold Analysis

This document aggregates threshold-analysis results for PaDiM (ResNet-18 backbone,
layers layer1/layer2/layer3) across four MVTec AD categories: bottle, cable,
capsule, and leather.

Each row is produced by:

```bash
PYTHONPATH=src python3 scripts/score_anomalib_test.py \
  --category $CAT \
  --output reports/scores_padim_$CAT.json \
  --work-dir artifacts/anomalib_scoring_$CAT

PYTHONPATH=src python3 scripts/threshold_analysis.py \
  --input reports/scores_padim_$CAT.json \
  --output reports/threshold_analysis_padim_$CAT.json
```

Score files and analysis JSONs live under `reports/` (gitignored; regenerable).
The summary below is the canonical reference.

## Per-Category Summary

| category | n_samples | n_pos | n_neg | AUROC  | Youden thr | Youden TPR | Youden FPR | Youden F1 | F1-max thr | F1-max TPR | F1-max FPR | F1-max F1 |
| -------- | --------: | ----: | ----: | -----: | ---------: | ---------: | ---------: | --------: | ---------: | ---------: | ---------: | --------: |
| bottle   |        83 |    63 |    20 | 0.9976 |     0.5502 |     0.9524 |     0.0000 |    0.9756 |     0.5000 |     1.0000 |     0.0500 |    0.9921 |
| cable    |       150 |    92 |    58 | 0.8720 |     0.5000 |     0.9022 |     0.3103 |    0.8601 |     0.5000 |     0.9022 |     0.3103 |    0.8601 |
| capsule  |       132 |   109 |    23 | 0.8807 |     0.5215 |     0.9266 |     0.1739 |    0.9439 |     0.5000 |     0.9725 |     0.3478 |    0.9507 |
| leather  |       124 |    92 |    32 | 0.9925 |     0.5097 |     0.9674 |     0.0000 |    0.9834 |     0.5000 |     0.9783 |     0.0313 |    0.9836 |

## Bootstrap 95% Confidence Intervals (Sprint 3)

Percentile bootstrap with stratified resampling (preserves per-bootstrap class
balance), n=1000 resamples, seed=0. Produced by
`scripts/bootstrap_auroc.py` from the same cached score files.

| category | AUROC point | AUROC CI low | AUROC CI high | AUROC median | AUROC std | Youden-F1 CI       | F1-max-F1 CI       |
| -------- | ----------: | -----------: | ------------: | -----------: | --------: | :----------------- | :----------------- |
| bottle   |      0.9976 |       0.9905 |        1.0000 |       0.9984 |    0.0031 | [0.9587, 1.0000]   | [0.9767, 1.0000]   |
| cable    |      0.8720 |       0.8156 |        0.9264 |       0.8749 |    0.0288 | [0.7133, 0.9072]   | [0.8177, 0.9082]   |
| capsule  |      0.8807 |       0.7786 |        0.9713 |       0.8855 |    0.0499 | [0.9135, 0.9770]   | [0.9375, 0.9813]   |
| leather  |      0.9925 |       0.9769 |        1.0000 |       0.9932 |    0.0064 | [0.9663, 1.0000]   | [0.9721, 1.0000]   |

Headline rigor read:

- **Cable AUROC CI width is 0.111** (0.8156 to 0.9264). The point estimate
  0.872 hides a substantial uncertainty band; the lower bound 0.816 is still
  well above chance but not deployment-strong.
- **Capsule AUROC CI is even wider at 0.193** (0.7786 to 0.9713). With only
  23 normal test images, a single sample swing moves AUROC by enough to
  cross the 0.80 line. Capsule's lower bound 0.779 is below 0.80, so a
  CI-aware reading of capsule says PaDiM is **not** demonstrably above 0.80
  AUROC on this split.
- **Bottle and leather** have CIs whose lower bound is above 0.97, so the
  high point estimates are robust to resampling noise.

## Key Observations

- Leather and bottle are strong fits for this PaDiM variant (AUROC 0.99+). Both
  achieve near-zero FPR at the Youden operating point.
- Cable and capsule are harder categories for ResNet-18 PaDiM (AUROC 0.87-0.88).
  The Youden point for cable collides with the F1-max point (both at threshold
  0.500, FPR 0.31), indicating the ROC curve is relatively flat in that region
  and the default post-processor rescaling compresses most scores toward 0.5.
- Capsule Youden and F1-max diverge: Youden at 0.5215 achieves FPR 0.17 / TPR
  0.93 while F1-max at 0.5000 increases recall to 0.97 at the cost of FPR 0.35.
  This trade-off is category-specific and cannot be generalized.
- Thresholds are **not portable across categories**. Cable at threshold 0.5000
  carries 31% FPR; the same threshold on leather or bottle carries near-zero FPR.

## Honest Bounds

- Scores are from `engine.predict` in the rescaled post-processor space.
  Raw PaDiM patch-distance scores produce a different distribution.
- These are per-test-split results, not deployment-calibrated thresholds.
  Deployment requires a separate calibration set, operator-specified FPR ceiling,
  and the prior class balance of the actual inspection line.
- Cable and capsule normal sample counts (58 and 23 respectively) are small
  denominators. A single false positive on capsule moves empirical FPR by ~0.04.
- No pixel-level AUROC is reported here; this analysis is image-level only.

## Category Operating-Point Detail

### bottle (reference; see also `threshold_analysis_padim_bottle.md`)

| operating point    | threshold | TPR    | FPR    | F1     |
| ------------------ | --------: | -----: | -----: | -----: |
| Youden (max TPR-FPR) | 0.5502  | 0.9524 | 0.0000 | 0.9756 |
| F1-max             |    0.5000 | 1.0000 | 0.0500 | 0.9921 |
| FPR <= 0.01        |    0.5502 | 0.9524 | 0.0000 | 0.9756 |
| FPR <= 0.05        |    0.5000 | 1.0000 | 0.0500 | 0.9921 |

### cable

| operating point    | threshold | TPR    | FPR    | F1     |
| ------------------ | --------: | -----: | -----: | -----: |
| Youden (max TPR-FPR) | 0.5000  | 0.9022 | 0.3103 | 0.8601 |
| F1-max             |    0.5000 | 0.9022 | 0.3103 | 0.8601 |
| FPR <= 0.01        |    0.6996 | 0.3804 | 0.0000 | 0.5512 |
| FPR <= 0.05        |    0.6214 | 0.5870 | 0.0345 | 0.7297 |
| FPR <= 0.10        |    0.5841 | 0.6413 | 0.0862 | 0.7564 |

### capsule

| operating point    | threshold | TPR    | FPR    | F1     |
| ------------------ | --------: | -----: | -----: | -----: |
| Youden (max TPR-FPR) | 0.5215  | 0.9266 | 0.1739 | 0.9439 |
| F1-max             |    0.5000 | 0.9725 | 0.3478 | 0.9507 |
| FPR <= 0.01        |    0.8605 | 0.1284 | 0.0000 | 0.2276 |
| FPR <= 0.05        |    0.8386 | 0.1560 | 0.0435 | 0.2677 |
| FPR <= 0.10        |    0.6121 | 0.5229 | 0.0870 | 0.6786 |

### leather

| operating point    | threshold | TPR    | FPR    | F1     |
| ------------------ | --------: | -----: | -----: | -----: |
| Youden (max TPR-FPR) | 0.5097  | 0.9674 | 0.0000 | 0.9834 |
| F1-max             |    0.5000 | 0.9783 | 0.0313 | 0.9836 |
| FPR <= 0.01        |    0.5097 | 0.9674 | 0.0000 | 0.9834 |
| FPR <= 0.05        |    0.5000 | 0.9783 | 0.0313 | 0.9836 |
| FPR <= 0.10        |    0.4855 | 0.9891 | 0.0938 | 0.9785 |
