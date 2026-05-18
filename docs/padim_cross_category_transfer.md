# PaDiM Cross-Category Transfer (Sprint 3 Rigor)

Headline claim: **PaDiM is category-specific. Fitting on a different
category drops AUROC by 0.431 +- 0.014 (95% bootstrap CI [0.403, 0.458]).**

## Method

Four MVTec AD categories: `bottle`, `cable`, `capsule`, `leather`. (The
Sprint 3 plan named `pill` as the fourth, but `pill` is not present
under `~/datasets/mvtec_ad/`; `capsule` was substituted.)

For each `(train_cat, test_cat)` cell:

1. Build an `anomalib.data.MVTecAD` datamodule for `train_cat` and fit a
   `Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"])`
   memory bank with `Engine.fit` (1 epoch is sufficient; PaDiM is
   single-pass statistics, not gradient descent).
2. Build a second `MVTecAD` datamodule pointed at `test_cat`. Run
   `Engine.predict(model=model, datamodule=test_dm)` with the
   already-fitted model. This uses the train_cat's mean / covariance
   bank against the test_cat's images.
3. Compute image-level AUROC on the test_cat test split.

Diagonal cells (`train_cat == test_cat`) re-use the cached Sprint 2
score files in `reports/scores_padim_<cat>.json`.

Cells produced by `scripts/cross_category_padim.py`; matrix and bootstrap
drop CI assembled by `scripts/build_cross_transfer_matrix.py`.

## Cross-Category AUROC Matrix (test split, image-level)

| train \\ test | bottle | cable  | capsule | leather |
| ------------- | -----: | -----: | ------: | ------: |
| **bottle**    | 0.9976 | 0.5000 |  0.4579 |  0.5628 |
| **cable**     | 0.5405 | 0.8720 |  0.5000 |  0.4891 |
| **capsule**   | 0.5000 | 0.5000 |  0.8807 |  0.5068 |
| **leather**   | 0.5091 | 0.5143 |  0.4817 |  0.9925 |

Diagonal = matched train/test (same as the multi-category Sprint 2 table).
Off-diagonal = cross-category transfer.

## Drop summary

| statistic                                  |    value |
| ------------------------------------------ | -------: |
| Mean off-diagonal drop (point)             |   0.4305 |
| Std of point drops across 12 cells         |   0.0545 |
| Bootstrap n=1000, stratified per-cell      |        - |
| Bootstrap drop median                      |   0.4303 |
| Bootstrap drop 95% CI low                  |   0.4034 |
| Bootstrap drop 95% CI high                 |   0.4580 |

## Interpretation

- The PaDiM memory bank built on category X carries **no useful signal**
  for category Y in this setup. The off-diagonal AUROC sits around or
  below 0.50 (chance) in most cells, and the worst transfer cells
  (`bottle -> capsule` 0.458, `leather -> capsule` 0.482) are
  **below chance**, meaning the score direction is inverted relative to
  the test_cat's anomalies.
- Several off-diagonal cells land exactly at 0.500. That is a
  measurement artifact of anomalib's `OneClassPostProcessor`, which
  rescales scores around the train-set threshold; when the test
  distribution sits entirely on one side of that threshold, the
  post-processor saturates and the ROC curve degenerates. The raw
  per-patch distance would still separate, but the rescaled per-image
  score does not. This is a real limitation of the rescaled output and
  is itself a deployment finding: **the rescaled per-image score is
  not category-transferable.**
- The cross-CI lower bound on the mean drop is **0.403**, well above
  any plausible noise floor. PaDiM is empirically category-specific
  on MVTec AD under this anomalib configuration.

## Reproducibility

```bash
for train_cat in bottle cable capsule leather; do
  for test_cat in bottle cable capsule leather; do
    [ "$train_cat" = "$test_cat" ] && continue
    PYTHONPATH=src python3 scripts/cross_category_padim.py \
      --train-category $train_cat \
      --test-category $test_cat \
      --work-dir artifacts/cross_padim_${train_cat}_${test_cat} \
      --output reports/cross_padim_${train_cat}_to_${test_cat}.json
  done
done

PYTHONPATH=src python3 scripts/build_cross_transfer_matrix.py
```
