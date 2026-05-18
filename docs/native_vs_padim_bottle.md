# Native ReconAE vs PaDiM Head-to-Head

Measured on MVTec AD `bottle` and `cable` test splits with the same
`scripts/bootstrap_auroc.py` methodology used for the PaDiM CI table.

## Bottle

| model       | params    | CUDA median ms/img (128 px) | AUROC point | AUROC 95% CI       | Youden-F1 CI       | F1-max-F1 CI       |
| ----------- | --------: | --------------------------: | ----------: | :----------------- | :----------------- | :----------------- |
| PaDiM (R18) | ~2.8e6 *  | n/a (CPU eval, anomalib)    |      0.9976 | [0.9905, 1.0000]   | [0.9587, 1.0000]   | [0.9767, 1.0000]   |
| ReconAE     |   855,619 |                       0.188 |      0.8238 | [0.7301, 0.9095]   | [0.6596, 0.8966]   | [0.8696, 0.9197]   |

\* PaDiM uses the frozen ResNet-18 backbone (~11.7M total, ~2.8M params
in the layer1/layer2/layer3 path being aggregated). The PaDiM "model"
itself is a memory bank of Gaussian statistics, not a learned network.

## Cable

| model       | params    | CUDA median ms/img (128 px) | AUROC point | AUROC 95% CI       | Youden-F1 CI       | F1-max-F1 CI       |
| ----------- | --------: | --------------------------: | ----------: | :----------------- | :----------------- | :----------------- |
| PaDiM (R18) | ~2.8e6 *  | n/a (CPU eval, anomalib)    |      0.8720 | [0.8156, 0.9264]   | [0.7133, 0.9072]   | [0.8177, 0.9082]   |
| ReconAE     |   855,619 |                       0.188 |      0.6966 | [0.6019, 0.7811]   | [0.5079, 0.8087]   | [0.7603, 0.8108]   |

## Headline Verdict (HONEST NEGATIVE RESULT)

**The ReconAE native baseline is clearly below PaDiM on both bottle and cable.**

- **Bottle.** Native CI [0.730, 0.910] is fully below PaDiM CI
  [0.991, 1.000]; the two intervals do not overlap. Native is not
  competitive with PaDiM on bottle.
- **Native bottle CI lower bound 0.730 is below 0.85**, so per the
  Sprint 3 acceptance criterion this is a documented negative result.
  The reconstruction baseline is **not** production-ready on bottle.
- **Cable.** Native CI [0.602, 0.781] is fully below PaDiM CI
  [0.816, 0.926]; again no overlap. Native is not competitive on cable
  either.

This is the honest Sprint 3 finding: a small reconstruction autoencoder
trained from scratch on ~200 normal images cannot match PaDiM's
ImageNet-pretrained feature embeddings as a category-specific anomaly
detector at this scale. Scaling the native model (deeper encoder,
pretrained feature reconstruction, perceptual loss, larger images) is
the obvious next step before any native deployment claim. The scaffold
did **not** become production.

## Reproducibility

```bash
PYTHONPATH=src python3 scripts/train_phase1_recon.py \
  --train-data-dir ~/datasets/mvtec_ad/bottle/train/good \
  --output-dir artifacts/phase1_recon_bottle \
  --epochs 30 --batch-size 16 --learning-rate 1e-3 \
  --image-size 128 --device cuda --seed 0

PYTHONPATH=src python3 scripts/eval_phase1_recon.py \
  --checkpoint artifacts/phase1_recon_bottle/best.pt \
  --test-root ~/datasets/mvtec_ad/bottle/test \
  --category bottle \
  --output reports/scores_phase1_recon_bottle.json --device cuda

PYTHONPATH=src python3 scripts/bootstrap_auroc.py \
  --input reports/scores_phase1_recon_bottle.json \
  --output reports/bootstrap_phase1_recon_bottle.json \
  --n-bootstrap 1000 --seed 0
```

Same commands with `cable` in place of `bottle`.
