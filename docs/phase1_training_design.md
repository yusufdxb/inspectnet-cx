# Phase 1 Native Detector Training Design

Sprint 3 replaces the Sprint 2 BCE-against-zero stub with an honest
reconstruction baseline.

## Why the Sprint 2 trainer was degenerate

`src/inspectnet_cx/training/phase1.py` trains the `anomaly_head` sigmoid
output against an all-zero target using `BCELoss`, on normal-only images.
Two problems compound:

1. **No anomaly signal in the loss.** The trainer never sees anomalies. The
   gradient pushes every pixel toward 0 regardless of input. At
   inference time the head therefore outputs near-zero for both normal
   and anomalous images, collapsing AUROC to ~chance.
2. **Saturating sigmoid against a constant target.** As outputs approach
   0, gradients shrink to zero, so even the "make everything 0" objective
   provides vanishing learning signal after a few epochs.

This is not a learned anomaly detector. It is a scaffold.

## Reconstruction baseline (this sprint)

A small symmetric conv autoencoder, trained with per-pixel MSE on
normal-only images. The anomaly score for a test image is the mean
squared reconstruction error.

Rationale for choosing reconstruction over (e.g.) feature-distance:

- **Honest with normal-only data.** Reconstruction loss is well-defined
  on normal-only training data and does not require any anomalous
  examples or pretrained feature extractors. It produces a real,
  observable per-image score at test time.
- **Small enough to train on the RTX 5070 in seconds.** 855,619
  parameters, ~0.2 ms/img CUDA forward at 128 px. 30 epochs on
  ~200 bottle/cable images completes in well under a minute.
- **No new architectural dependencies.** Lives in
  `inspectnet_cx.training.phase1_recon` and does not alter the
  released `InspectNetCXForAnomalyDetection` model surface. The
  v0.1.0 HF release stays unchanged.

The reconstruction AE is intentionally simple. It is **not** state of
the art; SOTA anomaly detection on MVTec AD uses memory banks, normalizing
flows, or pretrained-feature reconstructions. We choose this baseline
because it is the cheapest honest learned model to compare head to head
against PaDiM under the same AUROC + bootstrap-CI methodology. If the
reconstruction baseline cannot beat PaDiM in CI-overlap, the gap
documents that scaling up the native model is required before any native
deployment claim.

## Architecture

- Encoder: 4 stride-2 Conv2d blocks (3 -> 32 -> 64 -> 128 -> 128).
- Decoder: 4 stride-2 ConvTranspose2d blocks back to 3 channels with
  a terminal sigmoid.
- Input/output: RGB float in [0, 1], 128 x 128.
- Parameters: 855,619.

## Training protocol

- Data: MVTec AD `<category>/train/good`.
- Split: 80/20 train/val (deterministic, seed=0).
- Optimizer: Adam, lr=1e-3.
- Loss: MSE on pixel reconstruction.
- Epochs: 30.
- Best checkpoint: lowest val MSE (`best.pt`).
- Hardware: NVIDIA RTX 5070 (CUDA).

## Inference protocol

- Image score: per-image mean squared reconstruction error across all
  channels and pixels at the trained resolution.
- Bootstrap CI: same `scripts/bootstrap_auroc.py` used for PaDiM
  (stratified percentile, n=1000, seed=0).
