# Positioning

InspectNet-CX is a Hugging Face-style industrial anomaly detection project focused on
deployment-ready model contracts, not novelty claims around a backbone.

## Moat

- HF-style API for image score, heatmap, mask, threshold, confidence, and defect regions.
- Calibrated reject decisions learned from normal-only data.
- Few-shot normal-prototype adaptation for teams with scarce defect labels.

## Non-Moats

- The CNN backbone choice.
- Beating PatchCore on saturated MVTec AD alone.
- Repackaging Anomalib without a better user-facing model contract.

## Must-Prove Claims

1. Competitive on AD2 and LOCO, not just MVTec AD.
2. Threshold stability under normal-only calibration.
3. Measured latency on mewtwo (AMD Ryzen 9 9900X + NVIDIA RTX 5070): CUDA median 0.474 ms/img (p95 0.622 ms) at 512 px; CPU median 2.956 ms/img (p95 3.217 ms) at 512 px.
4. Future hardware: Jetson Orin NX 16GB (untested; target for a future lab session).
