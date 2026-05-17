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
3. Less than 30 ms per image on Jetson Orin NX 16GB at 512 px.
