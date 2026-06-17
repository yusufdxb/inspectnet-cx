# Native Detector Ablations (student-teacher)

Image-level AUROC on four MVTec AD categories, matched train/test. PaDiM/PatchCore are the
reference baselines; the rest are variants of the native student-teacher detector
(`src/inspectnet_cx/models/student_teacher.py`). Evidence JSONs:
`reports/eval_harness/inspectnet_st_*.json` (single-scale),
`inspectnet_st_ms_*.json` (multi-scale), `inspectnet_st_wide_*.json` (wide backbone).

| variant | bottle | cable | capsule | leather |
| ------- | -----: | ----: | ------: | ------: |
| PatchCore (reference) | 1.000 | 0.991 | 0.994 | 1.000 |
| PaDiM (reference)     | 0.998 | 0.872 | 0.881 | 0.993 |
| **Reverse distillation (shipped)** | 1.000 | 0.885 | 0.901 | 1.000 |
| student-teacher, ResNet18 single-scale | 1.000 | 0.751 | 0.888 | 0.913 |
| student-teacher, multi-scale input (224/256/320) | 1.000 | 0.728 | 0.874 | 0.938 |
| student-teacher, wide_resnet50_2 backbone | 0.589 | 0.538 | 0.446 | 0.603 |

Evidence: `reports/eval_harness/inspectnet_rd_*.json` (reverse distillation),
`inspectnet_st_*.json` / `inspectnet_st_ms_*.json` / `inspectnet_st_wide_*.json` (student-teacher
variants).

## Findings

**Reverse distillation closes most of the gap and is the shipped detector.** A *frozen*
wide_resnet50_2 teacher feeds a bottleneck; a trainable decoder reconstructs the teacher's
layer1-3 features, and per-pixel cosine distance is the anomaly score. Because the teacher is
frozen and the decoder only sees a compressed bottleneck, it cannot trivially copy anomalies,
the exact failure that sank the wide student-teacher below. Result: ties PatchCore on `bottle`
and `leather` (both 1.000), beats PaDiM on all four categories, and lifts the worst case from
0.751 (student-teacher `cable`) to 0.885. It still trails PatchCore on `cable` (0.991) and
`capsule` (0.994), so it does not beat PatchCore overall.

**The vanilla student-teacher does not beat PatchCore, as expected.**
PatchCore scores test patches by nearest-neighbour distance to a memory bank of *frozen*
wide_resnet50_2 features; the student-teacher trains a same-direction student and scores by the
residual. On MVTec the trained-residual family (STFPM-style) sits a few points below memory-bank
PatchCore, which is what we see.

**Multi-scale input fusion is roughly neutral.** Fusing anomaly maps over input resolutions
224/256/320 helped `leather` (0.913 -> 0.938) but slightly hurt `cable` and `capsule`. The
layer1-3 feature pyramid already supplies most of the multi-scale benefit; extra input scales add
little here.

**A bigger backbone makes it much worse, not better.** Swapping in wide_resnet50_2 (PatchCore's
backbone) collapsed every category toward chance. A from-scratch wide student has so much
capacity that on ~200 normal images it learns to reproduce the teacher's features almost
everywhere, including on defects, so the residual signal disappears (train loss fell to ~2e-4).
This is the known student-teacher capacity trap: PatchCore can use wide_resnet50_2 because it
keeps it *frozen* as a feature extractor; a *trained* student needs limited capacity to preserve
the normal-vs-anomaly gap. The lighter ResNet18 student is the right choice.

## Conclusion

Reverse distillation is the shipped native detector. The journey confirmed the hypothesis that a
*paradigm* change, not a bigger student, was needed: scaling student-teacher capacity made it
worse, while the frozen-teacher / bottleneck / decoder design ties PatchCore on half the
categories and beats PaDiM everywhere. PatchCore still leads on `cable`/`capsule`; fully matching
it is open work (a faithful one-class bottleneck, longer training, or a memory-bank head). All
numbers are single-seed; the large effects (wide collapse, reverse-distillation lift) are far
beyond plausible seed noise, but a publication-grade claim would report 3 seeds with error bars.
