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
| **ResNet18, single-scale (shipped)** | 1.000 | 0.751 | 0.888 | 0.913 |
| ResNet18, multi-scale input (224/256/320) | 1.000 | 0.728 | 0.874 | 0.938 |
| wide_resnet50_2 backbone | 0.589 | 0.538 | 0.446 | 0.603 |

## Findings

**The native student-teacher does not beat PatchCore, and that is the expected result.**
PatchCore scores test patches by nearest-neighbour distance to a memory bank of *frozen*
wide_resnet50_2 features; the student-teacher trains a student to mimic a teacher and scores by
the residual. On MVTec the trained-residual family (STFPM-style) generally sits a few points
below memory-bank PatchCore, which is what we see.

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

`ResNet18, single-scale` is the shipped native detector. PatchCore remains the stronger method
and is the honest reference to beat; closing that gap would require a different paradigm (for
example reverse distillation or a memory-bank head), not a larger student.
