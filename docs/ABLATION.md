# Phase B Ablation: PaDiM vs PatchCore on MVTec AD (bottle, cable, capsule, leather)

All numbers below come from the Phase A frozen eval harness
(`scripts/eval_harness.py`). PaDiM rows are reproduced from
`reports/eval_harness/padim_<cat>_repro.json` (Phase A, CPU). PatchCore rows
are produced by `scripts/train_patchcore.py` + the same harness with
`--method patchcore`, run on an NVIDIA (Blackwell) consumer GPU.

Backbone: WideResNet-50 (`wide_resnet50_2`), layers `(layer2, layer3)`,
anomalib 2.4.1 default for PatchCore. Single seed. AUPRO uses anomalib's
`_AUPRO` at `fpr_limit=0.3`.

## 1. Headline ablation

| category | method     | coreset | image AUROC | pixel AUROC | AUPRO  | image Δ vs PaDiM | pixel Δ vs PaDiM | AUPRO Δ vs PaDiM |
|----------|------------|---------|-------------|-------------|--------|------------------|-------------------|-------------------|
| bottle   | PaDiM      | -       | 0.9976      | 0.9816      | 0.9406 | -                | -                 | -                 |
| bottle   | PatchCore  | 0.01    | 1.0000      | 0.9851      | 0.9413 | +0.0024          | +0.0035           | +0.0007           |
| cable    | PaDiM      | -       | 0.8720      | 0.9551      | 0.8519 | -                | -                 | -                 |
| cable    | PatchCore  | 0.01    | 0.9910      | 0.9834      | 0.9281 | +0.1190          | +0.0283           | +0.0762           |
| capsule  | PaDiM      | -       | 0.8807      | 0.9849      | 0.9149 | -                | -                 | -                 |
| capsule  | PatchCore  | 0.01    | 0.9944      | 0.9902      | 0.9382 | +0.1137          | +0.0053           | +0.0233           |
| leather  | PaDiM      | -       | 0.9925      | 0.9882      | 0.9682 | -                | -                 | -                 |
| leather  | PatchCore  | 0.01    | 1.0000      | 0.9924      | 0.9760 | +0.0075          | +0.0042           | +0.0078           |

PatchCore beats PaDiM on every metric and every category at coreset=0.01.
The biggest gains land where PaDiM was weak: cable image AUROC (+0.119) and
capsule image AUROC (+0.114). On bottle and leather (where PaDiM was already
saturated) the gains are small but still positive on all three metrics.

## 2. Cable coreset sensitivity

PatchCore, cable category, three coreset subsampling ratios:

| coreset ratio | image AUROC | pixel AUROC | AUPRO  |
|---------------|-------------|-------------|--------|
| 0.01          | 0.9910      | 0.9834      | 0.9281 |
| 0.10          | 0.9856      | 0.9848      | 0.9304 |
| 0.25          | 0.9893      | 0.9844      | 0.9280 |

The three operating points are within a 0.005 band on every metric. Image
AUROC is non-monotone (0.01 > 0.25 > 0.10), pixel AUROC creeps up slightly
with coreset size, and AUPRO is essentially flat. For cable, the paper's
1% coreset default is not worse than the larger 10% and 25% memory banks,
which would have cost 2.1x and 4.0x more storage (108 MB vs 230 MB vs 431 MB
on disk). The sensitivity is dominated by sampling noise, not a clean trend.

## 3. Latency (PatchCore @ 0.01, batch size 1, 256x256)

Measured by `scripts/bench_latency.py` on the dev workstation
(AMD Ryzen 9 9900X 12-core, NVIDIA Blackwell consumer GPU,
driver 570.211.01). 50 timed iterations after 10 warmup iterations, batch size 1,
`torch.cuda.synchronize()` around GPU timing, `time.perf_counter`.

| category | device | min ms | median ms | p95 ms | mean ms | std ms |
|----------|--------|--------|-----------|--------|---------|--------|
| bottle   | cpu    | 28.32  | 30.16     | 31.57  | 30.18   | 1.01   |
| bottle   | cuda   | 5.14   | 5.20      | 6.41   | 5.51    | 0.47   |
| cable    | cpu    | 30.42  | 31.30     | 32.91  | 31.46   | 0.81   |
| cable    | cuda   | 5.13   | 5.29      | 6.56   | 5.51    | 0.47   |
| capsule  | cpu    | 30.71  | 31.78     | 42.23  | 38.63   | 38.97  |
| capsule  | cuda   | 5.11   | 5.35      | 6.18   | 5.47    | 0.38   |
| leather  | cpu    | 30.97  | 32.81     | 35.19  | 32.84   | 1.18   |
| leather  | cuda   | 5.17   | 5.35      | 6.36   | 5.61    | 0.47   |

The capsule CPU mean and std are dominated by a single outlier iteration; the
median (31.78 ms) is in line with the other categories. GPU latency is ~6x
faster than CPU and effectively category-independent.

## 4. Reproduce

Train then evaluate for each row. Replace `<cat>` with one of
`bottle | cable | capsule | leather`.

```bash
# PatchCore @0.01 main run (per category)
python scripts/train_patchcore.py \
  --category <cat> \
  --dataset-root ~/datasets/mvtec_ad \
  --backbone wide_resnet50_2 \
  --coreset-ratio 0.01 \
  --output-dir artifacts/patchcore_<cat> \
  --device cuda

python scripts/eval_harness.py \
  --method patchcore \
  --checkpoint artifacts/patchcore_<cat>/Patchcore/MVTecAD/<cat>/v0/weights/lightning/model.ckpt \
  --category <cat> \
  --dataset-root ~/datasets/mvtec_ad \
  --output reports/eval_harness/patchcore_<cat>.json \
  --device cuda

# Cable coreset sensitivity
python scripts/train_patchcore.py --category cable --coreset-ratio 0.10 \
  --output-dir artifacts/patchcore_cable_coreset010 --device cuda
python scripts/eval_harness.py --method patchcore \
  --checkpoint artifacts/patchcore_cable_coreset010/Patchcore/MVTecAD/cable/v0/weights/lightning/model.ckpt \
  --category cable --dataset-root ~/datasets/mvtec_ad \
  --output reports/eval_harness/patchcore_cable_coreset010.json --device cuda

python scripts/train_patchcore.py --category cable --coreset-ratio 0.25 \
  --output-dir artifacts/patchcore_cable_coreset025 --device cuda
python scripts/eval_harness.py --method patchcore \
  --checkpoint artifacts/patchcore_cable_coreset025/Patchcore/MVTecAD/cable/v0/weights/lightning/model.ckpt \
  --category cable --dataset-root ~/datasets/mvtec_ad \
  --output reports/eval_harness/patchcore_cable_coreset025.json --device cuda

# Latency (per category)
python scripts/bench_latency.py \
  --category <cat> \
  --checkpoint artifacts/patchcore_<cat>/Patchcore/MVTecAD/<cat>/v0/weights/lightning/model.ckpt \
  --dataset-root ~/datasets/mvtec_ad \
  --output reports/eval_harness/patchcore_<cat>_latency.json
```

## 5. Checkpoints

| category | coreset | path                                                                                            | SHA256 (first 16) | size MB |
|----------|---------|-------------------------------------------------------------------------------------------------|-------------------|---------|
| bottle   | 0.01    | artifacts/patchcore_bottle/Patchcore/MVTecAD/bottle/v0/weights/lightning/model.ckpt             | b0eb8834ae8d2bec  | 107.6   |
| cable    | 0.01    | artifacts/patchcore_cable/Patchcore/MVTecAD/cable/v0/weights/lightning/model.ckpt               | 29d451c6a03707c1  | 108.5   |
| capsule  | 0.01    | artifacts/patchcore_capsule/Patchcore/MVTecAD/capsule/v0/weights/lightning/model.ckpt           | 2545499571392618  | 108.2   |
| leather  | 0.01    | artifacts/patchcore_leather/Patchcore/MVTecAD/leather/v0/weights/lightning/model.ckpt           | 5cf7c7a793ad441a  | 109.8   |
| cable    | 0.10    | artifacts/patchcore_cable_coreset010/Patchcore/MVTecAD/cable/v0/weights/lightning/model.ckpt    | 5cac130be961ebfe  | 229.5   |
| cable    | 0.25    | artifacts/patchcore_cable_coreset025/Patchcore/MVTecAD/cable/v0/weights/lightning/model.ckpt    | 5979528b21adef3c  | 431.1   |

Full SHA256 strings are recorded in each `train_config.json` sidecar next to
the corresponding checkpoint, and the eval-harness JSONs record the
checkpoint SHA via `checkpoint_hash`.

## 6. What this does NOT show

- Single seed per (category, coreset) cell. No variance bars. The +0.0024
  bottle gain and the +0.0007 bottle AUPRO gain are within plausible
  seed-to-seed noise; the +0.119 cable and +0.114 capsule image-AUROC gains
  are large enough that single-seed is still informative but not
  statistically tested.
- AUPRO is anomalib's `_AUPRO` at `fpr_limit=0.3`. Different FPR limits or
  alternative AUPRO implementations will give different absolute numbers.
- Latency is PyTorch eager forward, batch 1, 256x256, on this specific
  hardware (Ryzen 9 9900X + NVIDIA Blackwell consumer GPU). No ONNX,
  no OpenVINO, no TensorRT,
  no Jetson, no end-to-end pipeline timing (no I/O, no preprocessing).
