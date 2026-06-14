#!/usr/bin/env python3
"""Train + evaluate the InspectNet-CX student-teacher detector on a MVTec AD category.

Trains the student on normal (train/good) images only, then reports image-level
AUROC on the held-out test split. Writes a result JSON compatible with the repo's
honest-evidence convention. This is the repo's first natively trained detector.

    PYTHONPATH=src python3 scripts/train_inspectnet_cx.py \
        --category bottle --dataset-root ~/datasets/mvtec_ad \
        --epochs 50 --device cuda --output reports/eval_harness/inspectnet_st_bottle.json

ponytail: glob + PIL for data (no custom Dataset class hierarchy); sklearn
roc_auc_score for the metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from inspectnet_cx.models.student_teacher import StudentTeacher

# ImageNet stats: the teacher is ImageNet-pretrained, so inputs must match.
_TF = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class _Images(Dataset):
    def __init__(self, paths: list[Path], labels: list[int] | None = None) -> None:
        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = _TF(Image.open(self.paths[i]).convert("RGB"))
        return (x, self.labels[i]) if self.labels is not None else x


def _train_paths(cat: Path) -> list[Path]:
    return sorted((cat / "train" / "good").glob("*.png"))


def _test_paths(cat: Path) -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    for sub in sorted((cat / "test").iterdir()):
        if not sub.is_dir():
            continue
        for p in sorted(sub.glob("*.png")):
            paths.append(p)
            labels.append(0 if sub.name == "good" else 1)
    return paths, labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", required=True)
    ap.add_argument("--dataset-root", required=True, help="path to mvtec_ad root")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "wide_resnet50_2"])
    ap.add_argument("--multiscale", action="store_true", help="fuse maps over 224/256/320 at eval")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cat = Path(args.dataset_root).expanduser() / args.category
    if not cat.is_dir():
        raise FileNotFoundError(f"missing category root {cat}")

    train_paths = _train_paths(cat)
    test_paths, test_labels = _test_paths(cat)
    if not train_paths or not test_paths:
        raise RuntimeError(f"no images found under {cat}")

    device = args.device
    model = StudentTeacher(backbone=args.backbone).to(device)
    opt = torch.optim.Adam(model.student.parameters(), lr=args.lr, weight_decay=1e-5)

    loader = DataLoader(
        _Images(train_paths), batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    model.student.train()
    for epoch in range(args.epochs):
        total = 0.0
        for x in loader:
            x = x.to(device)
            opt.zero_grad()
            loss = model.loss(x)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {total / len(train_paths):.6f}", flush=True)

    model.student.eval()
    test_loader = DataLoader(
        _Images(test_paths, test_labels), batch_size=args.batch_size, num_workers=4
    )
    scales = [224, 256, 320] if args.multiscale else None
    scores: list[float] = []
    for x, _ in test_loader:
        scores.extend(model.image_score(x.to(device), scales=scales).cpu().tolist())

    labels = np.array(test_labels, dtype=np.int64)
    auroc = float(roc_auc_score(labels, np.array(scores)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "inspectnet_cx.student_teacher.v1",
        "model": "inspectnet_cx_student_teacher",
        "backbone": f"{args.backbone} (teacher: ImageNet-pretrained, frozen)",
        "category": args.category,
        "dataset": "mvtec_ad",
        "image_auroc": auroc,
        "epochs": args.epochs,
        "n_train": len(train_paths),
        "n_test": len(test_paths),
        "n_test_anomaly": int(labels.sum()),
        "device": device,
        "seed": args.seed,
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"image_auroc={auroc:.4f}  ->  {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
