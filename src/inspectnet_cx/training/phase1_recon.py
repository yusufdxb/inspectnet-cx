"""Honest Phase 1 native detector: small autoencoder trained on normal-only data.

The Sprint 2 phase1 trainer used BCE-against-zero on a sigmoid head with
normal-only labels. That loss collapses to zero without learning anomaly
structure (the gradient on a saturating sigmoid trained against an
all-zero target shrinks rapidly and provides no useful supervision for
distinguishing normal from anomalous patches at test time).

This module replaces that scaffold with a reconstruction-style baseline:

- Encoder: 4 conv blocks (3 -> 32 -> 64 -> 128 -> 128), stride-2 downsampling.
- Decoder: mirrored 4 transposed-conv blocks back to 3 channels.
- Loss: per-pixel L2 MSE on RGB reconstruction.
- Per-image anomaly score at inference: mean squared reconstruction error
  across the image (or any reduction; mean is reported here).

Rationale: a small CNN autoencoder is the simplest honest baseline that
fits on the RTX 5070, trains in minutes per category, and produces a
calibrated per-image anomaly score that can be compared head to head
against PaDiM with the same AUROC + bootstrap-CI methodology. It is not
state of the art; it is an honest learned baseline.

This trainer does NOT modify the released ``InspectNetCXForAnomalyDetection``
surface; it lives in the training package so the HF release tag stays
unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split


class ReconAutoencoder(nn.Module):
    """Small symmetric conv autoencoder for normal-only training."""

    def __init__(self, in_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base * 4, base * 4, 4, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 4, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(base, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


class ImageFolderDataset(Dataset):
    """Loads images from a flat directory (or nested), resizes to image_size,
    returns float tensors in [0, 1]."""

    def __init__(self, image_paths: list[Path], image_size: int) -> None:
        self.image_paths = sorted(image_paths)
        self.image_size = image_size
        if not self.image_paths:
            raise ValueError("ImageFolderDataset: empty image_paths")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


@dataclass
class TrainResult:
    train_loss_history: list[float]
    val_loss_history: list[float]
    best_val_loss: float
    best_epoch: int
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    device: str
    parameter_count: int
    train_seconds: float
    train_size: int
    val_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "inspectnet_cx.phase1_recon.v1",
            "train_loss_history": self.train_loss_history,
            "val_loss_history": self.val_loss_history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "device": self.device,
            "parameter_count": self.parameter_count,
            "train_seconds": self.train_seconds,
            "train_size": self.train_size,
            "val_size": self.val_size,
        }


def _collect_pngs(d: Path) -> list[Path]:
    return sorted([p for p in d.rglob("*.png") if p.is_file()])


def train_recon(
    train_data_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    image_size: int = 128,
    val_ratio: float = 0.2,
    device: str | None = None,
    seed: int = 0,
) -> TrainResult:
    """Train the reconstruction autoencoder on normal-only images.

    The dataset is split deterministically into train / val by ``val_ratio``.
    Best checkpoint (lowest val MSE) is saved to ``output_dir/best.pt``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_pngs(Path(train_data_dir))
    if len(image_paths) < 4:
        raise ValueError(
            f"train_data_dir {train_data_dir} only has {len(image_paths)} PNGs; need >= 4"
        )
    full = ImageFolderDataset(image_paths, image_size=image_size)
    n_val = max(1, round(val_ratio * len(full)))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )

    model = ReconAutoencoder().to(device_obj)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    best_val = float("inf")
    best_epoch = -1
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device_obj, non_blocking=True)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_train = epoch_loss / max(1, n_batches)
        train_loss_history.append(avg_train)

        model.eval()
        v_loss = 0.0
        v_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device_obj, non_blocking=True)
                recon = model(batch)
                v_loss += float(loss_fn(recon, batch).item())
                v_batches += 1
        avg_val = v_loss / max(1, v_batches)
        val_loss_history.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_epoch = epoch
            torch.save(
                {"model_state_dict": model.state_dict(), "image_size": image_size},
                output_dir / "best.pt",
            )

        print(
            f"epoch {epoch + 1}/{epochs} train_mse={avg_train:.6f} val_mse={avg_val:.6f}"
            f" best={best_val:.6f}@{best_epoch + 1}"
        )

    train_seconds = time.perf_counter() - t0
    result = TrainResult(
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        best_val_loss=best_val,
        best_epoch=int(best_epoch),
        epochs=int(epochs),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        image_size=int(image_size),
        device=str(device_obj),
        parameter_count=model.parameter_count(),
        train_seconds=float(train_seconds),
        train_size=int(n_train),
        val_size=int(n_val),
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n"
    )
    return result


def load_recon(checkpoint: Path, device: str | None = None) -> tuple[ReconAutoencoder, int]:
    """Load a trained ``ReconAutoencoder`` from ``best.pt``."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    model = ReconAutoencoder().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, int(ckpt.get("image_size", 128))


def score_image_paths(
    model: ReconAutoencoder,
    image_paths: list[Path],
    image_size: int,
    device: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Return mean per-image squared reconstruction error."""
    ds = ImageFolderDataset(image_paths, image_size=image_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            err = (recon - batch).pow(2).mean(dim=(1, 2, 3))
            scores.extend(err.detach().cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    train_recon(
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
