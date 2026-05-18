"""Smoke tests for the reconstruction AE Phase 1 trainer."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from inspectnet_cx.training.phase1_recon import (
    ReconAutoencoder,
    score_image_paths,
    train_recon,
)


def test_recon_autoencoder_forward_shape() -> None:
    m = ReconAutoencoder()
    x = torch.randn(2, 3, 128, 128)
    y = m(x)
    assert y.shape == x.shape
    assert m.parameter_count() > 0


def test_recon_train_smoke(tmp_path) -> None:
    train_dir = tmp_path / "imgs"
    train_dir.mkdir()
    for i in range(6):
        arr = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(train_dir / f"img_{i:02d}.png")

    out = tmp_path / "model"
    res = train_recon(
        train_data_dir=train_dir,
        output_dir=out,
        epochs=2,
        batch_size=2,
        learning_rate=1e-3,
        image_size=64,
        val_ratio=0.25,
        device="cpu",
        seed=0,
    )
    assert (out / "best.pt").exists()
    assert (out / "training_metrics.json").exists()
    assert len(res.train_loss_history) == 2
    assert res.parameter_count > 0


def test_recon_score_image_paths(tmp_path) -> None:
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    for i in range(3):
        arr = np.random.default_rng(i).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_dir / f"img_{i:02d}.png")
    paths = sorted(img_dir.glob("*.png"))
    model = ReconAutoencoder()
    scores = score_image_paths(model, paths, image_size=64, device="cpu", batch_size=2)
    assert scores.shape == (3,)
    assert (scores >= 0).all()
