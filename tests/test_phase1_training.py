"""Tests for Phase 1 training module."""

import numpy as np
from PIL import Image

from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor
from inspectnet_cx.training.phase1 import train_phase1


def test_phase1_training_smoke(tmp_path):
    """Test Phase 1 training on synthetic fixture data."""
    # Create a synthetic 3-image fixture directory
    fixture_dir = tmp_path / "fixture_images"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        image_array = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        image.save(fixture_dir / f"image_{i}.png")

    # Train for 1 epoch on CPU
    output_dir = tmp_path / "trained_model"
    metrics = train_phase1(
        train_data_dir=fixture_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        image_size=64,
        device="cpu",
    )

    # Verify training completed
    assert metrics["epochs"] == 1
    assert metrics["final_loss"] > 0.0
    assert len(metrics["loss_history"]) == 1

    # Verify model was saved and can be loaded
    loaded_model = InspectNetCXForAnomalyDetection.from_pretrained(
        output_dir, trust_remote_code=True
    )
    assert loaded_model is not None
    assert loaded_model.config.image_size == 64

    # Verify processor was saved and can be loaded
    loaded_processor = InspectNetCXProcessor.from_pretrained(output_dir)
    assert loaded_processor is not None
    assert loaded_processor.image_size == 64

    # Verify the loaded model can do inference
    import torch

    test_input = torch.randn(1, 3, 64, 64)
    output = loaded_model(pixel_values=test_input)
    assert output.image_score.shape == (1,)
    assert output.anomaly_heatmap.shape == (1, 1, 64, 64)
