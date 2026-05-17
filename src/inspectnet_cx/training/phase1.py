"""Phase 1 native detector training script.

Trains the InspectNetCXForAnomalyDetection model on normal-only data using a
feature-distance loss. The goal is to minimize anomaly predictions on normal images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from inspectnet_cx import (
    InspectNetCXConfig,
    InspectNetCXForAnomalyDetection,
    InspectNetCXProcessor,
)


class NormalImageDataset(Dataset):
    """Dataset for loading normal training images."""

    def __init__(
        self,
        image_dir: Path,
        processor: InspectNetCXProcessor,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            msg = f"No .png files found in {image_dir}"
            raise ValueError(msg)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
        }


def train_phase1(
    train_data_dir: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    image_size: int = 64,
    device: str | None = None,
) -> dict[str, Any]:
    """Train Phase 1 native detector on normal-only data.

    Args:
        train_data_dir: Directory containing normal training images.
        output_dir: Directory to save the trained model.
        epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate for optimizer.
        image_size: Input image size (square).
        device: Device to use ("cpu", "cuda", or None for auto-detect).

    Returns:
        Dictionary with training metrics.
    """
    # Detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model and processor
    config = InspectNetCXConfig(image_size=image_size)
    model = InspectNetCXForAnomalyDetection(config).to(device_obj)
    processor = InspectNetCXProcessor(image_size=image_size)

    # Create dataset and dataloader
    dataset = NormalImageDataset(train_data_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    # Training loop
    model.train()
    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device_obj)

            # Forward pass
            output = model(pixel_values=pixel_values)

            # Loss: minimize anomaly predictions on normal images
            # We want the heatmap to be close to 0 for normal images
            target = torch.zeros_like(output.anomaly_heatmap)
            loss = loss_fn(output.anomaly_heatmap, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        loss_history.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs} Loss: {avg_epoch_loss:.6f}")

    # Save model and processor
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    return {
        "loss_history": loss_history,
        "final_loss": loss_history[-1],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "device": str(device_obj),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Phase 1 native InspectNet-CX detector on normal data."
    )
    parser.add_argument(
        "--train-data-dir",
        required=True,
        type=Path,
        help="Directory containing normal training images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Input image size (square).",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Device to use (cpu or cuda). Auto-detected if not specified.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    args = parse_args(argv)
    metrics = train_phase1(
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        device=args.device,
    )

    # Save metrics to JSON
    metrics_path = Path(args.output_dir) / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
