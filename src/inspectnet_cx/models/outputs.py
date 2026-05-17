from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class InspectNetCXOutput(ModelOutput):
    image_score: torch.Tensor | None = None
    anomaly_heatmap: torch.Tensor | None = None
    binary_mask: torch.Tensor | None = None
    threshold: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    defect_regions: list[list[dict[str, float]]] | None = None
