from contextlib import suppress

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from inspectnet_cx.models.configuration_inspectnet_cx import InspectNetCXConfig
from inspectnet_cx.models.outputs import InspectNetCXOutput
from inspectnet_cx.models.postprocessing import masks_to_bounding_boxes


class InspectNetCXForAnomalyDetection(PreTrainedModel):
    config_class = InspectNetCXConfig
    base_model_prefix = "inspectnet_cx"
    main_input_name = "pixel_values"

    def __init__(self, config: InspectNetCXConfig) -> None:
        super().__init__(config)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.anomaly_head = nn.Conv2d(32, 1, kernel_size=1)
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        support_pixel_values: torch.Tensor | None = None,
        threshold: float | None = None,
        **_: object,
    ) -> InspectNetCXOutput:
        del support_pixel_values
        # TODO(phase1): replace this stub with normal prototypes and memory-bank scoring.
        features = self.encoder(pixel_values)
        heatmap = torch.sigmoid(self.anomaly_head(features))
        heatmap = F.interpolate(
            heatmap,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        image_score = heatmap.flatten(1).amax(dim=1)
        threshold_value = self.config.threshold if threshold is None else threshold
        threshold_tensor = torch.full_like(image_score, float(threshold_value))
        binary_mask = (heatmap >= threshold_tensor.view(-1, 1, 1, 1)).to(pixel_values.dtype)
        confidence = (image_score - threshold_tensor).sigmoid()
        defect_regions = masks_to_bounding_boxes(binary_mask)

        return InspectNetCXOutput(
            image_score=image_score,
            anomaly_heatmap=heatmap,
            binary_mask=binary_mask,
            threshold=threshold_tensor,
            confidence=confidence,
            defect_regions=defect_regions,
        )

with suppress(AttributeError, ValueError):
    InspectNetCXForAnomalyDetection.register_for_auto_class("AutoModel")
