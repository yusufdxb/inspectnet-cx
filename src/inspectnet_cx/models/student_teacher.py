"""InspectNet-CX student-teacher anomaly detector.

A real, trainable detector (not the HF API scaffold in modeling_inspectnet_cx.py).
A frozen ImageNet-pretrained ResNet18 teacher supervises a from-scratch student on
normal images only; at test time the per-pixel feature discrepancy between the two
is the anomaly map (the Student-Teacher Feature Pyramid Matching idea, Wang et al.
2021). Normal regions match because the student saw them in training; defects do
not, because the student never learned to reproduce the teacher there.

ponytail: reuses torchvision resnet18 for both networks via create_feature_extractor;
no custom backbone, no pyramid-fusion config, no new dependency.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import gaussian_blur

_LAYERS = {"layer1": "l1", "layer2": "l2", "layer3": "l3"}


class StudentTeacher(nn.Module):
    """Frozen pretrained teacher + trainable student; anomaly = feature mismatch."""

    def __init__(self) -> None:
        super().__init__()
        self.teacher = create_feature_extractor(
            resnet18(weights=ResNet18_Weights.DEFAULT), return_nodes=_LAYERS
        ).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = create_feature_extractor(
            resnet18(weights=None), return_nodes=_LAYERS
        )

    def _features(self, net: nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
        # Channel-wise unit-normalize so the loss/score is scale-free per layer.
        return [F.normalize(f, dim=1) for f in net(x).values()]

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Training loss: student matches teacher features on normal images."""
        with torch.no_grad():
            t = self._features(self.teacher, x)
        s = self._features(self.student, x)
        return sum(F.mse_loss(si, ti) for si, ti in zip(s, t, strict=True)) / len(t)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        """Per-pixel anomaly map (B,H,W), upsampled to input resolution.

        Gaussian-smoothed (sigma=4, the STFPM default) so the image score is the
        peak of a defect *region*, not a single noisy pixel.
        """
        t = self._features(self.teacher, x)
        s = self._features(self.student, x)
        hw = x.shape[-2:]
        amap = torch.zeros(x.shape[0], 1, *hw, device=x.device)
        for si, ti in zip(s, t, strict=True):
            layer_map = 0.5 * ((si - ti) ** 2).sum(dim=1, keepdim=True)  # (B,1,h,w)
            amap += F.interpolate(layer_map, size=hw, mode="bilinear", align_corners=False)
        amap = gaussian_blur(amap, kernel_size=33, sigma=4.0)
        return amap.squeeze(1)

    @torch.no_grad()
    def image_score(self, x: torch.Tensor) -> torch.Tensor:
        """Image-level anomaly score (B,): max over the anomaly map."""
        return self.anomaly_map(x).amax(dim=(1, 2))
