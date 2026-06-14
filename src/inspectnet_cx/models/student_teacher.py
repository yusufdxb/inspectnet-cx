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
from torchvision.models import (
    ResNet18_Weights,
    Wide_ResNet50_2_Weights,
    resnet18,
    wide_resnet50_2,
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import gaussian_blur

_LAYERS = {"layer1": "l1", "layer2": "l2", "layer3": "l3"}

# (constructor, pretrained-weights enum) per supported backbone. wide_resnet50_2
# is PatchCore's backbone; resnet18 is the lighter default.
_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
    "wide_resnet50_2": (wide_resnet50_2, Wide_ResNet50_2_Weights.DEFAULT),
}


class StudentTeacher(nn.Module):
    """Frozen pretrained teacher + trainable student; anomaly = feature mismatch.

    Multi-scale by construction: the anomaly map sums the feature discrepancy
    across the layer1/2/3 pyramid. ``backbone`` selects the feature extractor;
    ``wide_resnet50_2`` matches PatchCore's backbone for a stronger comparison.
    """

    def __init__(self, backbone: str = "resnet18") -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; pick {list(_BACKBONES)}")
        ctor, weights = _BACKBONES[backbone]
        self.backbone = backbone
        self.teacher = create_feature_extractor(
            ctor(weights=weights), return_nodes=_LAYERS
        ).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.student = create_feature_extractor(ctor(weights=None), return_nodes=_LAYERS)

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
    def image_score(self, x: torch.Tensor, scales: list[int] | None = None) -> torch.Tensor:
        """Image-level anomaly score (B,): max over the anomaly map.

        With ``scales`` (e.g. [224, 256, 320]) the input is run at each resolution
        and the maps are fused before scoring, adding multi-resolution context on
        top of the layer pyramid. ``None`` keeps the single native-resolution path.
        """
        if not scales:
            return self.anomaly_map(x).amax(dim=(1, 2))
        base_hw = x.shape[-2:]
        fused = torch.zeros(x.shape[0], *base_hw, device=x.device)
        for s in scales:
            xr = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            m = self.anomaly_map(xr).unsqueeze(1)
            fused += F.interpolate(m, size=base_hw, mode="bilinear", align_corners=False).squeeze(1)
        return fused.amax(dim=(1, 2))
