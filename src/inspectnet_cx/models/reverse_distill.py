"""InspectNet-CX reverse-distillation anomaly detector.

The vanilla student-teacher detector (student_teacher.py) trails PatchCore because a
trained student that mimics the teacher in the *same* direction tends to also
reproduce anomalies. Reverse distillation (Deng & Li, CVPR 2022) fixes the failure
mode: a FROZEN teacher encodes the image, a compact bottleneck fuses its multi-scale
features, and a trainable DECODER reconstructs those teacher features from the
bottleneck. On normal data the decoder learns to rebuild teacher features; on a
defect the reconstruction diverges, so per-pixel cosine distance is the anomaly map.
Because the teacher is frozen and the decoder only sees a compressed bottleneck, the
decoder cannot trivially copy anomalous features, which is exactly the capacity trap
that sank the wide_resnet50_2 student.

ponytail: a compact bottleneck + progressive decoder over the frozen wide_resnet50_2
layer1-3 pyramid, not a full de-resnet; same loss/anomaly_map/image_score interface
as StudentTeacher so the existing train script drives it unchanged.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms.functional import gaussian_blur

_LAYERS = {"layer1": "l1", "layer2": "l2", "layer3": "l3"}
# wide_resnet50_2 channel widths at layer1/2/3.
_CH = (256, 512, 1024)


def _cbr(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
    )


class ReverseDistill(nn.Module):
    """Frozen teacher + bottleneck + trainable decoder; anomaly = reconstruction gap."""

    def __init__(self) -> None:
        super().__init__()
        self.teacher = create_feature_extractor(
            wide_resnet50_2(weights=Wide_ResNet50_2_Weights.DEFAULT), return_nodes=_LAYERS
        ).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        # Bottleneck: bring layer1/2 down to the layer3 grid, fuse, compress.
        self.reduce1 = nn.Conv2d(_CH[0], 256, 3, stride=4, padding=1)
        self.reduce2 = nn.Conv2d(_CH[1], 256, 3, stride=2, padding=1)
        self.reduce3 = nn.Conv2d(_CH[2], 256, 3, padding=1)
        self.fuse = _cbr(768, 256)

        # Progressive decoder: rebuild each teacher layer from the bottleneck.
        self.dec3 = _cbr(256, 256)
        self.head3 = nn.Conv2d(256, _CH[2], 1)
        self.dec2 = _cbr(256, 256)
        self.head2 = nn.Conv2d(256, _CH[1], 1)
        self.dec1 = _cbr(256, 256)
        self.head1 = nn.Conv2d(256, _CH[0], 1)

        self.trainable_module = nn.ModuleList(
            [self.reduce1, self.reduce2, self.reduce3, self.fuse,
             self.dec3, self.head3, self.dec2, self.head2, self.dec1, self.head1]
        )

    def _teacher_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        with torch.no_grad():
            return list(self.teacher(x).values())  # [l1 64x, l2 32x, l3 16x]

    def _reconstruct(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        t1, t2, t3 = feats
        b = self.fuse(torch.cat([self.reduce1(t1), self.reduce2(t2), self.reduce3(t3)], dim=1))
        d3 = self.dec3(b)
        r3 = self.head3(d3)  # match t3
        d2 = self.dec2(F.interpolate(d3, size=t2.shape[-2:], mode="bilinear", align_corners=False))
        r2 = self.head2(d2)  # match t2
        d1 = self.dec1(F.interpolate(d2, size=t1.shape[-2:], mode="bilinear", align_corners=False))
        r1 = self.head1(d1)  # match t1
        return [r1, r2, r3]

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        t = self._teacher_features(x)
        r = self._reconstruct(t)
        # 1 - cosine similarity along channels, averaged over pixels and layers.
        return sum(
            (1.0 - F.cosine_similarity(ri, ti, dim=1)).mean() for ri, ti in zip(r, t, strict=True)
        ) / len(t)

    @torch.no_grad()
    def anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        t = self._teacher_features(x)
        r = self._reconstruct(t)
        hw = x.shape[-2:]
        amap = torch.zeros(x.shape[0], 1, *hw, device=x.device)
        for ri, ti in zip(r, t, strict=True):
            layer_map = (1.0 - F.cosine_similarity(ri, ti, dim=1)).unsqueeze(1)  # (B,1,h,w)
            amap += F.interpolate(layer_map, size=hw, mode="bilinear", align_corners=False)
        return gaussian_blur(amap, kernel_size=33, sigma=4.0).squeeze(1)

    @torch.no_grad()
    def image_score(self, x: torch.Tensor, scales: list[int] | None = None) -> torch.Tensor:
        if not scales:
            return self.anomaly_map(x).amax(dim=(1, 2))
        base_hw = x.shape[-2:]
        fused = torch.zeros(x.shape[0], *base_hw, device=x.device)
        for s in scales:
            xr = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            m = self.anomaly_map(xr).unsqueeze(1)
            fused += F.interpolate(m, size=base_hw, mode="bilinear", align_corners=False).squeeze(1)
        return fused.amax(dim=(1, 2))
