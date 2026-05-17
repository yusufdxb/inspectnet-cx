from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NormalQuantileCalibrator:
    quantile: float = 0.995
    min_threshold: float = 0.0
    max_threshold: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile < 1.0:
            msg = "quantile must be between 0 and 1."
            raise ValueError(msg)
        if self.min_threshold > self.max_threshold:
            msg = "min_threshold must be less than or equal to max_threshold."
            raise ValueError(msg)

    def fit(self, normal_scores: torch.Tensor) -> torch.Tensor:
        if normal_scores.numel() == 0:
            msg = "normal_scores must not be empty."
            raise ValueError(msg)
        scores = normal_scores.detach().float().flatten()
        threshold = torch.quantile(scores, self.quantile)
        return threshold.clamp(self.min_threshold, self.max_threshold)

    def apply(
        self,
        scores: torch.Tensor,
        threshold: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        threshold_tensor = torch.as_tensor(threshold, dtype=scores.dtype, device=scores.device)
        decisions = scores >= threshold_tensor
        confidence = (scores - threshold_tensor).sigmoid()
        return decisions, confidence
