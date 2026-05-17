from __future__ import annotations

import torch


def masks_to_bounding_boxes(binary_mask: torch.Tensor) -> list[list[dict[str, float]]]:
    if binary_mask.ndim != 4 or binary_mask.shape[1] != 1:
        msg = "binary_mask must have shape Nx1xHxW."
        raise ValueError(msg)

    regions: list[list[dict[str, float]]] = []
    masks = binary_mask.detach().cpu()
    for mask in masks:
        active = torch.nonzero(mask[0] > 0, as_tuple=False)
        if active.numel() == 0:
            regions.append([])
            continue
        y_min, x_min = active.min(dim=0).values.tolist()
        y_max, x_max = active.max(dim=0).values.tolist()
        area = float(active.shape[0])
        regions.append(
            [
                {
                    "x_min": float(x_min),
                    "y_min": float(y_min),
                    "x_max": float(x_max),
                    "y_max": float(y_max),
                    "area": area,
                }
            ]
        )
    return regions
