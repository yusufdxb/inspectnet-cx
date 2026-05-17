import pytest
import torch

from inspectnet_cx.models.postprocessing import masks_to_bounding_boxes


def test_masks_to_bounding_boxes_returns_region_with_area():
    mask = torch.zeros(1, 1, 8, 8)
    mask[0, 0, 2:5, 3:7] = 1

    regions = masks_to_bounding_boxes(mask)

    assert regions == [[{"x_min": 3.0, "y_min": 2.0, "x_max": 6.0, "y_max": 4.0, "area": 12.0}]]


def test_masks_to_bounding_boxes_rejects_bad_shape():
    with pytest.raises(ValueError, match="Nx1xHxW"):
        masks_to_bounding_boxes(torch.zeros(1, 8, 8))
