import pytest
import torch
from PIL import Image

from inspectnet_cx.calibration import NormalQuantileCalibrator
from inspectnet_cx.calibration.normal_split import (
    build_normal_calibration_report,
    find_normal_images,
)
from inspectnet_cx.release.create_phase0_model import create_phase0_model


def test_normal_quantile_calibrator_fit_and_apply():
    calibrator = NormalQuantileCalibrator(quantile=0.5)
    scores = torch.tensor([0.1, 0.3, 0.9])

    threshold = calibrator.fit(scores)
    decisions, confidence = calibrator.apply(scores, threshold)

    assert torch.isclose(threshold, torch.tensor(0.3))
    assert decisions.tolist() == [False, True, True]
    assert confidence.shape == scores.shape


def test_normal_quantile_calibrator_rejects_empty_scores():
    calibrator = NormalQuantileCalibrator()

    with pytest.raises(ValueError, match="must not be empty"):
        calibrator.fit(torch.tensor([]))


def test_find_normal_images_mvtec_layout(tmp_path):
    image_path = tmp_path / "mvtec_ad" / "bottle" / "train" / "good" / "000.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(image_path)

    images = find_normal_images(tmp_path, "mvtec_ad", "bottle")

    assert images == [image_path]


def test_build_normal_calibration_report_blocks_without_images(tmp_path):
    model_dir = create_phase0_model(tmp_path / "model", image_size=16)

    report = build_normal_calibration_report(
        model_dir=model_dir,
        dataset_root=tmp_path,
        dataset="mvtec_ad",
        category="bottle",
    )

    assert report["status"] == "blocked"
    assert report["threshold_source"] == "none"
