import torch

from inspectnet_cx import InspectNetCXConfig, InspectNetCXForAnomalyDetection


def test_forward_output_shapes():
    config = InspectNetCXConfig(image_size=64)
    model = InspectNetCXForAnomalyDetection(config)

    output = model(pixel_values=torch.randn(2, 3, 64, 64))

    assert output.image_score.shape == (2,)
    assert output.anomaly_heatmap.shape == (2, 1, 64, 64)
    assert output.binary_mask.shape == (2, 1, 64, 64)
    assert output.threshold.shape == (2,)
    assert output.confidence.shape == (2,)
    assert len(output.defect_regions) == 2


def test_model_save_load_roundtrip(tmp_path):
    config = InspectNetCXConfig(image_size=64)
    model = InspectNetCXForAnomalyDetection(config)
    model.save_pretrained(tmp_path)

    loaded = InspectNetCXForAnomalyDetection.from_pretrained(tmp_path, trust_remote_code=True)
    output = loaded(pixel_values=torch.randn(1, 3, 64, 64))

    assert output.image_score.shape == (1,)
    assert loaded.config.image_size == 64
