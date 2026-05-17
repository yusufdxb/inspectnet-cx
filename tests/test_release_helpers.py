from inspectnet_cx import InspectNetCXForAnomalyDetection, InspectNetCXProcessor
from inspectnet_cx.release.create_phase0_model import create_phase0_model


def test_create_phase0_model_roundtrip(tmp_path):
    output = create_phase0_model(tmp_path / "model", image_size=32)

    processor = InspectNetCXProcessor.from_pretrained(output)
    model = InspectNetCXForAnomalyDetection.from_pretrained(output)

    assert processor.image_size == 32
    assert model.config.image_size == 32
    assert (output / "README.md").exists()
