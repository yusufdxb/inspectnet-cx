from inspectnet_cx import InspectNetCXConfig


def test_config_roundtrip(tmp_path):
    config = InspectNetCXConfig(image_size=128, prototype_size=32)
    config.save_pretrained(tmp_path)

    loaded = InspectNetCXConfig.from_pretrained(tmp_path)

    assert loaded.image_size == 128
    assert loaded.prototype_size == 32
    assert loaded.model_type == "inspectnet-cx"
