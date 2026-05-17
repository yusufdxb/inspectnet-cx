import numpy as np
from PIL import Image

from inspectnet_cx import InspectNetCXProcessor


def make_image():
    return Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))


def test_processor_handles_one_image():
    processor = InspectNetCXProcessor(image_size=64)

    output = processor(images=make_image())

    assert output["pixel_values"].shape == (1, 3, 64, 64)


def test_processor_handles_support_images():
    processor = InspectNetCXProcessor(image_size=64)

    output = processor(images=make_image(), support_images=[make_image(), make_image()])

    assert output["pixel_values"].shape == (1, 3, 64, 64)
    assert output["support_pixel_values"].shape == (2, 3, 64, 64)


def test_processor_save_load(tmp_path):
    processor = InspectNetCXProcessor(image_size=80)
    processor.save_pretrained(tmp_path)

    loaded = InspectNetCXProcessor.from_pretrained(tmp_path, trust_remote_code=True)

    assert loaded.image_size == 80
