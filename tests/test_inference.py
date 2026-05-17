import json

import numpy as np
from PIL import Image

from inspectnet_cx import InspectNetCXConfig, InspectNetCXForAnomalyDetection, InspectNetCXProcessor
from inspectnet_cx.inference import run_inference


def test_run_inference_writes_json(tmp_path):
    model_dir = tmp_path / "model"
    image_path = tmp_path / "part.png"
    output_path = tmp_path / "result.json"

    model = InspectNetCXForAnomalyDetection(InspectNetCXConfig(image_size=32))
    processor = InspectNetCXProcessor(image_size=32)
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    image.save(image_path)

    payload = run_inference(model_dir, image_path, output_path, threshold=0.5)
    written = json.loads(output_path.read_text())

    assert payload["heatmap_shape"] == [1, 1, 32, 32]
    assert written["mask_shape"] == [1, 1, 32, 32]
    assert "defect_regions" in written
