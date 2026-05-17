from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from inspectnet_cx.prediction import predict_images


def test_classical_prediction_writes_json_and_mask(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    category = root / "mvtec_ad" / "bottle"
    train = category / "train" / "good"
    test_good = category / "test" / "good"
    test_bad = category / "test" / "broken"
    train.mkdir(parents=True)
    test_good.mkdir(parents=True)
    test_bad.mkdir(parents=True)
    for idx in range(3):
        _write_image(train / f"{idx:03d}.png", value=64)
    good_path = test_good / "000.png"
    bad_path = test_bad / "000.png"
    _write_image(good_path, value=64)
    _write_image(bad_path, value=220)

    output = tmp_path / "predictions.json"
    report = predict_images(
        input_path=category / "test",
        dataset_root=root,
        dataset="mvtec_ad",
        category="bottle",
        backend="classical_patchdiff",
        output=output,
        image_size=16,
        threshold_quantile=0.95,
    )

    assert output.exists()
    reloaded = json.loads(output.read_text())
    assert reloaded["status"] == "completed_predictions"
    assert report["input_count"] == 2
    assert {row["expected_label"] for row in report["predictions"]} == {"normal", "anomaly"}
    assert all(Path(row["mask_path"]).exists() for row in report["predictions"])


def _write_image(path: Path, value: int) -> None:
    arr = np.full((24, 24, 3), value, dtype=np.uint8)
    Image.fromarray(arr).save(path)
