"""Tests for inspectnet_cx.batch_inference."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from inspectnet_cx import (
    InspectNetCXConfig,
    InspectNetCXForAnomalyDetection,
    InspectNetCXProcessor,
)
from inspectnet_cx.batch_inference import iter_images, main, score_images, write_csv


def _build_phase0_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    InspectNetCXForAnomalyDetection(InspectNetCXConfig(image_size=32)).save_pretrained(model_dir)
    InspectNetCXProcessor(image_size=32).save_pretrained(model_dir)
    return model_dir


def _make_images(root: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        path = root / f"img_{i:03d}.png"
        array = np.full((32, 32, 3), i * 10 % 255, dtype=np.uint8)
        Image.fromarray(array).save(path)
        paths.append(path)
    return paths


def test_iter_images_sorted_and_extension_filtered(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _make_images(image_dir, 3)
    (image_dir / "notes.txt").write_text("ignore me")
    results = list(iter_images(image_dir))
    assert [p.name for p in results] == ["img_000.png", "img_001.png", "img_002.png"]


def test_iter_images_handles_single_file(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    [single] = _make_images(image_dir, 1)
    results = list(iter_images(single))
    assert results == [single]


def test_score_images_emits_one_record_per_image(tmp_path: Path) -> None:
    model_dir = _build_phase0_model(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    images = _make_images(image_dir, 4)
    records = score_images(model_dir, images, batch_size=2, threshold=0.5)
    assert len(records) == 4
    for record in records:
        assert set(record.keys()) == {"path", "score", "threshold", "decision"}
        assert record["decision"] in {"anomaly", "normal"}
        assert isinstance(record["score"], float)


def test_write_csv_rel_paths_and_header(tmp_path: Path) -> None:
    records = [
        {"path": str(tmp_path / "a.png"), "score": 0.1, "threshold": 0.5, "decision": "normal"},
        {"path": str(tmp_path / "b.png"), "score": 0.9, "threshold": 0.5, "decision": "anomaly"},
    ]
    output = tmp_path / "out.csv"
    write_csv(records, output, root=tmp_path)
    rows = list(csv.reader(output.open()))
    assert rows[0] == ["path", "score", "threshold", "decision"]
    assert rows[1] == ["a.png", "0.100000", "0.500000", "normal"]
    assert rows[2] == ["b.png", "0.900000", "0.500000", "anomaly"]


def test_main_cli_writes_csv(tmp_path: Path) -> None:
    model_dir = _build_phase0_model(tmp_path)
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _make_images(image_dir, 3)
    output = tmp_path / "predictions.csv"

    main(
        [
            "--model",
            str(model_dir),
            "--input",
            str(image_dir),
            "--output",
            str(output),
            "--batch-size",
            "2",
            "--threshold",
            "0.5",
        ]
    )

    rows = list(csv.reader(output.open()))
    assert rows[0] == ["path", "score", "threshold", "decision"]
    assert len(rows) == 4  # header + 3 images


def test_main_cli_raises_on_empty_folder(tmp_path: Path) -> None:
    model_dir = _build_phase0_model(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    try:
        main(
            [
                "--model",
                str(model_dir),
                "--input",
                str(empty),
                "--output",
                str(tmp_path / "out.csv"),
            ]
        )
    except SystemExit as exc:
        assert "no images" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("expected SystemExit on empty folder")
