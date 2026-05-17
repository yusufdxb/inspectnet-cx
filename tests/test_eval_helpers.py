import json
from pathlib import Path

from inspectnet_cx.eval.aggregate import load_results, render_markdown
from inspectnet_cx.eval.baseline import main as baseline_main
from inspectnet_cx.eval.fixture_smoke import create_tiny_mvtec_fixture
from inspectnet_cx.eval.result_schema import validate_result_payload


def test_baseline_main_writes_placeholder_json(tmp_path):
    output = tmp_path / "result.json"

    baseline_main(
        [
            "--method",
            "patchcore",
            "--dataset",
            "mvtec_ad",
            "--category",
            "bottle",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )

    assert output.exists()
    assert '"status": "phase0_placeholder"' in output.read_text()


def test_classical_baseline_runs_on_tiny_mvtec_fixture(tmp_path):
    dataset_root = tmp_path / "datasets"
    create_tiny_mvtec_fixture(dataset_root)
    output = tmp_path / "result.json"

    baseline_main(
        [
            "--method",
            "classical-range",
            "--dataset",
            "mvtec_ad",
            "--category",
            "bottle",
            "--data-root",
            str(dataset_root),
            "--image-size",
            "16",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["status"] == "completed_classical_cpu_baseline"
    assert payload["baseline_version"] == "classical-range-v1"
    assert payload["train_image_count"] == 2
    assert payload["test_image_count"] == 2
    assert payload["image_auroc"] == 1.0
    assert payload["pixel_auroc"] == "TBD"
    assert validate_result_payload(payload) == []


def test_aggregate_renders_markdown(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        '{"method": "patchcore", "dataset": "mvtec_ad", "category": "bottle", "status": "TBD"}'
    )

    markdown = render_markdown(load_results(Path(tmp_path)))

    assert "Baseline Leaderboard" in markdown
    assert "patchcore" in markdown
