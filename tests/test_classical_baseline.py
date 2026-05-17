from inspectnet_cx.eval.classical_baseline import run_classical_baseline
from inspectnet_cx.eval.fixture_smoke import create_tiny_mvtec_fixture, run_fixture_smoke
from inspectnet_cx.eval.validate_results import validate_results


def test_classical_baseline_produces_numeric_fixture_metrics(tmp_path):
    dataset_root = tmp_path / "datasets"
    create_tiny_mvtec_fixture(dataset_root)

    report = run_classical_baseline(
        dataset_root=dataset_root,
        dataset="mvtec_ad",
        category="bottle",
        output=tmp_path / "classical_patchdiff_fixture.json",
        image_size=24,
        quantile=0.5,
    )

    assert report["status"] == "classical_baseline_completed"
    assert report["train_normal_count"] == 2
    assert report["test_sample_count"] == 2
    assert isinstance(report["image_auroc"], float)
    assert isinstance(report["image_f1"], float)
    assert report["pixel_f1"] == "TBD"
    assert report["peak_vram_mb"] == 0.0
    assert (tmp_path / "classical_patchdiff_fixture.json").exists()


def test_fixture_smoke_embeds_classical_baseline_without_benchmark_misclassification(tmp_path):
    report = run_fixture_smoke(tmp_path, image_size=16)

    assert report["classical_baseline"]["status"] == "classical_baseline_completed"
    assert report["classical_baseline"]["image_auroc"] == 1.0
    assert validate_results(tmp_path) == {}
