from inspectnet_cx.eval.fixture_smoke import run_fixture_smoke


def test_fixture_smoke_runs_end_to_end(tmp_path):
    report = run_fixture_smoke(tmp_path, image_size=16)

    assert report["status"] == "fixture_smoke_completed"
    assert report["dataset_check"]["datasets"]["mvtec_ad"]["ready_categories"] == ["bottle"]
    assert report["calibration"]["status"] == "calibrated_phase0_threshold"
    assert report["evaluation"]["sample_count"] == 2
    assert report["result_validation"]["status"] == "passed"
    assert report["proof_note"].startswith("This exercises dataset loading")
    assert (tmp_path / "fixture_smoke_report.json").exists()
