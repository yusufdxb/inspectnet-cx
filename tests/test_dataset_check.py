from inspectnet_cx.data import check_datasets


def test_check_datasets_reports_missing_paths(tmp_path):
    report = check_datasets(tmp_path)

    assert report["status"] == "blocked"
    assert report["datasets"]["mvtec_ad"]["exists"] is False


def test_check_datasets_discovers_mvtec_ready_category(tmp_path):
    image = tmp_path / "mvtec_ad" / "bottle" / "train" / "good" / "000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not-a-real-image-but-counted-by-extension")

    report = check_datasets(tmp_path)

    assert report["datasets"]["mvtec_ad"]["status"] == "ready_for_local_checks"
    assert report["datasets"]["mvtec_ad"]["ready_categories"] == ["bottle"]
    assert "inspectnet-baseline --plan-only" in report["datasets"]["mvtec_ad"]["next_actions"][0]
