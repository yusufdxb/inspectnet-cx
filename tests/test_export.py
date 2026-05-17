from inspectnet_cx.export import check_export_readiness


def test_export_readiness_reports_missing_model_config(tmp_path):
    report = check_export_readiness(model_dir=tmp_path, export_format="onnx")

    assert report["status"] == "blocked"
    assert any("model config not found" in reason for reason in report["blocked_reasons"])


def test_openvino_readiness_reports_missing_source_onnx(tmp_path):
    report = check_export_readiness(
        export_format="openvino",
        onnx_path=tmp_path / "missing.onnx",
    )

    assert report["status"] == "blocked"
    assert any("source ONNX file is missing" in reason for reason in report["blocked_reasons"])
