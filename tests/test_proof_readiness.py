from inspectnet_cx.eval.proof_readiness import build_readiness_report


def test_proof_readiness_reports_blockers(tmp_path):
    report = build_readiness_report(dataset_root=tmp_path)

    assert report["status"] in {"ready", "blocked"}
    assert "packages" in report
    assert "datasets" in report
    assert "export_readiness" in report
    assert "dependency_readiness" in report
    assert "commands_blocked" in report["dependency_readiness"]["anomalib"]
    onnx_readiness = report["dependency_readiness"]["onnx"]
    if onnx_readiness["installed"]:
        assert "inspectnet-export --format onnx --verify" in onnx_readiness["commands_enabled"]
    else:
        assert "inspectnet-export --format onnx --verify" in onnx_readiness["commands_blocked"]
    assert "blocked_reasons" in report
    assert report["datasets"]["mvtec_ad"]["exists"] is False
