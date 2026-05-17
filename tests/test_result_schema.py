from inspectnet_cx.eval.result_schema import validate_result_payload
from inspectnet_cx.eval.validate_results import validate_results


def test_validate_result_payload_accepts_placeholder():
    payload = {
        "method": "patchcore",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "image_auroc": "TBD",
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": "TBD",
        "peak_vram_mb": "TBD",
        "model_size_mb": "TBD",
        "status": "phase0_placeholder",
    }

    assert validate_result_payload(payload) == []


def test_validate_result_payload_accepts_classical_numeric_baseline():
    payload = {
        "method": "classical-range",
        "baseline_kind": "classical_cpu_image",
        "baseline_version": "classical-range-v1",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "image_auroc": 1.0,
        "image_f1": 1.0,
        "accuracy": 1.0,
        "threshold": 0.5,
        "train_image_count": 2,
        "test_image_count": 2,
        "feature_count": 90,
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": 1.2,
        "peak_vram_mb": 0.0,
        "model_size_mb": 0.0,
        "metrics_scope": "image_level_only",
        "status": "completed_classical_cpu_baseline",
    }

    assert validate_result_payload(payload) == []


def test_validate_result_payload_accepts_patchdiff_classical_baseline():
    payload = {
        "method": "classical_patchdiff",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "train_normal_count": 2,
        "test_sample_count": 2,
        "test_normal_count": 1,
        "test_anomaly_count": 1,
        "threshold": 0.5,
        "image_auroc": 1.0,
        "image_f1": 1.0,
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": 1.2,
        "peak_vram_mb": 0.0,
        "model_size_mb": 0.1,
        "status": "classical_baseline_completed",
    }

    assert validate_result_payload(payload) == []


def test_validate_result_payload_accepts_anomalib_baseline():
    payload = {
        "method": "anomalib_padim",
        "baseline_kind": "anomalib",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "train_normal_count": 209,
        "test_sample_count": 83,
        "image_auroc": 1.0,
        "image_f1": 0.99,
        "pixel_auroc": 0.98,
        "au_pro": "TBD",
        "pixel_f1": 0.69,
        "latency_ms_per_image": "TBD",
        "peak_vram_mb": 0.0,
        "model_size_mb": "TBD",
        "fit_elapsed_s": 10.0,
        "test_elapsed_s": 5.0,
        "status": "completed_anomalib_baseline",
    }

    assert validate_result_payload(payload) == []


def test_validate_result_payload_rejects_patchdiff_missing_numeric_auroc():
    payload = {
        "method": "classical_patchdiff",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "train_normal_count": 2,
        "test_sample_count": 2,
        "test_normal_count": 1,
        "test_anomaly_count": 1,
        "threshold": 0.5,
        "image_auroc": "TBD",
        "image_f1": 1.0,
        "pixel_auroc": "TBD",
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": 1.2,
        "peak_vram_mb": 0.0,
        "model_size_mb": 0.1,
        "status": "classical_baseline_completed",
    }

    assert "image_auroc must be numeric when normal and anomaly tests exist" in (
        validate_result_payload(payload)
    )


def test_validate_result_payload_rejects_invalid_classical_numeric_baseline():
    payload = {
        "method": "classical-range",
        "baseline_kind": "classical_cpu_image",
        "baseline_version": "classical-range-v1",
        "dataset": "mvtec_ad",
        "category": "bottle",
        "image_auroc": 1.5,
        "image_f1": "TBD",
        "accuracy": 1.0,
        "threshold": 0.5,
        "train_image_count": 2,
        "test_image_count": 2,
        "feature_count": 90,
        "pixel_auroc": 0.7,
        "au_pro": "TBD",
        "pixel_f1": "TBD",
        "latency_ms_per_image": 1.2,
        "peak_vram_mb": 0.0,
        "model_size_mb": 0.0,
        "metrics_scope": "image_level_only",
        "status": "completed_classical_cpu_baseline",
    }

    errors = validate_result_payload(payload)

    assert "image_auroc must be between 0 and 1" in errors
    assert "image_f1 must be numeric for completed classical baselines" in errors
    assert "pixel_auroc must remain TBD for image-level classical baselines" in errors


def test_validate_result_payload_rejects_missing_field():
    assert validate_result_payload({"method": "patchcore"}) != []


def test_validate_results_skips_non_benchmark_proof_reports(tmp_path):
    (tmp_path / "jetson_latency_smoke.json").write_text(
        '{"status": "blocked", "target_hardware": "jetson-orin-nx-16gb"}'
    )
    (tmp_path / "fixture_smoke_report.json").write_text(
        '{"status": "fixture_smoke_completed", "proof_note": "not benchmark evidence"}'
    )

    assert validate_results(tmp_path) == {}
