import importlib.util
from pathlib import Path


def test_scripts_import():
    for script in (
        "aggregate_results.py",
        "benchmark_latency.py",
        "calibrate_normal_threshold.py",
        "check_datasets.py",
        "check_hf_package.py",
        "check_proof_readiness.py",
        "create_phase0_model.py",
        "export_phase0.py",
        "investigate_anomalib_export.py",
        "investigate_openvino_parity.py",
        "predict_anomaly.py",
        "run_anomalib_baseline.py",
        "run_baseline.py",
        "run_fixture_smoke.py",
        "run_inference.py",
        "validate_padim_export.py",
        "validate_results.py",
    ):
        path = Path("scripts") / script
        spec = importlib.util.spec_from_file_location(script, path)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
