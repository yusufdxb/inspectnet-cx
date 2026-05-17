import importlib


def test_package_modules_import():
    modules = [
        "inspectnet_cx",
        "inspectnet_cx.data.dataset_check",
        "inspectnet_cx.inference",
        "inspectnet_cx.models",
        "inspectnet_cx.release",
        "inspectnet_cx.release.create_phase0_model",
        "inspectnet_cx.data",
        "inspectnet_cx.calibration",
        "inspectnet_cx.eval",
        "inspectnet_cx.eval.aggregate",
        "inspectnet_cx.eval.baseline",
        "inspectnet_cx.eval.fixture_smoke",
        "inspectnet_cx.eval.latency",
        "inspectnet_cx.eval.proof_readiness",
        "inspectnet_cx.eval.result_schema",
        "inspectnet_cx.eval.validate_results",
        "inspectnet_cx.export",
        "inspectnet_cx.space",
    ]

    for module in modules:
        assert importlib.import_module(module)
