.PHONY: setup setup-all lint test build baseline-patchcore baseline-efficientad baseline-anomalib-padim baseline-classical-range baseline-classical-fixture eval-mvtec eval-visa proof-readiness dataset-check fixture-smoke validate-results export-check export-onnx calibrate-threshold latency jetson-latency create-phase0-model release-smoke clean-generated space

PYTHON ?= python3

setup:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.10 && uv pip install -e ".[dev]"; \
	else \
		$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"; \
	fi

setup-all:
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.10 && uv pip install -e ".[all]"; \
	else \
		$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -e ".[all]"; \
	fi

lint:
	ruff check src tests scripts

test:
	pytest -q

build:
	$(PYTHON) -m build

baseline-patchcore:
	$(PYTHON) scripts/run_baseline.py --method patchcore --dataset mvtec_ad --category bottle --device cpu --output reports/patchcore_mvtec_ad_bottle.json

baseline-efficientad:
	$(PYTHON) scripts/run_baseline.py --method efficientad --dataset mvtec_ad --category bottle --device cpu --output reports/efficientad_mvtec_ad_bottle.json

baseline-anomalib-padim:
	$(PYTHON) scripts/run_anomalib_baseline.py --method padim --dataset mvtec_ad --category bottle --dataset-root ~/datasets --device cpu --output reports/verification/anomalib_padim_mvtec_ad_bottle_result.json --work-dir artifacts/verification/anomalib

baseline-classical-range:
	$(PYTHON) scripts/run_baseline.py --method classical-range --dataset mvtec_ad --category bottle --data-root ~/datasets --output reports/baseline_classical_range_v1/mvtec_ad_bottle/result.json

baseline-classical-fixture:
	$(PYTHON) scripts/run_classical_baseline.py --dataset-root reports/fixture_smoke_classical/datasets --dataset mvtec_ad --category bottle --image-size 32 --quantile 0.5 --output reports/classical_patchdiff_fixture.json

eval-mvtec:
	$(PYTHON) scripts/aggregate_results.py --input reports --output reports/baselines.md

eval-visa:
	$(PYTHON) scripts/run_baseline.py --method patchcore --dataset visa --category candle --device cpu --output reports/patchcore_visa_candle.json

proof-readiness:
	$(PYTHON) scripts/check_proof_readiness.py --output reports/proof_readiness.json

dataset-check:
	$(PYTHON) scripts/check_datasets.py --output reports/dataset_check.json

fixture-smoke:
	$(PYTHON) scripts/run_fixture_smoke.py --output-dir reports/fixture_smoke --image-size 32

export-check:
	$(PYTHON) scripts/export_phase0.py --check-only --format onnx

export-onnx:
	$(PYTHON) scripts/export_phase0.py --format onnx --model artifacts/inspectnet-cx-phase0 --output artifacts/inspectnet-cx-phase0/model.onnx --verify

calibrate-threshold:
	$(PYTHON) scripts/calibrate_normal_threshold.py --model artifacts/inspectnet-cx-phase0 --dataset mvtec_ad --category bottle --dataset-root ~/datasets --output reports/normal_threshold.json

validate-results:
	$(PYTHON) scripts/validate_results.py --input reports

latency:
	$(PYTHON) scripts/benchmark_latency.py --device auto --image-size 512 --warmup 5 --iterations 20 --output reports/local_latency.json

jetson-latency:
	$(PYTHON) scripts/benchmark_latency.py --device auto --target-hardware jetson-orin-nx-16gb --require-jetson --image-size 512 --warmup 10 --iterations 50 --output reports/jetson_latency.json

create-phase0-model:
	$(PYTHON) scripts/create_phase0_model.py --output artifacts/inspectnet-cx-phase0 --image-size 224

release-smoke:
	$(PYTHON) scripts/release_smoke.py

clean-generated:
	rm -rf artifacts build dist src/inspectnet_cx.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

space:
	$(PYTHON) -m inspectnet_cx.space
