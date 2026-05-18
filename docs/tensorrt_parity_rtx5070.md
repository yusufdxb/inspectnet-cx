# TensorRT Parity on RTX 5070 (Sprint 3, BLOCKED)

Attempted: 2026-05-17 on mewtwo (AMD Ryzen 9 9900X + NVIDIA RTX 5070, CUDA 12, sm_120).

Outcome: **BLOCKED, no parity claim made.**

## What blocked it

- `python -c 'import tensorrt'`: `ModuleNotFoundError: No module named 'tensorrt'`
- `python -c 'import polygraphy'`: `ModuleNotFoundError: No module named 'polygraphy'`
- `which trtexec`: not found.
- `dpkg -l | grep -iE 'tensorrt|nvinfer'`: empty.

Neither the Python TensorRT bindings nor `trtexec` nor `polygraphy` is
installed on mewtwo. The Sprint 3 hard constraint forbids installing
new system dependencies for this deliverable. Installing TensorRT 10.x
for sm_120 (Blackwell) also requires the NVIDIA Container Toolkit or a
matching CUDA-12-aligned `.deb` set; that is a non-trivial setup that
cannot be completed safely inside the 90 min wall-clock budget.

## What this means for the project

- No TensorRT FP32 vs ORT FP32 parity numbers exist yet.
- The OpenVINO FP32 parity work (`docs/openvino_parity_resolution.md`)
  remains the only export-parity evidence.
- RTX 5070 CUDA inference is measured via PyTorch native (see
  `docs/latency_baseline.md` and `reports/latency_mewtwo.json`); that
  is **not** a TensorRT measurement.

## Unblock path

When a TensorRT install is available, the planned parity protocol is
identical to the OpenVINO investigation:

1. Export the trained PaDiM ONNX with the same dynamic axes used for
   OpenVINO.
2. Build a TensorRT FP32 engine via `trtexec --onnx=... --fp16=disable`
   or `polygraphy convert`.
3. Run the same 83 MVTec AD bottle test images through ORT FP32 and
   TRT FP32 and report per-pixel max abs diff on the anomaly map and
   any binary mask label flips at the deployment threshold.
4. Write the result as a JSON next to
   `reports/agent_b/openvino_parity_investigation.json`.

Sprint 3 closes without TensorRT evidence.
