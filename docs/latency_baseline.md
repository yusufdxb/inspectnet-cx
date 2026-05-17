# Latency Baseline

## CPU Baseline (mewtwo, 2026-05-17)

- Median latency per image: 0.9335 ms
- p95 latency per image: 1.1002 ms
- Mean latency per image: 0.9193 ms

### Hardware Fingerprint

- CPU: AMD Ryzen 9 9900X 12-Core Processor
- GPU: NVIDIA GeForce RTX 5070
- Jetson: false
- Platform: Linux-6.8.0-111-generic-x86_64-with-glibc2.35

### Test Configuration

- Image size: 256 x 256
- Batch size: 1
- Warmup runs: 5
- Measurement runs: 20

### Measurement Method

Per-iteration timing (perf_counter before and after model forward pass with device sync). Sorted timings for median and p95 calculation. Ready for Jetson Orin NX validation later.
