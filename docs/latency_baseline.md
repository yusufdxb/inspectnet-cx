# Latency Baseline

Primary deployment target: the dev workstation (AMD Ryzen 9 9900X 12-Core Processor + NVIDIA Blackwell consumer GPU, x86_64 Linux).

## CUDA Baseline (dev workstation, 2026-05-17)

| image size | median ms/img | p95 ms/img |
| ---------- | ------------: | ---------: |
| 256 x 256  |         0.275 |      0.391 |
| 512 x 512  |         0.474 |      0.622 |

Device: NVIDIA Blackwell consumer GPU (CUDA).

## CPU Baseline (dev workstation, 2026-05-17)

| image size | median ms/img | p95 ms/img |
| ---------- | ------------: | ---------: |
| 256 x 256  |         0.685 |      0.894 |
| 512 x 512  |         2.956 |      3.217 |

Device: AMD Ryzen 9 9900X 12-Core Processor (CPU only).

## Hardware Fingerprint

- CPU: AMD Ryzen 9 9900X 12-Core Processor
- GPU: NVIDIA Blackwell consumer GPU
- Platform: Linux-6.8.0-111-generic-x86_64-with-glibc2.35
- Jetson: false (x86_64 workstation, not Jetson)

## Measurement Method

Per-iteration timing (perf_counter before and after model forward pass with CUDA sync on GPU
runs). Sorted timings for median and p95 calculation. Warmup: 10 runs. Measurement runs: 50.

## Future Hardware (Unmeasured)

Jetson Orin NX 16GB is the planned edge deployment target. No latency measurement has been
taken on Jetson hardware yet. When a Jetson session is available, run:

```bash
inspectnet-latency --device auto --image-size 512 --target-hardware jetson-orin-nx-16gb \
  --require-jetson --output reports/latency_jetson.json
```
