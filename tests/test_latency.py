from inspectnet_cx.eval.latency import benchmark_latency


def test_benchmark_latency_smoke_cpu():
    result = benchmark_latency(image_size=32, warmup=1, iterations=1, device="cpu")

    assert result["status"] == "local_phase0_latency"
    assert result["latency_ms_per_image"] > 0
    assert result["device"] == "cpu"


def test_benchmark_latency_can_require_jetson():
    result = benchmark_latency(
        image_size=32,
        warmup=1,
        iterations=1,
        device="cpu",
        target_hardware="jetson-orin-nx-16gb",
        require_jetson=True,
    )

    assert result["status"] in {"blocked", "local_phase0_latency"}
    if result["status"] == "blocked":
        assert "Jetson Orin NX" in result["blocked_reasons"][0]
