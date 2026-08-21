"""Static and measured efficiency records for a forecasting model."""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, Sequence


def measure_efficiency(
    model: Any,
    input_length: int,
    num_features: int,
    batch_size: int,
    device: str,
    warmup_iterations: int,
    timed_iterations: int,
    use_timestamps: bool = False,
    timestamp_sizes: Sequence[int] | None = None,
) -> Dict[str, Any]:
    """Measure architecture cost before training; unsupported FLOPs remain null."""

    import torch

    model = model.to(device).eval()
    inputs = torch.randn(batch_size, input_length, num_features, device=device)
    kwargs: Dict[str, Any] = {}
    if use_timestamps:
        kwargs["inputs_timestamps"] = torch.zeros(
            batch_size,
            input_length,
            len(timestamp_sizes or []),
            device=device,
        )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    def synchronize() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(inputs, **kwargs)
        synchronize()
        latencies_ms = []
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for _ in range(timed_iterations):
            synchronize()
            start = time.perf_counter()
            model(inputs, **kwargs)
            synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    flops = None
    flops_error = None
    try:
        activity = torch.profiler.ProfilerActivity.CUDA if device.startswith("cuda") else torch.profiler.ProfilerActivity.CPU
        with torch.profiler.profile(activities=[activity], with_flops=True) as profile:
            with torch.inference_mode():
                model(inputs, **kwargs)
            synchronize()
        counted = sum(int(event.flops or 0) for event in profile.key_averages())
        flops = counted or None
    except Exception as exc:  # operator support differs across torch versions
        flops = None
        flops_error = f"{type(exc).__name__}: {exc}"

    median_latency = statistics.median(latencies_ms)
    ordered = sorted(latencies_ms)
    p90_index = min(len(ordered) - 1, int(0.9 * len(ordered)))
    peak_memory = torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
    return {
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "flops_per_batch": flops,
        "flops_error": flops_error,
        "benchmark_batch_size": batch_size,
        "latency_median_ms": median_latency,
        "latency_p90_ms": ordered[p90_index],
        "throughput_samples_per_second": batch_size * 1000.0 / median_latency,
        "benchmark_peak_cuda_bytes": peak_memory,
        "device": device,
        "warmup_iterations": warmup_iterations,
        "timed_iterations": timed_iterations,
    }
