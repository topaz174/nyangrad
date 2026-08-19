"""Measure the machine's practical compute and bandwidth ceilings.

Spec sheet numbers assume a boost clock the card will not hold and a memory
efficiency nothing achieves, so every "percent of peak" in this suite is quoted
against what a mature library actually reaches on this machine, measured here.
The spec figures are still recorded for reference.

These run before anything else, which means they run on an idle GPU sitting at
its lowest clock. A card takes a few seconds of sustained load to settle at its
working frequency, so a ceiling measured cold understates the machine and would
flatter everything compared against it. `settle_clocks` burns that time first.
"""
from __future__ import annotations

import time

import numpy as np

from harness import (
    HAVE_TORCH,
    HAVE_TORCH_CUDA,
    bench,
    cuda_sync,
    gemm_flops,
    torch,
)

SETTLE_S = 5.0


def settle_clocks(seconds: float = SETTLE_S) -> dict:
    """Hold the GPU under load until its clocks stop climbing.

    Returns what the clock did, so the report can show the ramp really happened
    rather than asserting it.
    """
    if not HAVE_TORCH_CUDA:
        return {"applied": False}
    from harness import _nvidia_smi

    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float32)
    before = _nvidia_smi("clocks.sm")
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for _ in range(4):
            torch.matmul(a, b)
        cuda_sync()
    after = _nvidia_smi("clocks.sm")
    del a, b
    torch.cuda.empty_cache()
    return {
        "applied": True,
        "seconds": seconds,
        "sm_mhz_before": float(before) if before else None,
        "sm_mhz_after": float(after) if after else None,
    }


def gpu_compute_ceiling(n: int = 4096) -> dict | None:
    """Best fp32 GEMM throughput cuBLAS reaches here."""
    if not HAVE_TORCH_CUDA:
        return None
    a = torch.randn(n, n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, n, device="cuda", dtype=torch.float32)
    # keep tf32 off: the nyangrad kernel is true fp32, so the baseline must be too
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        stats = bench(lambda: torch.matmul(a, b), sync=cuda_sync, budget_s=2.0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return {
        "source": f"torch.matmul fp32 {n}^3 (tf32 disabled)",
        "gflops": gemm_flops(n, n, n) / (stats.median_ms / 1e3) / 1e9,
    }


def gpu_bandwidth_ceiling(elements: int = 64 << 20) -> dict | None:
    """Streaming bandwidth from a large device-to-device copy (read + write)."""
    if not HAVE_TORCH_CUDA:
        return None
    src = torch.empty(elements, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    nbytes = src.numel() * src.element_size() * 2  # one read + one write
    stats = bench(lambda: dst.copy_(src), sync=cuda_sync, budget_s=2.0)
    return {
        "source": f"torch device-to-device copy of {elements / 1e6:.0f}M fp32",
        "gb_per_s": nbytes / (stats.median_ms / 1e3) / 1e9,
    }


def cpu_compute_ceiling(n: int = 2048) -> dict | None:
    """Best fp32 GEMM throughput on the CPU, single thread and all threads.

    The single-thread number is the meaningful comparison for the nyangrad CPU
    kernels, which do not thread.
    """
    if not HAVE_TORCH:
        return None
    a = torch.randn(n, n, dtype=torch.float32)
    b = torch.randn(n, n, dtype=torch.float32)
    flops = gemm_flops(n, n, n)
    default_threads = torch.get_num_threads()

    result = {"source": f"torch.matmul fp32 {n}^3"}
    for label, threads in (("single_thread", 1), ("all_threads", default_threads)):
        torch.set_num_threads(threads)
        stats = bench(lambda: torch.matmul(a, b), budget_s=0.75)
        result[f"gflops_{label}"] = flops / (stats.median_ms / 1e3) / 1e9
        result[f"threads_{label}"] = threads
    torch.set_num_threads(default_threads)
    return result


def cpu_bandwidth_ceiling(elements: int = 32 << 20) -> dict:
    """Streaming bandwidth from a large host copy (read + write)."""
    src = np.empty(elements, dtype=np.float32)
    dst = np.empty_like(src)
    nbytes = src.nbytes * 2
    stats = bench(lambda: np.copyto(dst, src), budget_s=0.75)
    return {
        "source": f"numpy copyto of {elements / 1e6:.0f}M fp32",
        "gb_per_s": nbytes / (stats.median_ms / 1e3) / 1e9,
    }


def measure_all() -> dict:
    return {
        "clock_settle": settle_clocks(),
        "gpu_compute": gpu_compute_ceiling(),
        "gpu_bandwidth": gpu_bandwidth_ceiling(),
        "cpu_compute": cpu_compute_ceiling(),
        "cpu_bandwidth": cpu_bandwidth_ceiling(),
    }
