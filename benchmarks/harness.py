"""Timing core and hardware telemetry for the nyangrad benchmark suite.

Methodology, following what is standard for kernel benchmarking (CUTLASS's GEMM
measurement guidelines, triton's do_bench, tinygrad's speed tests):

- Launches on the GPU are asynchronous, so a host timer wrapped around a single
  launch measures queue latency rather than device time. Every measurement here
  enqueues a batch of repetitions and synchronises the device once at the end,
  then divides by the batch size. The batch is sized so each sample lands around
  `TARGET_SAMPLE_MS`, which keeps launch and synchronise overhead below the noise
  floor without hiding it entirely.
- Warmup repetitions are run and discarded, to absorb one-time costs: driver and
  context init, cuBLAS algorithm selection, allocator growth, and GPU clock ramp.
  A floor of `MIN_WARMUP` repetitions applies regardless of what the calibration
  probe suggested, because a threaded BLAS whose pool is still spinning up gives a
  slow probe, which would otherwise buy a short warmup and more slow samples.
- The batch size is recalculated from a second probe taken after warmup, so it
  reflects warm performance rather than cold.
- The median is reported rather than the mean, since latency distributions here
  are heavy tailed. p10/p90 and the IQR come along so the spread is visible.
- A sample set whose IQR exceeds `STABLE_REL_IQR` of its median was contaminated
  by something outside the measurement, so it is taken again, up to
  `MAX_ATTEMPTS` times, keeping the most stable attempt rather than the fastest.
  Anything still unstable is flagged in the record instead of being presented as
  though it were solid.
- Anything whose first repetition is slower than `SINGLE_SHOT_S` is measured once
  and flagged, instead of being given a repetition budget it cannot afford. The
  naive CPU matmul at large N is the reason this exists.
- SM clock, temperature and power are sampled on a background thread for the
  whole run, so throttling during a measurement is visible after the fact rather
  than silently folded into the numbers.
"""
from __future__ import annotations

import json
import math
import platform
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parent.parent

TARGET_SAMPLE_MS = 5.0
SINGLE_SHOT_S = 2.0
MIN_WARMUP = 10
# a threaded BLAS at these sizes genuinely jitters by this much between batches,
# so anything under it is the machine rather than the measurement
STABLE_REL_IQR = 0.25
MAX_ATTEMPTS = 3

try:
    import torch

    HAVE_TORCH = True
    HAVE_TORCH_CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover - torch is optional for the suite
    torch = None
    HAVE_TORCH = False
    HAVE_TORCH_CUDA = False


# --------------------------------------------------------------------------- #
# synchronisation and cache control
# --------------------------------------------------------------------------- #

def cuda_sync() -> None:
    """Block until every queued kernel on the device has finished.

    torch's synchronise is device wide, so it covers kernels launched from the
    nyangrad extension too, not just torch's own.
    """
    if HAVE_TORCH_CUDA:
        torch.cuda.synchronize()


class L2Flusher:
    """Overwrite a buffer larger than L2 so the next read comes from DRAM.

    Memory-bound measurements are otherwise reading a warm cache and reporting
    a bandwidth the hardware cannot actually sustain.
    """

    def __init__(self) -> None:
        self.gpu = None
        self.cpu = None
        if HAVE_TORCH_CUDA:
            l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 4 << 20)
            self.gpu = torch.empty(int(l2 * 2), dtype=torch.uint8, device="cuda")
        if HAVE_TORCH:
            self.cpu = torch.empty(64 << 20, dtype=torch.uint8)

    def flush_gpu(self) -> None:
        if self.gpu is not None:
            self.gpu.zero_()

    def flush_cpu(self) -> None:
        if self.cpu is not None:
            self.cpu.zero_()


# --------------------------------------------------------------------------- #
# the timer
# --------------------------------------------------------------------------- #

@dataclass
class Stats:
    """Timing distribution for one benchmark configuration."""

    median_ms: float
    min_ms: float
    mean_ms: float
    p10_ms: float
    p90_ms: float
    stdev_ms: float
    iqr_ms: float
    reps: int
    samples: int
    inner: int
    warmup: int
    wall_s: float
    single_shot: bool = False
    attempts: int = 1
    unstable: bool = False

    @property
    def rel_iqr(self) -> float:
        return self.iqr_ms / self.median_ms if self.median_ms else 0.0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["rel_iqr"] = self.rel_iqr
        return out


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def bench(
    fn: Callable[[], Any],
    *,
    sync: Optional[Callable[[], None]] = None,
    flush: Optional[Callable[[], None]] = None,
    budget_s: float = 0.5,
    min_samples: int = 7,
    max_samples: int = 200,
    warmup_budget_s: float = 0.05,
) -> Stats:
    """Time fn() and return the distribution of per-repetition latency.

    Repeats the whole measurement if the sample spread says something interfered,
    and keeps the most stable attempt.
    """
    best: Optional[Stats] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        stats = _bench_once(
            fn, sync=sync, flush=flush, budget_s=budget_s,
            min_samples=min_samples, max_samples=max_samples,
            warmup_budget_s=warmup_budget_s,
        )
        stats.attempts = attempt
        if best is None or stats.rel_iqr < best.rel_iqr:
            best = stats
        if stats.single_shot or stats.rel_iqr <= STABLE_REL_IQR:
            break
    assert best is not None
    best.unstable = not best.single_shot and best.rel_iqr > STABLE_REL_IQR
    return best


def _bench_once(
    fn: Callable[[], Any],
    *,
    sync: Optional[Callable[[], None]] = None,
    flush: Optional[Callable[[], None]] = None,
    budget_s: float = 0.5,
    min_samples: int = 7,
    max_samples: int = 200,
    warmup_budget_s: float = 0.05,
) -> Stats:
    """One measurement: calibrate, warm up, then collect batched samples."""
    sync = sync or (lambda: None)
    flush = flush or (lambda: None)
    wall_start = time.perf_counter()

    # An untimed call first. The one-time costs are large and varied - cuBLAS
    # picking an algorithm, a kernel module loading, first-touch page faults on a
    # freshly allocated output - and timing them would report a cold number as
    # though it were steady state.
    fn()
    sync()

    flush()
    sync()
    probe_start = time.perf_counter()
    fn()
    sync()
    probe_s = time.perf_counter() - probe_start

    if probe_s > SINGLE_SHOT_S:
        # too expensive to repeat; report the single warm observation honestly
        ms = probe_s * 1e3
        return Stats(
            median_ms=ms, min_ms=ms, mean_ms=ms, p10_ms=ms, p90_ms=ms,
            stdev_ms=0.0, iqr_ms=0.0, reps=1, samples=1, inner=1, warmup=1,
            wall_s=time.perf_counter() - wall_start, single_shot=True,
        )

    probe_ms = max(probe_s * 1e3, 1e-6)
    warmup = max(MIN_WARMUP, int(warmup_budget_s * 1e3 / probe_ms))

    for _ in range(warmup):
        fn()
    sync()

    # size the batch from warm performance, not from the cold probe
    warm_start = time.perf_counter()
    fn()
    sync()
    warm_ms = max((time.perf_counter() - warm_start) * 1e3, 1e-6)
    inner = max(1, int(TARGET_SAMPLE_MS / warm_ms))

    per_rep_ms: list[float] = []
    while True:
        flush()
        sync()
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        sync()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        per_rep_ms.append(elapsed_ms / inner)

        spent = time.perf_counter() - wall_start
        if len(per_rep_ms) >= max_samples:
            break
        if len(per_rep_ms) >= min_samples and spent > budget_s:
            break

    return Stats(
        median_ms=statistics.median(per_rep_ms),
        min_ms=min(per_rep_ms),
        mean_ms=statistics.fmean(per_rep_ms),
        p10_ms=_percentile(per_rep_ms, 0.10),
        p90_ms=_percentile(per_rep_ms, 0.90),
        stdev_ms=statistics.stdev(per_rep_ms) if len(per_rep_ms) > 1 else 0.0,
        iqr_ms=_percentile(per_rep_ms, 0.75) - _percentile(per_rep_ms, 0.25),
        reps=len(per_rep_ms) * inner,
        samples=len(per_rep_ms),
        inner=inner,
        warmup=warmup,
        wall_s=time.perf_counter() - wall_start,
    )


# --------------------------------------------------------------------------- #
# hardware and software inventory
# --------------------------------------------------------------------------- #

def _nvidia_smi(query: str) -> Optional[str]:
    if not shutil.which("nvidia-smi"):
        return None
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip().splitlines()[0]


def cpu_name() -> str:
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def cpu_topology() -> dict:
    physical = None
    info = Path("/proc/cpuinfo")
    if info.exists():
        cores = re.findall(r"^cpu cores\s*:\s*(\d+)", info.read_text(), re.M)
        if cores:
            physical = int(cores[0])
    import os

    return {"logical": os.cpu_count(), "physical": physical}


def numpy_blas() -> str:
    import numpy

    try:
        cfg = numpy.__config__.show(mode="dicts")
        build = cfg.get("Build Dependencies", {}).get("blas", {})
        name = build.get("name", "unknown")
        return f"{name} {build.get('version', '')}".strip()
    except Exception:
        return "unknown"


def collect_environment() -> dict:
    import numpy

    env: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": {"version": numpy.__version__, "blas": numpy_blas()},
        "cpu": {"name": cpu_name(), **cpu_topology()},
    }

    if HAVE_TORCH:
        env["torch"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "default_threads": torch.get_num_threads(),
        }

    if HAVE_TORCH_CUDA:
        props = torch.cuda.get_device_properties(0)
        sm_clock_mhz = _nvidia_smi("clocks.max.sm")
        mem_clock_mhz = _nvidia_smi("clocks.max.mem")
        # consumer Ampere (GA10x) carries 128 fp32 lanes per SM
        lanes_per_sm = 128
        peak_tflops = None
        if sm_clock_mhz:
            peak_tflops = (
                props.multi_processor_count * lanes_per_sm * 2 * float(sm_clock_mhz) * 1e6 / 1e12
            )
        env["gpu"] = {
            "name": props.name,
            "sm_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / 1e9, 2),
            "l2_cache_mb": round(getattr(props, "L2_cache_size", 0) / 1e6, 2),
            "max_sm_clock_mhz": float(sm_clock_mhz) if sm_clock_mhz else None,
            "max_mem_clock_mhz": float(mem_clock_mhz) if mem_clock_mhz else None,
            "driver": _nvidia_smi("driver_version"),
            "spec_fp32_tflops_at_max_boost": round(peak_tflops, 2) if peak_tflops else None,
        }

    try:
        import nyangrad

        env["nyangrad"] = {
            "cpu_backend_built": nyangrad.cpu().enabled(),
            "cuda_backend_built": nyangrad.cuda().enabled(),
            "cpu_tile_size": nyangrad.cpu().__tile_size__ if nyangrad.cpu().enabled() else None,
            "cuda_tile_size": nyangrad.cuda().__tile_size__ if nyangrad.cuda().enabled() else None,
        }
    except Exception as exc:  # pragma: no cover
        env["nyangrad"] = {"error": repr(exc)}

    return env


# --------------------------------------------------------------------------- #
# clock / thermal monitoring
# --------------------------------------------------------------------------- #

class ClockMonitor:
    """Poll GPU clocks, temperature and power on a background thread.

    Lets the report state whether clocks were stable across the run instead of
    assuming they were.
    """

    FIELDS = "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"

    # polling spawns nvidia-smi, so keep it infrequent enough that it cannot
    # perturb a threaded CPU measurement while still catching a throttling event
    def __init__(self, interval_s: float = 1.0) -> None:
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.available = shutil.which("nvidia-smi") is not None and HAVE_TORCH_CUDA

    def _poll(self) -> None:
        start = time.perf_counter()
        while not self._stop.is_set():
            row = _nvidia_smi(self.FIELDS)
            if row:
                parts = [p.strip() for p in row.split(",")]
                try:
                    self.samples.append({
                        "t": round(time.perf_counter() - start, 3),
                        "sm_mhz": float(parts[0]),
                        "mem_mhz": float(parts[1]),
                        "temp_c": float(parts[2]),
                        "power_w": float(parts[3]),
                        "util_pct": float(parts[4]),
                    })
                except (ValueError, IndexError):
                    pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ClockMonitor":
        if self.available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict:
        if not self.samples:
            return {"available": False}
        sm = [s["sm_mhz"] for s in self.samples]
        temp = [s["temp_c"] for s in self.samples]
        power = [s["power_w"] for s in self.samples]
        return {
            "available": True,
            "samples": len(self.samples),
            "sm_mhz": {
                "min": min(sm), "max": max(sm),
                "median": statistics.median(sm),
                "spread_pct": round((max(sm) - min(sm)) / max(sm) * 100, 1),
            },
            "temp_c": {"min": min(temp), "max": max(temp)},
            "power_w": {"min": min(power), "max": max(power), "max_observed": max(power)},
            "trace": self.samples,
        }


# --------------------------------------------------------------------------- #
# results plumbing
# --------------------------------------------------------------------------- #

@dataclass
class Record:
    """One benchmark observation, flat enough to land straight in a table."""

    suite: str
    case: str
    backend: str
    shape: str
    stats: dict
    metrics: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def gemm_flops(m: int, n: int, k: int) -> float:
    """2mnk, the conventional GEMM flop count (one multiply + one add per term)."""
    return 2.0 * m * n * k


def throughput_metrics(flops: float, seconds: float) -> dict:
    return {
        "gflops": flops / seconds / 1e9,
        "flops": flops,
    }


def bandwidth_metrics(bytes_moved: float, seconds: float) -> dict:
    return {
        "gb_per_s": bytes_moved / seconds / 1e9,
        "bytes": bytes_moved,
    }


def write_results(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
