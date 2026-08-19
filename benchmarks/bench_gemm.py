"""GEMM scaling across every backend nyangrad can reach, plus reference libraries.

Backends compared at each shape:

  numpy                  OpenBLAS through numpy, all cores
  torch_cpu_1t           single-threaded MKL/oneDNN, the honest peer for the
                         nyangrad CPU kernels, which are single-threaded
  torch_cpu_all          all cores, to show what threading alone is worth
  nyangrad_cpu_tiled     the 8x8 register-blocked C++ kernel, reached through
                         NDArray.__matmul__ when every dimension divides by the
                         tile size
  nyangrad_cpu_naive     the three-loop C++ kernel, called directly so the same
                         shape can be measured on both code paths
  nyangrad_cuda          the shared-memory + register-tiled CUDA kernel
  torch_cuda             cuBLAS fp32 with tf32 disabled, so it is the same
                         arithmetic the nyangrad kernel performs
"""
from __future__ import annotations

import numpy as np

import nyangrad as nyan
from nyangrad import ndarray as nd

from harness import (
    HAVE_TORCH,
    HAVE_TORCH_CUDA,
    L2Flusher,
    Record,
    bench,
    cuda_sync,
    gemm_flops,
    torch,
)

SQUARE = [256, 512, 1024, 2048, 4096]
RECTANGULAR = [
    (8192, 256, 256),    # tall and skinny
    (512, 2048, 2048),   # wide inner dimension
    (2048, 2048, 512),   # narrow output
]

# these kernels are too slow to run at every size; measured up to here only
NAIVE_MAX_N = 1024
TILED_MAX_N = 4096


def _sizes() -> list[tuple[int, int, int]]:
    return [(n, n, n) for n in SQUARE] + RECTANGULAR


def _record(case, backend, m, n, k, stats, ceilings, notes="") -> Record:
    flops = gemm_flops(m, n, k)
    seconds = stats.median_ms / 1e3
    gflops = flops / seconds / 1e9
    metrics = {
        "gflops": gflops,
        "flops": flops,
        # working set if nothing is reused: A + B + C
        "bytes_min_traffic": (m * k + k * n + m * n) * 4,
        "arithmetic_intensity": flops / ((m * k + k * n + m * n) * 4),
    }
    ceiling = None
    if backend.endswith("cuda") and ceilings.get("gpu_compute"):
        ceiling = ceilings["gpu_compute"]["gflops"]
    elif ceilings.get("cpu_compute"):
        # compare like with like: single-threaded kernels against the
        # single-threaded ceiling, threaded libraries against the threaded one
        threaded = backend in ("numpy", "torch_cpu_all")
        ceiling = ceilings["cpu_compute"][
            "gflops_all_threads" if threaded else "gflops_single_thread"
        ]
    if ceiling:
        metrics["pct_of_measured_ceiling"] = gflops / ceiling * 100
    return Record("gemm", case, backend, f"{m}x{k}x{n}", stats.to_dict(), metrics, notes)


def run(ceilings: dict) -> list[Record]:
    records: list[Record] = []
    flusher = L2Flusher()
    cpu_dev = nyan.cpu()
    cuda_dev = nyan.cuda()
    default_threads = torch.get_num_threads() if HAVE_TORCH else None

    for m, k, n in _sizes():
        case = f"{m}x{k}x{n}"
        square = m == k == n
        label = f"N={m}" if square else case
        print(f"  gemm {label:<18}", flush=True)

        a_np = np.random.randn(m, k).astype(np.float32)
        b_np = np.random.randn(k, n).astype(np.float32)

        # ---- numpy / OpenBLAS -------------------------------------------- #
        stats = bench(lambda: a_np @ b_np, budget_s=0.5)
        records.append(_record(case, "numpy", m, n, k, stats, ceilings,
                               "OpenBLAS, all cores"))

        # ---- torch cpu ---------------------------------------------------- #
        if HAVE_TORCH:
            a_t = torch.from_numpy(a_np)
            b_t = torch.from_numpy(b_np)
            for backend, threads in (("torch_cpu_1t", 1), ("torch_cpu_all", default_threads)):
                torch.set_num_threads(threads)
                stats = bench(lambda: torch.matmul(a_t, b_t), budget_s=0.5)
                records.append(_record(case, backend, m, n, k, stats, ceilings,
                                       f"{threads} thread(s)"))
            torch.set_num_threads(default_threads)

        # ---- nyangrad cpu ------------------------------------------------- #
        if cpu_dev.enabled():
            a_cpu = nd.array(a_np, device=cpu_dev)
            b_cpu = nd.array(b_np, device=cpu_dev)
            tile = cpu_dev.__tile_size__
            aligned = all(d % tile == 0 for d in (m, k, n))

            if aligned and max(m, k, n) <= TILED_MAX_N:
                # __matmul__ selects the tiled kernel for aligned shapes
                stats = bench(lambda: a_cpu @ b_cpu, budget_s=0.5, flush=flusher.flush_cpu)
                records.append(_record(case, "nyangrad_cpu_tiled", m, n, k, stats, ceilings,
                                       f"{tile}x{tile} register blocking, 1 thread"))

            if max(m, k, n) <= NAIVE_MAX_N:
                a_c = a_cpu.compact()
                b_c = b_cpu.compact()
                out = nd.NDArray.make((m, n), device=cpu_dev)

                def naive_cpu():
                    cpu_dev.matmul(a_c._handle, b_c._handle, out._handle, m, k, n)

                stats = bench(naive_cpu, budget_s=0.5, flush=flusher.flush_cpu)
                records.append(_record(case, "nyangrad_cpu_naive", m, n, k, stats, ceilings,
                                       "three-loop kernel, 1 thread"))

        # ---- nyangrad cuda ------------------------------------------------ #
        if cuda_dev.enabled():
            a_cu = nd.array(a_np, device=cuda_dev)
            b_cu = nd.array(b_np, device=cuda_dev)
            stats = bench(lambda: a_cu @ b_cu, sync=cuda_sync, budget_s=0.5)
            records.append(_record(case, "nyangrad_cuda", m, n, k, stats, ceilings,
                                   "shared memory + register tiling"))

        # ---- torch cuda / cuBLAS ------------------------------------------ #
        if HAVE_TORCH_CUDA:
            a_tc = torch.from_numpy(a_np).cuda()
            b_tc = torch.from_numpy(b_np).cuda()
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                stats = bench(lambda: torch.matmul(a_tc, b_tc), sync=cuda_sync, budget_s=0.5)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
            records.append(_record(case, "torch_cuda", m, n, k, stats, ceilings,
                                   "cuBLAS fp32, tf32 disabled"))
            del a_tc, b_tc
            torch.cuda.empty_cache()

    return records


def verify_correctness() -> list[dict]:
    """Confirm each backend computes the same GEMM before believing its timings."""
    checks = []
    m, k, n = 256, 256, 256
    a_np = np.random.randn(m, k).astype(np.float32)
    b_np = np.random.randn(k, n).astype(np.float32)
    reference = a_np @ b_np

    for name, device in (("nyangrad_cpu", nyan.cpu()), ("nyangrad_cuda", nyan.cuda())):
        if not device.enabled():
            continue
        got = (nd.array(a_np, device=device) @ nd.array(b_np, device=device)).numpy()
        err = float(np.abs(got - reference).max())
        rel = err / float(np.abs(reference).max())
        checks.append({
            "backend": name,
            "max_abs_err": err,
            "max_rel_err": rel,
            "passed": rel < 1e-4,
        })
    return checks
