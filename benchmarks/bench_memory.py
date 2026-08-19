"""Cost of turning a strided view back into contiguous memory.

Every view operation in nyangrad (permute, broadcast, slice) is free: it only
rewrites shape/stride/offset metadata. The bill arrives when a kernel needs
contiguous input and `compact()` has to walk the strided layout. That kernel is
pure memory traffic with no arithmetic to hide it, so it is the clearest look at
how the backends behave when they are bandwidth-bound rather than compute-bound.

Each case is compared against numpy's `ascontiguousarray` and torch's
`.contiguous()` on the identical layout. The last case is a single streaming pass
over already-compact memory, run as `x + 0.0` on every backend, which gives each
one's own bandwidth ceiling to be judged against.

Bytes are counted as the minimum traffic the operation requires: one read of the
source elements plus one write of the destination. Broadcasting therefore reads
almost nothing, which is the honest accounting - a stride-0 row stays in cache.
Where a backend moves more than the minimum (strided reads pulling whole cache
lines it will not use) that shows up as a low effective bandwidth, which is the
interesting part.
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
    torch,
)

# (label, base shape, how to build the view, note)
CASES = [
    ("transpose_2d_1k", (1024, 1024), "transpose", "2D transpose, stride-1 reads become strided"),
    ("transpose_2d_2k", (2048, 2048), "transpose", "2D transpose at 16 MB"),
    ("transpose_2d_4k", (4096, 4096), "transpose", "2D transpose at 64 MB, far past L2"),
    ("nhwc_to_nchw", (64, 56, 56, 64), "nhwc", "4D permute, the layout change a conv needs"),
    ("broadcast_row", (1, 4096), "broadcast", "stride-0 read expanded to 4096x4096"),
    ("strided_slice", (4096, 4096), "slice", "every other element on both axes"),
    ("elementwise_stream", (4096, 4096), "identity", "one streaming pass over compact memory"),
]


def _build_views(kind: str, shape: tuple[int, ...], base_np: np.ndarray):
    """Return (numpy_view, nyangrad_view_fn, torch_view_fn, out_elems, src_elems)."""
    if kind == "transpose":
        return (
            base_np.T,
            lambda arr: arr.permute((1, 0)),
            lambda t: t.t(),
            base_np.size,
            base_np.size,
        )
    if kind == "nhwc":
        return (
            base_np.transpose(0, 3, 1, 2),
            lambda arr: arr.permute((0, 3, 1, 2)),
            lambda t: t.permute(0, 3, 1, 2),
            base_np.size,
            base_np.size,
        )
    if kind == "broadcast":
        target = (4096, 4096)
        return (
            np.broadcast_to(base_np, target),
            lambda arr: arr.broadcast_to(target),
            lambda t: t.expand(target),
            target[0] * target[1],
            base_np.size,  # a stride-0 read touches only the source row
        )
    if kind == "slice":
        np_view = base_np[::2, ::2]
        return (
            np_view,
            lambda arr: arr[::2, ::2],
            lambda t: t[::2, ::2],
            np_view.size,
            np_view.size,
        )
    if kind == "identity":
        return base_np, lambda arr: arr, lambda t: t, base_np.size, base_np.size
    raise ValueError(kind)


def _kernel_only(device, view, compact_already: bool):
    """Return a callable that runs just the kernel, into a buffer allocated once.

    `compact()` and the elementwise ops both allocate their output on every call,
    so timing them measures kernel plus allocator. Reaching past them to the
    backend entry point with a reused output isolates the kernel itself.
    """
    out = nd.NDArray.make(view.shape, device=device)
    if compact_already:
        source = view.compact()._handle
        return lambda: device.scalar_add(source, 0.0, out._handle)
    return lambda: device.compact(
        view._handle, out._handle, view.shape, view.strides, view._offset
    )


def _record(case, backend, shape, stats, out_elems, src_elems, ceilings, notes) -> Record:
    seconds = stats.median_ms / 1e3
    # minimum required traffic: read the source, write the destination
    nbytes = (out_elems + src_elems) * 4
    gbs = nbytes / seconds / 1e9
    metrics = {
        "gb_per_s": gbs,
        "bytes": nbytes,
        "elements": out_elems,
        "us_per_million_elements": stats.median_ms * 1e3 / (out_elems / 1e6),
    }
    if "cuda" in backend and ceilings.get("gpu_bandwidth"):
        metrics["pct_of_measured_ceiling"] = gbs / ceilings["gpu_bandwidth"]["gb_per_s"] * 100
    elif ceilings.get("cpu_bandwidth"):
        metrics["pct_of_measured_ceiling"] = gbs / ceilings["cpu_bandwidth"]["gb_per_s"] * 100
    return Record("memory", case, backend, shape, stats.to_dict(), metrics, notes)


def run(ceilings: dict) -> list[Record]:
    records: list[Record] = []
    flusher = L2Flusher()
    cpu_dev = nyan.cpu()
    cuda_dev = nyan.cuda()

    for label, shape, kind, note in CASES:
        print(f"  memory {label:<18}", flush=True)
        base_np = np.random.randn(*shape).astype(np.float32)
        np_view, make_nyan_view, make_torch_view, out_elems, src_elems = _build_views(
            kind, shape, base_np
        )
        shape_str = "x".join(str(s) for s in np_view.shape)
        compact_already = kind == "identity"

        def record(backend, stats, notes=note):
            records.append(
                _record(label, backend, shape_str, stats, out_elems, src_elems, ceilings, notes)
            )

        # ---- numpy -------------------------------------------------------- #
        # ascontiguousarray is a no-op on compact input, so the streaming case
        # has to be an operation that really touches every element
        numpy_op = (lambda: np_view + 0.0) if compact_already \
            else (lambda: np.ascontiguousarray(np_view))
        stats = bench(numpy_op, budget_s=0.4, flush=flusher.flush_cpu)
        record("numpy", stats)

        # ---- torch cpu ---------------------------------------------------- #
        if HAVE_TORCH:
            base_t = torch.from_numpy(base_np)
            view_t = make_torch_view(base_t)
            torch_op = (lambda: view_t + 0.0) if compact_already \
                else (lambda: view_t.contiguous())
            stats = bench(torch_op, budget_s=0.4, flush=flusher.flush_cpu)
            record("torch_cpu", stats)

        # ---- nyangrad cpu ------------------------------------------------- #
        if cpu_dev.enabled():
            base_cpu = nd.array(base_np, device=cpu_dev)
            view_cpu = make_nyan_view(base_cpu)
            # compact() short-circuits on compact input, so the streaming case
            # goes through the elementwise kernel instead
            cpu_op = (lambda: view_cpu + 0.0) if compact_already \
                else (lambda: view_cpu.compact())
            stats = bench(cpu_op, budget_s=0.4, flush=flusher.flush_cpu)
            record("nyangrad_cpu", stats)

            # again with the output buffer preallocated, which separates the
            # kernel from the per-op allocation it normally pays for
            stats = bench(_kernel_only(cpu_dev, view_cpu, compact_already),
                          budget_s=0.4, flush=flusher.flush_cpu)
            record("nyangrad_cpu_kernel", stats, note + " (output preallocated)")

        # ---- nyangrad cuda ------------------------------------------------ #
        if cuda_dev.enabled():
            base_cu = nd.array(base_np, device=cuda_dev)
            view_cu = make_nyan_view(base_cu)
            cuda_op = (lambda: view_cu + 0.0) if compact_already \
                else (lambda: view_cu.compact())
            stats = bench(cuda_op, sync=cuda_sync, budget_s=0.4, flush=flusher.flush_gpu)
            record("nyangrad_cuda", stats)

            stats = bench(_kernel_only(cuda_dev, view_cu, compact_already),
                          sync=cuda_sync, budget_s=0.4, flush=flusher.flush_gpu)
            record("nyangrad_cuda_kernel", stats, note + " (output preallocated)")

        # ---- torch cuda --------------------------------------------------- #
        if HAVE_TORCH_CUDA:
            base_tc = torch.from_numpy(base_np).cuda()
            view_tc = make_torch_view(base_tc)
            torch_cuda_op = (lambda: view_tc + 0.0) if compact_already \
                else (lambda: view_tc.contiguous())
            stats = bench(torch_cuda_op, sync=cuda_sync, budget_s=0.4, flush=flusher.flush_gpu)
            record("torch_cuda", stats)
            del base_tc, view_tc
            torch.cuda.empty_cache()

    return records


ALLOC_SHAPES = [(256, 256), (1024, 1024), (4096, 4096)]


def allocator_records() -> list[Record]:
    """Cost of one buffer allocation, since every op here allocates its output.

    nyangrad calls straight into cudaMalloc/free per operation while torch serves
    the same request from a caching allocator, and on a large buffer that gap is
    bigger than any kernel difference. Worth measuring separately rather than
    leaving it hidden inside the elementwise timings.
    """
    records: list[Record] = []
    cpu_dev = nyan.cpu()
    cuda_dev = nyan.cuda()

    for shape in ALLOC_SHAPES:
        elements = shape[0] * shape[1]
        mb = elements * 4 / 1e6
        shape_str = "x".join(str(s) for s in shape)
        case = f"alloc_free_{mb:.2f}mb" if mb < 1 else f"alloc_free_{mb:.0f}mb"

        def add(backend, stats, notes):
            records.append(Record(
                "allocator", case, backend, shape_str, stats.to_dict(),
                {"mb": mb, "gb_per_s": mb / 1e3 / (stats.median_ms / 1e3),
                 "us": stats.median_ms * 1e3},
                notes,
            ))

        if cpu_dev.enabled():
            stats = bench(lambda: nd.NDArray.make(shape, device=cpu_dev), budget_s=0.3)
            add("nyangrad_cpu", stats, "aligned_alloc + free per op")
        if cuda_dev.enabled():
            stats = bench(lambda: nd.NDArray.make(shape, device=cuda_dev),
                          sync=cuda_sync, budget_s=0.3)
            add("nyangrad_cuda", stats, "cudaMalloc + cudaFree per op")
        if HAVE_TORCH:
            stats = bench(lambda: torch.empty(shape, dtype=torch.float32), budget_s=0.3)
            add("torch_cpu", stats, "caching allocator")
        if HAVE_TORCH_CUDA:
            stats = bench(lambda: torch.empty(shape, dtype=torch.float32, device="cuda"),
                          sync=cuda_sync, budget_s=0.3)
            add("torch_cuda", stats, "caching allocator")

    return records


def dispatch_anatomy() -> list[Record]:
    """Split one elementwise op into allocate, launch, and free.

    The parts do not add up, and that is the point. Allocating a buffer costs a
    few microseconds and launching the kernel costs a few more, but doing both in
    the same operation costs several times their sum, because freeing the output
    while the kernel that writes it is still in flight makes cudaFree wait for the
    device to go idle. That serialises every op against the previous one and is
    what a caching allocator exists to avoid.
    """
    records: list[Record] = []
    tiny = np.ones((1, 1), dtype="float32")

    for backend, device in (("nyangrad_cpu", nyan.cpu()), ("nyangrad_cuda", nyan.cuda())):
        if not device.enabled():
            continue
        sync = cuda_sync if "cuda" in backend else None
        a = nd.array(tiny, device=device)
        out = nd.NDArray.make((1, 1), device=device)
        kept: list = []

        def launch_only():
            device.scalar_add(a._handle, 0.0, out._handle)

        def alloc_only():
            nd.NDArray.make((1, 1), device=device)

        def alloc_launch_free():
            fresh = nd.NDArray.make((1, 1), device=device)
            device.scalar_add(a._handle, 0.0, fresh._handle)

        def alloc_launch_kept():
            fresh = nd.NDArray.make((1, 1), device=device)
            device.scalar_add(a._handle, 0.0, fresh._handle)
            kept.append(fresh)

        for case, fn, note in (
            ("launch_only", launch_only, "output reused, nothing allocated or freed"),
            ("alloc_free_only", alloc_only, "allocate and free, no kernel"),
            ("alloc_launch_free", alloc_launch_free, "what every op actually does"),
            ("alloc_launch_kept", alloc_launch_kept,
             "same, but the output is kept alive so nothing is freed"),
        ):
            stats = bench(fn, sync=sync, budget_s=0.4, max_samples=20)
            records.append(Record(
                "dispatch", case, backend, "1x1", stats.to_dict(),
                {"us": stats.median_ms * 1e3}, note,
            ))
        kept.clear()

    return records


def verify_correctness() -> list[dict]:
    """A compaction that returns wrong data is not worth timing."""
    checks = []
    base_np = np.random.randn(64, 56, 56, 64).astype(np.float32)
    reference = np.ascontiguousarray(base_np.transpose(0, 3, 1, 2))

    for name, device in (("nyangrad_cpu", nyan.cpu()), ("nyangrad_cuda", nyan.cuda())):
        if not device.enabled():
            continue
        got = nd.array(base_np, device=device).permute((0, 3, 1, 2)).compact().numpy()
        err = float(np.abs(got - reference).max())
        checks.append({
            "backend": name,
            "case": "nhwc_to_nchw compaction",
            "max_abs_err": err,
            "passed": err == 0.0,
        })
    return checks
