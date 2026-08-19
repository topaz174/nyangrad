"""End-to-end layer and model passes: nyangrad on each device against PyTorch.

The model is a plain MLP (784 -> h -> h -> h -> 10, ReLU between, softmax cross
entropy at the end) so the comparison is about framework and kernel efficiency
rather than about who implements a fancier layer. Three sizes are used on
purpose: the small one is dominated by per-op dispatch, the large one by actual
arithmetic, and watching the ranking flip between them is the whole point.

Alongside the timings this module measures two things that explain them:

- per-op dispatch latency on a 1-element tensor, split into the autograd layer
  (Tensor + Tensor) and the bare array layer (NDArray + NDArray), which isolates
  graph bookkeeping from kernel launch and allocation cost
- how many kernel launches, device allocations and host/device transfers a single
  training step actually issues, counted through a proxy on the kernel module
"""
from __future__ import annotations

import numpy as np

import nyangrad as nyan
import nyangrad.nn as nn
from nyangrad import ndarray as nd

from harness import (
    HAVE_TORCH,
    HAVE_TORCH_CUDA,
    Record,
    bench,
    cuda_sync,
    gemm_flops,
    torch,
)
from instrument import graph_size, instrumented_device

# (label, batch, hidden)
CONFIGS = [
    ("small_b64_h256", 64, 256),
    ("medium_b256_h1024", 256, 1024),
    ("large_b512_h2048", 512, 2048),
]
IN_FEATURES = 784
NUM_CLASSES = 10
DEPTH = 3  # hidden layers


def _model_flops(batch: int, hidden: int) -> float:
    """Forward GEMM flops; backward is about twice this."""
    dims = [(IN_FEATURES, hidden)] + [(hidden, hidden)] * (DEPTH - 1) + [(hidden, NUM_CLASSES)]
    return sum(gemm_flops(batch, out_f, in_f) for in_f, out_f in dims)


def _build_nyangrad(batch: int, hidden: int, device):
    np.random.seed(0)
    layers = [nn.Linear(IN_FEATURES, hidden, device=device), nn.ReLU()]
    for _ in range(DEPTH - 1):
        layers += [nn.Linear(hidden, hidden, device=device), nn.ReLU()]
    layers.append(nn.Linear(hidden, NUM_CLASSES, device=device))
    model = nn.Sequential(*layers)

    x = nyan.Tensor(
        np.random.randn(batch, IN_FEATURES).astype("float32"), device=device, dtype="float32"
    )
    y = nyan.Tensor(
        np.random.randint(0, NUM_CLASSES, size=(batch,)).astype("float32"),
        device=device, dtype="float32",
    )
    return model, x, y, nn.SoftmaxLoss()


def _build_torch(batch: int, hidden: int, device: str, threads: int | None = None):
    torch.manual_seed(0)
    if threads:
        torch.set_num_threads(threads)
    mods = [torch.nn.Linear(IN_FEATURES, hidden), torch.nn.ReLU()]
    for _ in range(DEPTH - 1):
        mods += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()]
    mods.append(torch.nn.Linear(hidden, NUM_CLASSES))
    model = torch.nn.Sequential(*mods).to(device)

    x = torch.randn(batch, IN_FEATURES, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch,), device=device)
    return model, x, y


def _record(case, backend, backend_shape, mode, stats, batch, hidden, notes="") -> Record:
    seconds = stats.median_ms / 1e3
    fwd_flops = _model_flops(batch, hidden)
    flops = fwd_flops if mode == "forward" else fwd_flops * 3  # fwd + 2x for bwd
    metrics = {
        "gflops": flops / seconds / 1e9,
        "flops": flops,
        "samples_per_s": batch / seconds,
        "mode": mode,
    }
    return Record(f"model_{mode}", case, backend, backend_shape, stats.to_dict(), metrics, notes)


def run(ceilings: dict) -> list[Record]:
    records: list[Record] = []
    default_threads = torch.get_num_threads() if HAVE_TORCH else None

    nyan_devices = [("nyangrad_numpy", nyan.numpy_device())]
    if nyan.cpu().enabled():
        nyan_devices.append(("nyangrad_cpu", nyan.cpu()))
    if nyan.cuda().enabled():
        nyan_devices.append(("nyangrad_cuda", nyan.cuda()))

    for case, batch, hidden in CONFIGS:
        shape = f"b{batch}xh{hidden}"
        print(f"  model {case:<20}", flush=True)

        for backend, device in nyan_devices:
            model, x, y, loss_fn = _build_nyangrad(batch, hidden, device)
            sync = cuda_sync if "cuda" in backend else None

            def forward():
                return loss_fn(model(x), y)

            def forward_backward():
                loss = loss_fn(model(x), y)
                loss.backward()
                return loss

            stats = bench(forward, sync=sync, budget_s=0.6)
            records.append(_record(case, backend, shape, "forward", stats, batch, hidden))

            stats = bench(forward_backward, sync=sync, budget_s=0.6)
            records.append(_record(case, backend, shape, "forward_backward", stats, batch, hidden))
            del model, x, y

        if HAVE_TORCH:
            for backend, threads in (("torch_cpu_1t", 1), ("torch_cpu_all", default_threads)):
                model, x, y = _build_torch(batch, hidden, "cpu", threads)

                def forward():
                    with torch.no_grad():
                        return torch.nn.functional.cross_entropy(model(x), y)

                def forward_backward():
                    model.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    return loss

                stats = bench(forward, budget_s=0.5)
                records.append(_record(case, backend, shape, "forward", stats, batch, hidden,
                                       f"{threads} thread(s)"))
                stats = bench(forward_backward, budget_s=0.5)
                records.append(_record(case, backend, shape, "forward_backward", stats,
                                       batch, hidden, f"{threads} thread(s)"))
            torch.set_num_threads(default_threads)

        if HAVE_TORCH_CUDA:
            model, x, y = _build_torch(batch, hidden, "cuda")
            prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False

            def forward():
                with torch.no_grad():
                    return torch.nn.functional.cross_entropy(model(x), y)

            def forward_backward():
                model.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x), y)
                loss.backward()
                return loss

            try:
                stats = bench(forward, sync=cuda_sync, budget_s=0.5)
                records.append(_record(case, "torch_cuda", shape, "forward", stats,
                                       batch, hidden, "tf32 disabled"))
                stats = bench(forward_backward, sync=cuda_sync, budget_s=0.5)
                records.append(_record(case, "torch_cuda", shape, "forward_backward", stats,
                                       batch, hidden, "tf32 disabled"))
            finally:
                torch.backends.cuda.matmul.allow_tf32 = prev
            del model, x, y
            torch.cuda.empty_cache()

    return records


def op_overhead() -> list[Record]:
    """Per-op latency on a 1-element tensor, where only overhead remains."""
    records: list[Record] = []
    tiny = np.ones((1, 1), dtype="float32")

    devices = [("nyangrad_numpy", nyan.numpy_device())]
    if nyan.cpu().enabled():
        devices.append(("nyangrad_cpu", nyan.cpu()))
    if nyan.cuda().enabled():
        devices.append(("nyangrad_cuda", nyan.cuda()))

    for backend, device in devices:
        sync = cuda_sync if "cuda" in backend else None

        t = nyan.Tensor(tiny, device=device, dtype="float32")
        stats = bench(lambda: t + t, sync=sync, budget_s=0.3)
        records.append(Record("overhead", "tensor_add_1elem", backend, "1x1",
                              stats.to_dict(), {"us_per_op": stats.median_ms * 1e3},
                              "autograd layer: graph node + kernel + allocation"))

        stats = bench(lambda: t @ t, sync=sync, budget_s=0.3)
        records.append(Record("overhead", "tensor_matmul_1elem", backend, "1x1",
                              stats.to_dict(), {"us_per_op": stats.median_ms * 1e3},
                              "autograd layer"))

        if backend != "nyangrad_numpy":
            a = nd.array(tiny, device=device)
            stats = bench(lambda: a + a, sync=sync, budget_s=0.3)
            records.append(Record("overhead", "ndarray_add_1elem", backend, "1x1",
                                  stats.to_dict(), {"us_per_op": stats.median_ms * 1e3},
                                  "array layer only: kernel + allocation, no graph"))

    if HAVE_TORCH:
        t = torch.ones(1, 1)
        stats = bench(lambda: t + t, budget_s=0.3)
        records.append(Record("overhead", "tensor_add_1elem", "torch_cpu", "1x1",
                              stats.to_dict(), {"us_per_op": stats.median_ms * 1e3}, ""))
    if HAVE_TORCH_CUDA:
        t = torch.ones(1, 1, device="cuda")
        stats = bench(lambda: t + t, sync=cuda_sync, budget_s=0.3)
        records.append(Record("overhead", "tensor_add_1elem", "torch_cuda", "1x1",
                              stats.to_dict(), {"us_per_op": stats.median_ms * 1e3}, ""))

    return records


def step_anatomy() -> dict:
    """Graph size, kernel launches, allocations and transfers for one step."""
    anatomy: dict = {}
    for case, batch, hidden in CONFIGS:
        entry: dict = {}
        for backend, factory in (("nyangrad_cpu", nyan.cpu), ("nyangrad_cuda", nyan.cuda)):
            device = factory()
            if not device.enabled():
                continue
            counter = instrumented_device(device)
            model, x, y, loss_fn = _build_nyangrad(batch, hidden, device)

            # forward only
            counter.reset()
            loss = loss_fn(model(x), y)
            forward_counts = counter.report()
            forward_nodes = graph_size(loss)

            # forward + backward
            counter.reset()
            loss = loss_fn(model(x), y)
            loss.backward()
            step_counts = counter.report()

            entry[backend] = {
                "forward_graph_nodes": forward_nodes,
                "forward": forward_counts,
                "forward_backward": step_counts,
            }
            del model, x, y
        anatomy[case] = entry
    return anatomy


def verify_correctness() -> list[dict]:
    """Same inputs and weights through nyangrad and torch should agree."""
    checks = []
    batch, hidden = 64, 256
    np.random.seed(0)
    x_np = np.random.randn(batch, IN_FEATURES).astype("float32")
    weights = [
        np.random.randn(IN_FEATURES, hidden).astype("float32") * 0.05,
        np.random.randn(hidden, NUM_CLASSES).astype("float32") * 0.05,
    ]

    def nyangrad_forward(device):
        l1 = nn.Linear(IN_FEATURES, hidden, device=device)
        l2 = nn.Linear(hidden, NUM_CLASSES, device=device)
        l1.weight = nn.Parameter(nyan.Tensor(weights[0], device=device, dtype="float32"))
        l1.bias = nn.Parameter(nyan.Tensor(np.zeros((1, hidden), "float32"),
                                           device=device, dtype="float32"))
        l2.weight = nn.Parameter(nyan.Tensor(weights[1], device=device, dtype="float32"))
        l2.bias = nn.Parameter(nyan.Tensor(np.zeros((1, NUM_CLASSES), "float32"),
                                           device=device, dtype="float32"))
        x = nyan.Tensor(x_np, device=device, dtype="float32")
        return nn.Sequential(l1, nn.ReLU(), l2)(x).numpy()

    reference = np.maximum(x_np @ weights[0], 0) @ weights[1]

    for name, device in (("nyangrad_numpy", nyan.numpy_device()),
                         ("nyangrad_cpu", nyan.cpu()),
                         ("nyangrad_cuda", nyan.cuda())):
        if hasattr(device, "enabled") and not device.enabled():
            continue
        got = nyangrad_forward(device)
        err = float(np.abs(got - reference).max())
        rel = err / float(np.abs(reference).max())
        checks.append({
            "backend": name,
            "case": "2-layer mlp forward vs numpy reference",
            "max_abs_err": err,
            "max_rel_err": rel,
            "passed": rel < 1e-4,
        })
    return checks
