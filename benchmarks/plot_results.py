"""Render benchmarks/results.json as dark-mode charts under assets/.

    python benchmarks/plot_results.py
    python benchmarks/plot_results.py --results benchmarks/results.json --out assets

Produces:
    gemm_scaling.png        GFLOP/s against matrix size, every backend
    gemm_roofline.png       achieved throughput against arithmetic intensity, with
                            the measured compute and bandwidth roofs
    memory_bandwidth.png    effective bandwidth per compaction case
    model_latency.png       MLP step latency per config
    overhead.png            per-op dispatch cost and allocation cost
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BG = "#0d1117"
PANEL = "#0d1117"
GRID = "#30363d"
TEXT = "#c9d1d9"
MUTED = "#8b949e"

COLORS = {
    "nyangrad_cpu_naive": "#f0883e",
    "nyangrad_cpu_tiled": "#e5534b",
    "nyangrad_cpu": "#e5534b",
    "nyangrad_cpu_kernel": "#ffa657",
    "nyangrad_cuda": "#d2a8ff",
    "nyangrad_cuda_kernel": "#a371f7",
    "nyangrad_numpy": "#bb8009",
    "numpy": "#3fb950",
    "torch_cpu_1t": "#58a6ff",
    "torch_cpu_all": "#1f6feb",
    "torch_cpu": "#58a6ff",
    "torch_cuda": "#39c5cf",
}
LABELS = {
    "numpy": "numpy (OpenBLAS)",
    "torch_cpu_1t": "torch cpu, 1 thread",
    "torch_cpu_all": "torch cpu, 6 threads",
    "torch_cpu": "torch cpu",
    "torch_cuda": "torch cuda (cuBLAS)",
    "nyangrad_cpu_naive": "nyangrad cpu, naive",
    "nyangrad_cpu_tiled": "nyangrad cpu, tiled",
    "nyangrad_cpu": "nyangrad cpu",
    "nyangrad_cuda": "nyangrad cuda",
    "nyangrad_numpy": "nyangrad numpy",
    "nyangrad_cpu_kernel": "nyangrad cpu, kernel only",
    "nyangrad_cuda_kernel": "nyangrad cuda, kernel only",
}
ORDER = [
    "nyangrad_cpu_naive", "nyangrad_cpu_tiled", "nyangrad_cpu", "nyangrad_cpu_kernel",
    "nyangrad_numpy", "nyangrad_cuda", "nyangrad_cuda_kernel",
    "numpy", "torch_cpu_1t", "torch_cpu_all", "torch_cpu", "torch_cuda",
]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def label(backend: str) -> str:
    return LABELS.get(backend, backend)


def color(backend: str) -> str:
    return COLORS.get(backend, "#8b949e")


def sort_backends(names) -> list[str]:
    return sorted(names, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))


def pick(records, **filters):
    out = records
    for key, value in filters.items():
        out = [r for r in out if r[key] == value]
    return out


def annotate_source(fig, data, y: float = -0.02) -> None:
    env = data["environment"]
    gpu = env.get("gpu", {}).get("name", "no gpu")
    fig.text(
        0.5, y,
        f"{env['cpu']['name']}  |  {gpu}  |  fp32, tf32 disabled  |  median of repeated runs",
        ha="center", va="top", color=MUTED, fontsize=8,
    )


# --------------------------------------------------------------------------- #

def plot_gemm_scaling(data, out: Path) -> None:
    records = [r for r in pick(data["records"], suite="gemm")
               if len(set(r["case"].split("x"))) == 1]
    if not records:
        return
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))

    sizes = sorted({int(r["case"].split("x")[0]) for r in records})
    all_gflops, all_latency = [], []
    for backend in sort_backends({r["backend"] for r in records}):
        xs, ys, ls = [], [], []
        for n in sizes:
            hit = pick(records, case=f"{n}x{n}x{n}", backend=backend)
            if hit:
                xs.append(n)
                ys.append(hit[0]["metrics"]["gflops"])
                ls.append(hit[0]["stats"]["median_ms"])
        if not xs:
            continue
        all_gflops += ys
        all_latency += ls
        style = dict(color=color(backend), marker="o", label=label(backend))
        if backend.startswith("nyangrad"):
            style["linewidth"] = 2.6
        else:
            style["linestyle"] = "--"
            style["alpha"] = 0.85
        ax_left.plot(xs, ys, **style)
        ax_right.plot(xs, ls, **style)

    ceilings = data.get("ceilings") or {}
    gpu_roof = (ceilings.get("gpu_compute") or {}).get("gflops")
    cpu_roof = (ceilings.get("cpu_compute") or {}).get("gflops_single_thread")
    for roof, text, tint in ((gpu_roof, "measured gpu fp32 ceiling", "#39c5cf"),
                             (cpu_roof, "measured cpu ceiling, 1 thread", "#58a6ff")):
        if roof:
            ax_left.axhline(roof, color=tint, linestyle=":", linewidth=1.2, alpha=0.7)
            ax_left.text(sizes[-1], roof * 1.2, text, color=tint, fontsize=8, ha="right")

    for ax, ylabel, title, values in (
        (ax_left, "GFLOP/s", "Throughput", all_gflops + [r for r in (gpu_roof,) if r]),
        (ax_right, "latency (ms)", "Latency", all_latency),
    ):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("square matrix size N")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        # limits go on after the log scale, or matplotlib keeps the linear bounds
        if values:
            ax.set_ylim(min(values) / 3, max(values) * 3)

    # the band between the cpu kernels and the gpu curves is the empty part
    ax_left.legend(loc="center right", fontsize=8, ncol=2)
    fig.suptitle("fp32 GEMM scaling, C = A @ B", y=1.02, fontsize=13, fontweight="bold")
    annotate_source(fig, data)
    fig.savefig(out / "gemm_scaling.png")
    plt.close(fig)


def plot_roofline(data, out: Path) -> None:
    ceilings = data.get("ceilings") or {}
    gpu_compute = (ceilings.get("gpu_compute") or {}).get("gflops")
    gpu_bw = (ceilings.get("gpu_bandwidth") or {}).get("gb_per_s")
    if not (gpu_compute and gpu_bw):
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ridge = gpu_compute / gpu_bw
    xs_bw = [ridge / 512, ridge]
    ax.plot(xs_bw, [gpu_bw * x for x in xs_bw], color="#39c5cf", linewidth=2)
    ax.plot([ridge, ridge * 512], [gpu_compute, gpu_compute], color="#39c5cf", linewidth=2)
    ax.text(ridge / 300, gpu_bw * (ridge / 300) * 1.25,
            f"{gpu_bw:.0f} GB/s measured bandwidth roof",
            color="#39c5cf", fontsize=8, rotation=34)
    ax.text(ridge * 3, gpu_compute * 1.15,
            f"{gpu_compute / 1000:.1f} TFLOP/s measured fp32 roof", color="#39c5cf", fontsize=8)
    ax.axvline(ridge, color=GRID, linestyle=":", linewidth=1)
    ax.text(ridge * 1.1, gpu_bw * ridge / 90, f"ridge point\n{ridge:.0f} FLOP/byte",
            color=MUTED, fontsize=8)

    gemm = [r for r in pick(data["records"], suite="gemm", backend="nyangrad_cuda")
            if len(set(r["case"].split("x"))) == 1]
    if gemm:
        xs = [r["metrics"]["arithmetic_intensity"] for r in gemm]
        ys = [r["metrics"]["gflops"] for r in gemm]
        ax.scatter(xs, ys, s=70, color="#d2a8ff", zorder=5, label="nyangrad cuda GEMM")
        for i, (r, x, y) in enumerate(zip(gemm, xs, ys)):
            # alternate the offset so consecutive labels do not collide
            offset = (8, 6) if i % 2 == 0 else (8, -12)
            ax.annotate(f"N={r['case'].split('x')[0]}", (x, y), textcoords="offset points",
                        xytext=offset, color="#d2a8ff", fontsize=8)

    torch_gemm = [r for r in pick(data["records"], suite="gemm", backend="torch_cuda")
                  if len(set(r["case"].split("x"))) == 1]
    if torch_gemm:
        ax.scatter([r["metrics"]["arithmetic_intensity"] for r in torch_gemm],
                   [r["metrics"]["gflops"] for r in torch_gemm],
                   s=55, color="#39c5cf", marker="D", zorder=5, label="cuBLAS GEMM")

    # compaction is deliberately absent: it performs no arithmetic, so it has no
    # arithmetic intensity to place it at. Its ceiling is the bandwidth roof, and
    # it is measured against that in memory_bandwidth.png instead.

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP per byte of minimum traffic)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("Roofline, RTX 3060, against ceilings measured on this machine")
    ax.legend(loc="lower right", fontsize=8)
    annotate_source(fig, data)
    fig.savefig(out / "gemm_roofline.png")
    plt.close(fig)


def _grouped_bars(ax, cases, backends, values, ylabel, title, log=True):
    width = 0.8 / max(len(backends), 1)
    for i, backend in enumerate(backends):
        offsets = [j + i * width - 0.4 + width / 2 for j in range(len(cases))]
        heights = [values[(c, backend)] for c in cases]
        ax.bar(offsets, heights, width=width, color=color(backend), label=label(backend))
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels(cases, rotation=20, ha="right", fontsize=8)
    finite = [v for v in values.values() if v > 0]
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        if finite:
            ax.set_ylim(min(finite) / 2, max(finite) * 2.5)
    elif finite:
        ax.set_ylim(0, max(finite) * 1.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def shared_legend(fig, axes, ncol=4, y=-0.16):
    """One legend under the figure, so bars never sit behind it."""
    handles, labels = [], []
    for ax in axes:
        for handle, text in zip(*ax.get_legend_handles_labels()):
            if text not in labels:
                handles.append(handle)
                labels.append(text)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, y),
               ncol=ncol, fontsize=9)


def plot_memory(data, out: Path) -> None:
    records = pick(data["records"], suite="memory")
    if not records:
        return
    cases = []
    for r in records:
        if r["case"] not in cases:
            cases.append(r["case"])
    backends = sort_backends({r["backend"] for r in records})
    values = {}
    for case in cases:
        for backend in backends:
            hit = pick(records, case=case, backend=backend)
            values[(case, backend)] = hit[0]["metrics"]["gb_per_s"] if hit else 0.0

    fig, ax = plt.subplots(figsize=(11, 5))
    _grouped_bars(ax, cases, backends, values, "effective GB/s",
                  "Making a strided view contiguous: effective bandwidth")

    ceilings = data.get("ceilings") or {}
    if ceilings.get("gpu_bandwidth"):
        ax.axhline(ceilings["gpu_bandwidth"]["gb_per_s"], color="#39c5cf",
                   linestyle=":", linewidth=1.2)
        ax.text(-0.45, ceilings["gpu_bandwidth"]["gb_per_s"] * 1.06,
                "gpu bandwidth ceiling", color="#39c5cf", fontsize=8)
    if ceilings.get("cpu_bandwidth"):
        ax.axhline(ceilings["cpu_bandwidth"]["gb_per_s"], color="#58a6ff",
                   linestyle=":", linewidth=1.2)
        ax.text(-0.45, ceilings["cpu_bandwidth"]["gb_per_s"] * 1.06,
                "cpu bandwidth ceiling", color="#58a6ff", fontsize=8)
    shared_legend(fig, [ax], ncol=4, y=-0.12)
    annotate_source(fig, data, y=-0.22)
    fig.savefig(out / "memory_bandwidth.png")
    plt.close(fig)


def plot_model(data, out: Path) -> None:
    modes = [("model_forward", "forward"), ("model_forward_backward", "forward + backward")]
    available = [(m, t) for m, t in modes if pick(data["records"], suite=m)]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(6.2 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, (mode, title) in zip(axes, available):
        records = pick(data["records"], suite=mode)
        cases = []
        for r in records:
            if r["case"] not in cases:
                cases.append(r["case"])
        backends = sort_backends({r["backend"] for r in records})
        values = {}
        for case in cases:
            for backend in backends:
                hit = pick(records, case=case, backend=backend)
                values[(case, backend)] = hit[0]["stats"]["median_ms"] if hit else 0.0
        _grouped_bars(ax, cases, backends, values, "latency (ms), log scale",
                      f"MLP {title}")
    shared_legend(fig, axes, ncol=4, y=-0.06)
    fig.suptitle("End-to-end MLP step: 784 -> h -> h -> h -> 10",
                 y=1.02, fontsize=13, fontweight="bold")
    annotate_source(fig, data, y=-0.18)
    fig.savefig(out / "model_latency.png")
    plt.close(fig)


def plot_overhead(data, out: Path) -> None:
    overhead = pick(data["records"], suite="overhead")
    alloc = pick(data["records"], suite="allocator")
    if not overhead and not alloc:
        return
    panels = int(bool(overhead)) + int(bool(alloc))
    fig, axes = plt.subplots(1, panels, figsize=(6.4 * panels, 4.8))
    axes = [axes] if panels == 1 else list(axes)
    idx = 0

    if overhead:
        ax = axes[idx]
        idx += 1
        cases = []
        for r in overhead:
            if r["case"] not in cases:
                cases.append(r["case"])
        backends = sort_backends({r["backend"] for r in overhead})
        values = {}
        for case in cases:
            for backend in backends:
                hit = pick(overhead, case=case, backend=backend)
                values[(case, backend)] = hit[0]["metrics"]["us_per_op"] if hit else 0.0
        _grouped_bars(ax, cases, backends, values, "microseconds per op, log scale",
                      "Dispatch cost on a 1-element tensor")

    if alloc:
        ax = axes[idx]
        cases = []
        for r in alloc:
            if r["case"] not in cases:
                cases.append(r["case"])
        backends = sort_backends({r["backend"] for r in alloc})
        values = {}
        for case in cases:
            for backend in backends:
                hit = pick(alloc, case=case, backend=backend)
                values[(case, backend)] = hit[0]["metrics"]["us"] if hit else 0.0
        _grouped_bars(ax, cases, backends, values, "microseconds, log scale",
                      "Allocating and freeing one output buffer")

    shared_legend(fig, axes, ncol=4, y=-0.06)
    fig.suptitle("Where the time goes when the arrays are small",
                 y=1.02, fontsize=13, fontweight="bold")
    annotate_source(fig, data, y=-0.18)
    fig.savefig(out / "overhead.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(Path(__file__).parent / "results.json"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "assets"))
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    apply_style()

    for plot in (plot_gemm_scaling, plot_roofline, plot_memory, plot_model, plot_overhead):
        plot(data, out)

    written = sorted(p.name for p in out.glob("*.png"))
    print(f"wrote {len(written)} charts to {out}/")
    for name in written:
        print(f"  {name}")


if __name__ == "__main__":
    main()
