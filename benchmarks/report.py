"""Render benchmarks/results.json as ASCII tables.

    python benchmarks/report.py                     # everything, to stdout
    python benchmarks/report.py --out benchmarks/RESULTS.md
    python benchmarks/report.py --section gemm

Reads only; run_all.py is what produces the numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results.json"

BACKEND_LABELS = {
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

# column order in every table
ORDER = [
    "nyangrad_numpy", "nyangrad_cpu_naive", "nyangrad_cpu_tiled", "nyangrad_cpu",
    "nyangrad_cpu_kernel", "nyangrad_cuda", "nyangrad_cuda_kernel",
    "numpy", "torch_cpu_1t", "torch_cpu_all", "torch_cpu", "torch_cuda",
]


def label(backend: str) -> str:
    return BACKEND_LABELS.get(backend, backend)


def sort_backends(names) -> list[str]:
    return sorted(names, key=lambda n: (ORDER.index(n) if n in ORDER else 99, n))


def table(headers: list[str], rows: list[list[str]], align: str | None = None) -> str:
    """Fixed-width ASCII table with a rule under the header."""
    if not rows:
        return "(no data)\n"
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    align = align or ("l" + "r" * (len(headers) - 1))

    def fmt(cells):
        out = []
        for cell, width, a in zip(cells, widths, align):
            text = str(cell)
            out.append(text.ljust(width) if a == "l" else text.rjust(width))
        return "  ".join(out).rstrip()

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(row) for row in rows]
    return "\n".join(lines) + "\n"


def pick(records, suite=None, case=None, backend=None):
    out = records
    if suite:
        out = [r for r in out if r["suite"] == suite]
    if case:
        out = [r for r in out if r["case"] == case]
    if backend:
        out = [r for r in out if r["backend"] == backend]
    return out


def num(value, spec=".2f", dash="-"):
    return format(value, spec) if value is not None else dash


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #

def section_environment(data) -> str:
    env = data["environment"]
    out = ["## Machine and build\n"]
    rows = [
        ["cpu", f"{env['cpu']['name']} ({env['cpu']['physical']} cores / "
                f"{env['cpu']['logical']} threads)"],
        ["numpy", f"{env['numpy']['version']} on {env['numpy']['blas']}"],
        ["python", env["python"]],
    ]
    if "gpu" in env:
        gpu = env["gpu"]
        rows.insert(1, ["gpu", f"{gpu['name']} ({gpu['sm_count']} SMs, sm_{gpu['compute_capability'].replace('.', '')}, "
                               f"{gpu['total_memory_gb']} GB, {gpu['l2_cache_mb']} MB L2)"])
        rows.append(["driver / cuda", f"{gpu['driver']} / {env.get('torch', {}).get('cuda_version', '?')}"])
    if "torch" in env:
        rows.append(["torch", env["torch"]["version"]])
    out.append(table(["", ""], rows, align="ll"))

    ceilings = data.get("ceilings") or {}
    if ceilings:
        out.append("\n### Measured ceilings on this machine\n")
        out.append("Every 'percent of peak' below is against these, not against a spec sheet.\n\n")
        rows = []
        gc = ceilings.get("gpu_compute")
        if gc:
            rows.append(["gpu fp32 gemm", f"{gc['gflops']:.0f} GFLOP/s", gc["source"]])
        gb = ceilings.get("gpu_bandwidth")
        if gb:
            rows.append(["gpu bandwidth", f"{gb['gb_per_s']:.0f} GB/s", gb["source"]])
        cc = ceilings.get("cpu_compute")
        if cc:
            rows.append(["cpu fp32 gemm, 1 thread",
                         f"{cc['gflops_single_thread']:.0f} GFLOP/s", cc["source"]])
            rows.append([f"cpu fp32 gemm, {cc['threads_all_threads']} threads",
                         f"{cc['gflops_all_threads']:.0f} GFLOP/s", cc["source"]])
        cb = ceilings.get("cpu_bandwidth")
        if cb:
            rows.append(["cpu bandwidth", f"{cb['gb_per_s']:.0f} GB/s", cb["source"]])
        if "gpu" in env and env["gpu"].get("spec_fp32_tflops_at_max_boost"):
            rows.append(["gpu fp32 spec at max boost",
                         f"{env['gpu']['spec_fp32_tflops_at_max_boost'] * 1000:.0f} GFLOP/s",
                         "SMs x 128 lanes x 2 x max boost clock"])
        out.append(table(["ceiling", "value", "how it was measured"], rows, align="lll"))
        settle = ceilings.get("clock_settle") or {}
        if settle.get("applied"):
            out.append(
                f"\nThe GPU was held under load for {settle['seconds']:.0f}s before these were\n"
                f"taken, which brought the SM clock from {settle['sm_mhz_before']:.0f} to "
                f"{settle['sm_mhz_after']:.0f} MHz. Measuring a ceiling on an\n"
                "idle card understates it, and would flatter everything compared against it.\n"
            )

    clocks = data.get("clocks", {})
    if clocks.get("available"):
        # the full range spans the idle clock during the CPU-only phases, which
        # says nothing about stability; only the loaded samples do
        loaded = [s for s in clocks.get("trace", []) if s["util_pct"] >= 50]
        out.append("\n### Clock stability\n\n")
        if loaded:
            sm = sorted(s["sm_mhz"] for s in loaded)
            all_temp = [s["temp_c"] for s in clocks.get("trace", [])]
            all_power = [s["power_w"] for s in clocks.get("trace", [])]
            out.append(
                "This is a throttling check rather than a per-measurement record of the clock:\n"
                "sampling runs once a second, which is far coarser than most measurements here,\n"
                f"and much of the run is CPU-only with the GPU idling. Across the whole run the\n"
                f"card stayed between {min(all_temp):.0f} and {max(all_temp):.0f} C "
                f"and never drew more than {max(all_power):.0f} W, well inside\n"
                f"its limits, so nothing here was measured on a throttled card. Under load the SM\n"
                f"clock ran at a median of {sm[len(sm) // 2]:.0f} MHz and peaked at {sm[-1]:.0f} MHz. "
                "Clocks were left on the\ndefault governor rather than pinned, so these are the "
                "frequencies the hardware\ndelivers in ordinary use rather than a tuned best case.\n"
            )
        else:
            sm = clocks["sm_mhz"]
            out.append(f"SM clock median {sm['median']:.0f} MHz, "
                       f"range {sm['min']:.0f}-{sm['max']:.0f} MHz.\n")
    return "\n".join(out)


def section_correctness(data) -> str:
    checks = data.get("correctness", [])
    if not checks:
        return ""
    out = ["\n## Correctness before performance\n",
           "Timings from a backend that computes the wrong answer are worthless, so every\n"
           "backend is checked against a numpy reference first.\n\n"]
    rows = []
    for check in checks:
        rows.append([
            check["backend"],
            check.get("case", "gemm 256^3 vs numpy"),
            f"{check.get('max_abs_err', 0):.2e}",
            f"{check.get('max_rel_err', 0):.2e}" if "max_rel_err" in check else "-",
            "pass" if check["passed"] else "FAIL",
        ])
    out.append(table(["backend", "check", "max abs err", "max rel err", ""], rows,
                     align="llrrl"))
    return "\n".join(out)


def section_gemm(data) -> str:
    records = pick(data["records"], suite="gemm")
    if not records:
        return ""
    out = ["\n## GEMM scaling\n",
           "Square matmul, C = A @ B, fp32. Latency is the median of repeated batched\n"
           "launches; GFLOP/s is 2mnk over that latency. tf32 is disabled for the torch\n"
           "CUDA rows so every backend is doing the same fp32 arithmetic.\n\n"]

    cases = []
    for r in records:
        if r["case"] not in cases:
            cases.append(r["case"])
    square = [c for c in cases if len(set(c.split("x"))) == 1]
    rect = [c for c in cases if c not in square]

    for title, group in (("### Square", square), ("### Rectangular", rect)):
        if not group:
            continue
        out.append(f"\n{title}\n")
        for case in group:
            case_records = pick(records, suite="gemm", case=case)
            n = case.split("x")
            heading = f"N = {n[0]}" if case in square else f"({n[0]}x{n[1]}) @ ({n[1]}x{n[2]})"
            flops = case_records[0]["metrics"]["flops"]
            work = f"{flops / 1e9:.1f} GFLOP" if flops >= 1e9 else f"{flops / 1e6:.0f} MFLOP"
            out.append(f"\n{heading}   ({work})\n\n")
            best = max(r["metrics"]["gflops"] for r in case_records)
            rows = []
            for backend in sort_backends({r["backend"] for r in case_records}):
                r = pick(case_records, case=case, backend=backend)[0]
                stats, metrics = r["stats"], r["metrics"]
                rows.append([
                    label(backend),
                    f"{stats['median_ms']:.3f}",
                    f"{metrics['gflops']:.1f}",
                    num(metrics.get("pct_of_measured_ceiling"), ".0f") + "%",
                    f"{best / metrics['gflops']:.1f}x",
                    "1 shot" if stats["single_shot"]
                    else f"{stats['rel_iqr'] * 100:.0f}%" + ("*" if stats.get("unstable") else ""),
                ])
            out.append(table(
                ["backend", "ms", "GFLOP/s", "of peak", "vs best", "IQR"], rows))
    out.append(
        "\nIQR is the interquartile range as a fraction of the median, i.e. run to run\n"
        "spread. A star marks a measurement still jittery after being retaken; those are\n"
        "threaded BLAS calls whose thread scheduling varies, not timing errors.\n"
        "\nA few CPU rows read above 100% of peak. The ceiling was measured at N=2048, where\n"
        "the operands do not fit in cache, so a smaller GEMM that stays resident in L3 can\n"
        "legitimately beat it. It is a property of the cache, not a broken measurement.\n"
    )
    return "\n".join(out)


def section_gemm_summary(data) -> str:
    """One wide table: GFLOP/s per backend across the square sizes."""
    records = pick(data["records"], suite="gemm")
    if not records:
        return ""
    cases = []
    for r in records:
        if r["case"] not in cases and len(set(r["case"].split("x"))) == 1:
            cases.append(r["case"])
    cases.sort(key=lambda c: int(c.split("x")[0]))
    backends = sort_backends({r["backend"] for r in records})

    rows = []
    for backend in backends:
        row = [label(backend)]
        for case in cases:
            hit = pick(records, suite="gemm", case=case, backend=backend)
            row.append(f"{hit[0]['metrics']['gflops']:.1f}" if hit else "-")
        rows.append(row)
    headers = ["GFLOP/s"] + [f"N={c.split('x')[0]}" for c in cases]
    return "\n### Square GEMM throughput summary\n\n" + table(headers, rows)


def section_speedup(data) -> str:
    """The naive -> tiled -> cuda progression, which is the actual story."""
    records = pick(data["records"], suite="gemm")
    if not records:
        return ""
    cases = sorted({r["case"] for r in records if len(set(r["case"].split("x"))) == 1},
                   key=lambda c: int(c.split("x")[0]))
    rows = []
    for case in cases:
        def gflops(backend):
            hit = pick(records, suite="gemm", case=case, backend=backend)
            return hit[0]["metrics"]["gflops"] if hit else None

        naive, tiled, cuda = gflops("nyangrad_cpu_naive"), gflops("nyangrad_cpu_tiled"), gflops("nyangrad_cuda")
        blas_1t, cublas = gflops("torch_cpu_1t"), gflops("torch_cuda")
        rows.append([
            f"N={case.split('x')[0]}",
            num(naive, ".1f"), num(tiled, ".1f"),
            f"{tiled / naive:.2f}x" if naive and tiled else "-",
            num(cuda, ".1f"),
            f"{cuda / tiled:.1f}x" if cuda and tiled else "-",
            f"{cuda / blas_1t:.1f}x" if cuda and blas_1t else "-",
            f"{cublas / cuda:.1f}x" if cuda and cublas else "-",
        ])
    return ("\n### Optimisation progression\n\n"
            "Each of my kernels against the previous one, then against the libraries.\n\n"
            + table(["size", "naive", "tiled", "tiled/naive", "cuda",
                     "cuda/tiled", "cuda/1t BLAS", "cuBLAS/cuda"], rows))


def section_memory(data) -> str:
    records = pick(data["records"], suite="memory")
    if not records:
        return ""
    out = ["\n## Striding and compaction\n",
           "Views are metadata only, so nothing above is paid for until a kernel needs\n"
           "contiguous input. This is that bill. Bandwidth counts the minimum required\n"
           "traffic (read the source, write the destination), so a backend that moves more\n"
           "than the minimum shows up here as low effective bandwidth.\n\n"]
    cases = []
    for r in records:
        if r["case"] not in cases:
            cases.append(r["case"])
    backends = sort_backends({r["backend"] for r in records})

    for metric, spec, title in (("median_ms", ".3f", "Latency (ms)"),
                                ("gb_per_s", ".1f", "Effective bandwidth (GB/s)")):
        out.append(f"\n### {title}\n\n")
        rows = []
        for case in cases:
            row = [case]
            for backend in backends:
                hit = pick(records, suite="memory", case=case, backend=backend)
                if not hit:
                    row.append("-")
                    continue
                source = hit[0]["stats"] if metric == "median_ms" else hit[0]["metrics"]
                row.append(format(source[metric], spec))
            rows.append(row)
        out.append(table(["case"] + [label(b) for b in backends], rows))

    notes = {r["case"]: r["notes"] for r in records}
    out.append(
        "\nThe 'kernel only' rows reuse an output buffer instead of allocating one, which\n"
        "separates the kernel from the allocation every op normally pays for. A kernel\n"
        "reading at or just over the measured bandwidth ceiling is at the roof; the ceiling\n"
        "and the kernel were timed minutes apart, so agreement to a percent is agreement.\n\n"
    )
    for case in cases:
        out.append(f"  {case:<20} {notes.get(case, '')}\n")
    return "".join(out) if isinstance(out, list) else out


def section_allocator(data) -> str:
    records = pick(data["records"], suite="allocator")
    if not records:
        return ""
    cases = []
    for r in records:
        if r["case"] not in cases:
            cases.append(r["case"])
    backends = sort_backends({r["backend"] for r in records})
    rows = []
    for case in cases:
        sample = pick(records, suite="allocator", case=case)[0]
        mb = sample["metrics"]["mb"]
        row = [f"{mb:.2f} MB" if mb < 1 else f"{mb:.0f} MB"]
        for backend in backends:
            hit = pick(records, suite="allocator", case=case, backend=backend)
            row.append(f"{hit[0]['metrics']['us']:.1f}" if hit else "-")
        rows.append(row)
    return ("\n## Allocation cost per operation (microseconds)\n\n"
            "Every nyangrad op allocates its output buffer straight from the device, while\n"
            "torch serves the same request from a caching allocator. On a large buffer that\n"
            "difference is bigger than any kernel difference.\n\n"
            + table(["buffer"] + [label(b) for b in backends], rows))


def section_dispatch(data) -> str:
    records = pick(data["records"], suite="dispatch")
    if not records:
        return ""
    cases = []
    for r in records:
        if r["case"] not in cases:
            cases.append(r["case"])
    backends = sort_backends({r["backend"] for r in records})
    rows = []
    for case in cases:
        row = [case]
        for backend in backends:
            hit = pick(records, suite="dispatch", case=case, backend=backend)
            row.append(f"{hit[0]['metrics']['us']:.1f}" if hit else "-")
        row.append(pick(records, suite="dispatch", case=case)[0]["notes"])
        rows.append(row)
    return ("\n## Anatomy of one elementwise op (microseconds)\n\n"
            "The parts do not add up, and that is the finding. Allocating costs a few\n"
            "microseconds and launching costs a few more, but an op that does both costs\n"
            "several times their sum, because freeing the output while the kernel writing it\n"
            "is still in flight makes cudaFree wait for the device to drain. Keeping the\n"
            "output alive removes the free and most of the cost with it.\n\n"
            + table(["step"] + [label(b) for b in backends] + ["what it isolates"],
                    rows, align="l" + "r" * len(backends) + "l"))


def section_model(data) -> str:
    out = []
    for mode, title in (("model_forward", "Forward pass"),
                        ("model_forward_backward", "Forward + backward")):
        records = pick(data["records"], suite=mode)
        if not records:
            continue
        cases = []
        for r in records:
            if r["case"] not in cases:
                cases.append(r["case"])
        backends = sort_backends({r["backend"] for r in records})
        rows = []
        for case in cases:
            row = [case]
            for backend in backends:
                hit = pick(records, suite=mode, case=case, backend=backend)
                row.append(f"{hit[0]['stats']['median_ms']:.2f}" if hit else "-")
            rows.append(row)
        out.append(f"\n### {title}, latency in ms\n\n"
                   + table(["config"] + [label(b) for b in backends], rows))

        rows = []
        for case in cases:
            row = [case]
            for backend in backends:
                hit = pick(records, suite=mode, case=case, backend=backend)
                row.append(f"{hit[0]['metrics']['gflops']:.0f}" if hit else "-")
            rows.append(row)
        out.append(f"\n### {title}, effective GFLOP/s\n\n"
                   + table(["config"] + [label(b) for b in backends], rows))

    if not out:
        return ""
    header = ("\n## End-to-end MLP\n\n"
              "784 -> h -> h -> h -> 10 with ReLU and softmax cross entropy. The small\n"
              "config is dominated by per-op dispatch, the large one by arithmetic, and the\n"
              "ranking changes between them.\n")
    return header + "".join(out)


def section_overhead(data) -> str:
    records = pick(data["records"], suite="overhead")
    if not records:
        return ""
    cases = []
    for r in records:
        if r["case"] not in cases:
            cases.append(r["case"])
    backends = sort_backends({r["backend"] for r in records})
    rows = []
    for case in cases:
        row = [case]
        for backend in backends:
            hit = pick(records, suite="overhead", case=case, backend=backend)
            row.append(f"{hit[0]['metrics']['us_per_op']:.1f}" if hit else "-")
        rows.append(row)
    return ("\n## Per-op dispatch cost on a 1-element tensor (microseconds)\n\n"
            "With one element there is no arithmetic left to measure, so this is pure\n"
            "overhead. The ndarray row skips the autograd layer entirely, which separates\n"
            "graph bookkeeping from kernel launch and allocation.\n\n"
            + table(["op"] + [label(b) for b in backends], rows))


def section_anatomy(data) -> str:
    anatomy = data.get("step_anatomy")
    if not anatomy:
        return ""
    out = ["\n## What one step actually asks of the backend\n\n",
           "Counted by wrapping the kernel module in a proxy, so these are real call\n"
           "counts rather than estimates.\n\n"]
    rows = []
    for case, backends in anatomy.items():
        for backend, entry in backends.items():
            fwd, step = entry["forward"], entry["forward_backward"]
            rows.append([
                case, label(backend),
                entry["forward_graph_nodes"],
                fwd["kernel_launches"], step["kernel_launches"],
                step["device_allocations"], step["host_device_transfers"],
            ])
    out.append(table(["config", "backend", "graph nodes", "launches fwd",
                      "launches fwd+bwd", "allocations", "h<->d copies"], rows))

    first = next(iter(anatomy.values()))
    for backend, entry in first.items():
        top = list(entry["forward_backward"]["by_name"].items())[:6]
        out.append(f"\nMost-called kernels in one {label(backend)} step: "
                   + ", ".join(f"{k} x{v}" for k, v in top) + "\n")
    return "".join(out)


SECTIONS = {
    "environment": section_environment,
    "correctness": section_correctness,
    "gemm_summary": section_gemm_summary,
    "speedup": section_speedup,
    "gemm": section_gemm,
    "memory": section_memory,
    "allocator": section_allocator,
    "dispatch": section_dispatch,
    "model": section_model,
    "overhead": section_overhead,
    "anatomy": section_anatomy,
}


def render(data, sections=None) -> str:
    chosen = sections or list(SECTIONS)
    parts = [
        "# nyangrad benchmark results\n",
        f"Generated from `benchmarks/results.json` ({data['environment']['timestamp']}), "
        f"{len(data['records'])} measurements in {data.get('wall_seconds', 0):.0f}s of wall time.\n",
    ]
    if data.get("quick_mode"):
        parts.append("\n> Quick mode: shortened budgets, indicative only.\n")
    for name in chosen:
        parts.append(SECTIONS[name](data))
    return "\n".join(p for p in parts if p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--section", nargs="+", choices=list(SECTIONS))
    parser.add_argument("--out")
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    text = render(data, args.section)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
