"""Run the whole benchmark suite and write benchmarks/results.json.

    python benchmarks/run_all.py                  # everything
    python benchmarks/run_all.py --suites gemm    # just the GEMM scaling
    python benchmarks/run_all.py --quick          # short budgets, for a smoke test

Nothing here imports from the framework except through its public API, and no
framework state is mutated, so a benchmark run cannot change how nyangrad behaves.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]

import harness
from harness import ClockMonitor, collect_environment, write_results

SUITES = ("gemm", "memory", "model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=list(SUITES), choices=SUITES)
    parser.add_argument("--out", default=str(Path(__file__).parent / "results.json"))
    parser.add_argument("--quick", action="store_true", help="short budgets for a smoke test")
    parser.add_argument("--skip-ceilings", action="store_true")
    args = parser.parse_args()

    if args.quick:
        harness.TARGET_SAMPLE_MS = 1.0
        harness.SINGLE_SHOT_S = 0.3

    started = time.perf_counter()
    print("collecting environment")
    environment = collect_environment()
    for line in (
        environment["cpu"]["name"],
        environment.get("gpu", {}).get("name", "no gpu"),
        f"numpy {environment['numpy']['version']} on {environment['numpy']['blas']}",
    ):
        print(f"  {line}")

    payload = {
        "schema": 1,
        "environment": environment,
        "methodology": {
            "timing": "batched launches, one device synchronise per sample, median reported",
            "target_sample_ms": harness.TARGET_SAMPLE_MS,
            "single_shot_threshold_s": harness.SINGLE_SHOT_S,
            "aggregation": "median of per-repetition latency; p10/p90 and IQR recorded",
            "warmup": "budget-derived, discarded",
            "l2_flush": "applied to the memory suite and the CPU GEMM kernels",
            "tf32": "disabled for every torch CUDA baseline, matching nyangrad's true fp32",
            "ceilings": "measured on this machine, not taken from spec sheets",
        },
        "records": [],
        "correctness": [],
        "quick_mode": args.quick,
    }

    with ClockMonitor() as monitor:
        if not args.skip_ceilings:
            print("measuring machine ceilings")
            import ceilings

            payload["ceilings"] = ceilings.measure_all()
            settle = payload["ceilings"].get("clock_settle") or {}
            if settle.get("applied"):
                print(f"  settled clocks for {settle['seconds']:.0f}s: "
                      f"{settle['sm_mhz_before']:.0f} -> {settle['sm_mhz_after']:.0f} MHz")
            for key, value in payload["ceilings"].items():
                if not value or key == "clock_settle":
                    continue
                if "gb_per_s" in value:
                    print(f"  {key:<16} {value['gb_per_s']:>9.1f} GB/s")
                elif "gflops" in value:
                    print(f"  {key:<16} {value['gflops']:>9.1f} GFLOP/s")
                else:
                    print(f"  {key:<16} {value['gflops_single_thread']:>9.1f} GFLOP/s (1 thread)"
                          f"  {value['gflops_all_threads']:>9.1f} GFLOP/s (all)")
        else:
            payload["ceilings"] = {}

        if "gemm" in args.suites:
            print("suite: gemm")
            import bench_gemm

            payload["correctness"] += bench_gemm.verify_correctness()
            payload["records"] += [r.to_dict() for r in bench_gemm.run(payload["ceilings"])]

        if "memory" in args.suites:
            print("suite: memory / compaction")
            import bench_memory

            payload["correctness"] += bench_memory.verify_correctness()
            payload["records"] += [r.to_dict() for r in bench_memory.run(payload["ceilings"])]
            print("  allocator")
            payload["records"] += [r.to_dict() for r in bench_memory.allocator_records()]
            print("  dispatch anatomy")
            payload["records"] += [r.to_dict() for r in bench_memory.dispatch_anatomy()]

        if "model" in args.suites:
            print("suite: model")
            import bench_model

            payload["correctness"] += bench_model.verify_correctness()
            payload["records"] += [r.to_dict() for r in bench_model.run(payload["ceilings"])]
            print("  per-op overhead")
            payload["records"] += [r.to_dict() for r in bench_model.op_overhead()]
            print("  step anatomy")
            payload["step_anatomy"] = bench_model.step_anatomy()

    payload["clocks"] = monitor.summary()
    payload["wall_seconds"] = round(time.perf_counter() - started, 1)

    out = Path(args.out)
    write_results(payload, out)
    print(f"\n{len(payload['records'])} records in {payload['wall_seconds']}s -> {out}")

    failed = [c for c in payload["correctness"] if not c["passed"]]
    if failed:
        print("\ncorrectness failures:")
        for check in failed:
            print(f"  {check}")
        sys.exit(1)
    print(f"all {len(payload['correctness'])} correctness checks passed")


if __name__ == "__main__":
    main()
