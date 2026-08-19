"""Benchmarks for the array backends and for end to end training.

Two things get measured: raw matmul throughput for each backend, and how long an
epoch of MNIST takes with the MLP-ResNet from examples/. Run it as

    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --devices cpu cuda --epochs 2
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import nyangrad as nyan
from nyangrad import ndarray as nd
from mlp_resnet import MLPResNet, epoch

DEVICES = {
    "numpy": nyan.numpy_device,
    "cpu": nyan.cpu,
    "cuda": nyan.cuda,
    "cpu_numpy": nyan.cpu_numpy,
}


def hardware():
    lines = []
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                lines.append("cpu:  " + line.split(":", 1)[1].strip())
                break
    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if out.returncode == 0 and out.stdout.strip():
            lines.append("gpu:  " + out.stdout.strip().splitlines()[0])
    return lines


def sync(array):
    """Wait for queued work to finish; a device to host copy blocks until it is."""
    array[0:1, 0:1].compact().numpy()


def timed(run, budget=1.0):
    """Time run(), repeating it until the budget is spent, and return the best."""
    run()
    best, elapsed, reps = float("inf"), 0.0, 0
    while elapsed < budget or reps < 2:
        start = time.perf_counter()
        run()
        took = time.perf_counter() - start
        best = min(best, took)
        elapsed += took
        reps += 1
        if took > budget:
            break
    return best


def matmul_benchmarks(sizes, naive_limit):
    print("\nmatmul, square n x n, best of several runs")
    print(f"\n| {'n':>5} | {'backend':<16} | {'time (ms)':>10} | {'gflop/s':>8} |")
    print(f"| {'-' * 5} | {'-' * 16} | {'-' * 10} | {'-' * 8} |")

    for n in sizes:
        flops = 2 * n ** 3
        a_np = np.random.randn(n, n).astype("float32")
        b_np = np.random.randn(n, n).astype("float32")

        rows = [("numpy (openblas)", timed(lambda: a_np @ b_np))]

        for name in ("cpu", "cuda"):
            device = DEVICES[name]()
            if not device.enabled():
                continue
            a = nd.array(a_np, device=device)
            b = nd.array(b_np, device=device)

            def run_matmul():
                sync(a @ b)

            rows.append((f"{name}", timed(run_matmul)))

            # the cpu backend picks the tiled kernel whenever the sizes line up,
            # so reach past __matmul__ to time the plain one for comparison
            if name == "cpu" and n <= naive_limit:
                out = nd.NDArray.make((n, n), device=device)

                def run_naive():
                    device.matmul(a.compact()._handle, b.compact()._handle, out._handle, n, n, n)

                rows.append(("cpu (untiled)", timed(run_naive)))

        for label, seconds in rows:
            print(f"| {n:>5} | {label:<16} | {seconds * 1e3:>10.2f} | {flops / seconds / 1e9:>8.2f} |")


def training_benchmarks(device_names, epochs, batch_size, hidden_dim, data_dir):
    train_images = Path(data_dir) / "train-images-idx3-ubyte.gz"
    if not train_images.exists():
        print(f"\nskipping training: no mnist data in {data_dir}/")
        return

    print(f"\nmnist mlp-resnet, batch {batch_size}, hidden {hidden_dim}, {epochs} epoch(s)")
    print(f"\n| {'device':<10} | {'s / epoch':>10} | {'train err':>10} | {'test err':>9} |")
    print(f"| {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 9} |")

    train_set = nyan.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    test_set = nyan.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )

    for name in device_names:
        device = DEVICES[name]()
        if not device.enabled():
            print(f"| {name:<10} | {'unavailable':>10} | {'-':>10} | {'-':>9} |")
            continue

        np.random.seed(4)
        model = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10, device=device)
        opt = nyan.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        train_loader = nyan.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = nyan.data.DataLoader(test_set, batch_size=batch_size)

        start = time.perf_counter()
        for _ in range(epochs):
            train_err, _ = epoch(train_loader, model, opt, device=device)
        seconds = (time.perf_counter() - start) / epochs

        test_err, _ = epoch(test_loader, model, device=device)
        print(f"| {name:<10} | {seconds:>10.1f} | {train_err:>10.4f} | {test_err:>9.4f} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["numpy", "cpu", "cuda"], choices=list(DEVICES))
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--naive-limit", type=int, default=512, help="largest n to time the untiled cpu kernel at")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--skip-matmul", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    for line in hardware():
        print(line)

    if not args.skip_matmul:
        matmul_benchmarks(args.sizes, args.naive_limit)
    if not args.skip_training:
        training_benchmarks(
            args.devices, args.epochs, args.batch_size, args.hidden_dim, args.data_dir
        )


if __name__ == "__main__":
    main()
