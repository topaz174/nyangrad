"""Count what a nyangrad step actually asks of the backend.

A `BackendDevice` forwards unknown attributes to its kernel module, so wrapping
that module in a counting proxy makes every kernel launch and every device
allocation visible without touching a line of framework code. The proxy is
installed on a device instance the benchmark owns, so nothing global changes.
"""
from __future__ import annotations

from collections import Counter


class CountingModule:
    """Proxy around a kernel module that tallies calls by name."""

    # these are allocations rather than compute
    ALLOC_NAMES = {"Array"}
    TRANSFER_NAMES = {"to_numpy", "from_numpy"}

    def __init__(self, wrapped) -> None:
        self._wrapped = wrapped
        self.calls: Counter[str] = Counter()

    def __getattr__(self, name):
        attr = getattr(self._wrapped, name)
        if not callable(attr):
            return attr

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return attr(*args, **kwargs)

        return counted

    def reset(self) -> None:
        self.calls.clear()

    def report(self) -> dict:
        allocations = sum(v for k, v in self.calls.items() if k in self.ALLOC_NAMES)
        transfers = sum(v for k, v in self.calls.items() if k in self.TRANSFER_NAMES)
        launches = sum(
            v for k, v in self.calls.items()
            if k not in self.ALLOC_NAMES and k not in self.TRANSFER_NAMES
        )
        return {
            "kernel_launches": launches,
            "device_allocations": allocations,
            "host_device_transfers": transfers,
            "by_name": dict(sorted(self.calls.items(), key=lambda kv: -kv[1])),
        }


def instrumented_device(device):
    """Wrap a device's kernel module in place and return the counter."""
    counter = CountingModule(device.mod)
    device.mod = counter
    return counter


def graph_size(tensor) -> int:
    """Number of nodes in the graph feeding this tensor."""
    from nyangrad.autograd import find_topo_sort

    return len(find_topo_sort([tensor]))
