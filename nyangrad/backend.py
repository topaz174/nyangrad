"""Devices a tensor's data can live on, and which one gets used by default."""
import numpy

from .ndarray import (
    BackendDevice,
    Device,
    NDArray,
    all_backend_devices,
    cpu,
    cpu_numpy,
    cuda,
)


class NumpyDevice(Device):
    """Holds plain numpy arrays instead of an NDArray.

    This is the portable default and the only backend with dtype support beyond
    float32, which is useful for finite-difference checks.
    """

    def __repr__(self):
        return "nyangrad.numpy_device()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, NumpyDevice)

    def enabled(self):
        return True

    def array(self, a, dtype="float32"):
        return numpy.array(a, dtype=dtype)

    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        return numpy.random.randn(*shape)

    def rand(self, *shape):
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]

    def empty(self, shape, dtype="float32"):
        return numpy.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return numpy.full(shape, fill_value, dtype=dtype)


def numpy_device():
    """Return a device backed by NumPy arrays."""
    return NumpyDevice()


_default_device = numpy_device()


def default_device():
    """Device used whenever a tensor is created without an explicit one.

    Tensor ops still compute through numpy, so this is the numpy device; the
    NDArray devices below are driven directly through the array API for now.
    """
    return _default_device


def set_default_device(device):
    """Set the device new tensors land on, e.g. set_default_device(cuda())."""
    global _default_device
    if not device.enabled():
        raise RuntimeError(f"{device} is not available")
    _default_device = device


def all_devices():
    """Return every known device, including optional disabled backends."""
    return [numpy_device()] + all_backend_devices()
