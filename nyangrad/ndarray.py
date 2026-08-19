"""Strided n-dimensional array backed by a pluggable compute device."""
import operator
from functools import reduce
from typing import Any, Callable, Iterable, Union

import numpy as np

from . import _numpy_kernels


def prod(x: Iterable[int]) -> int:
    return reduce(operator.mul, x, 1)


class Device:
    """Base class for anything a tensor's data can live on."""


class BackendDevice(Device):
    """A device, i.e. a thin wrapper around one of the kernel modules."""

    def __init__(self, name: str, mod: Any) -> None:
        self.name: str = name
        self.mod: Any = mod

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BackendDevice) and self.name == other.name

    def __repr__(self) -> str:
        return self.name + "()"

    def __hash__(self) -> int:
        return hash(self.name)

    def __getattr__(self, name: str) -> Any:
        if self.mod is None:
            raise RuntimeError(
                f"the {self.name} backend is not available; run `make` to build it"
            )
        return getattr(self.mod, name)

    def enabled(self) -> bool:
        return self.mod is not None

    def randn(self, *shape: int, dtype: str = "float32") -> "NDArray":
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape: int, dtype: str = "float32") -> "NDArray":
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n: int, i: int, dtype: str = "float32") -> "NDArray":
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def array(self, a: Any, dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray(a, device=self)

    def empty(self, shape: tuple[int, ...], dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape: tuple[int, ...], fill_value: float, dtype: str = "float32") -> "NDArray":
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda() -> BackendDevice:
    """Return cuda device"""
    try:
        from . import _cuda_backend  # type: ignore[attr-defined]

        return BackendDevice("cuda", _cuda_backend)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu() -> BackendDevice:
    """Return cpu device, i.e. the native C++ backend"""
    try:
        from . import _cpu_backend  # type: ignore[attr-defined]

        return BackendDevice("cpu", _cpu_backend)
    except ImportError:
        return BackendDevice("cpu", None)


def cpu_numpy() -> BackendDevice:
    """Return the numpy reference device, useful for isolating backend bugs"""
    return BackendDevice("cpu_numpy", _numpy_kernels)


def default_device() -> BackendDevice:
    return cpu()


def all_backend_devices() -> list[BackendDevice]:
    """return a list of all devices backed by an NDArray"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A strided array whose memory and kernels live on a backend device.

    Shape, strides and offset are tracked here in Python, so reshape, permute,
    broadcast and slicing are all zero-copy views over the same handle. Only
    float32 is supported, which is all the backends implement.
    """

    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _device: BackendDevice
    _handle: Any

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other: "NDArray") -> None:
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(
        shape: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        device: BackendDevice | None = None,
        handle: Any = None,
        offset: int = 0,
    ) -> "NDArray":
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def device(self) -> BackendDevice:
        return self._device

    @property
    def dtype(self) -> str:
        # only support float32 for now
        return "float32"

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        return prod(self._shape)

    def __repr__(self) -> str:
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self) -> str:
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value: float) -> None:
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device: BackendDevice) -> "NDArray":
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self) -> np.ndarray:
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self) -> bool:
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self) -> "NDArray":
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape: tuple[int, ...], strides: tuple[int, ...]) -> "NDArray":
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle, offset=self._offset
        )

    @property
    def flat(self) -> "NDArray":
        return self.reshape((self.size,))

    def reshape(self, new_shape: tuple[int, ...]) -> "NDArray":
        """Return a compact-strided view with the same elements under a new shape.

        Raises:
            ValueError if the element count changes or the array is not compact.
        """

        return NDArray.make(
            new_shape,
            strides=NDArray.compact_strides(new_shape),
            device=self.device,
            handle=self._handle,
            offset=self._offset
        )

    def permute(self, new_axes: tuple[int, ...]) -> "NDArray":
        """Reorder the dimensions, e.g. "BHWC" -> "BCHW" with (0, 3, 1, 2).

        Only shape and strides change, so the result shares memory with self.
        """

        return NDArray.make(
            shape=tuple([self.shape[i] for i in new_axes]),
            strides=tuple([self.strides[i] for i in new_axes]),
            device=self._device,
            handle=self._handle,
            offset=self._offset
        )

    def broadcast_to(self, new_shape: tuple[int, ...]) -> "NDArray":
        """Broadcast size-1 dimensions out to new_shape by giving them stride 0.

        No memory is copied, so the result aliases self.
        """

        new_strides = []

        for i in range(len(new_shape)):
            if self.shape[i] == 1 or i >= len(self.shape):
                new_strides.append(0)
            elif self.shape[i] != new_shape[i]:
                raise AssertionError("new_shape[i] must be shape[i] for all i where shape[i] != 1!")
            else:
                new_strides.append(self.strides[i])

        return NDArray.make(
            shape=new_shape,
            strides=tuple(new_strides),
            device=self._device,
            handle=self._handle,
            offset=self._offset
        )

    ### Get and set elements

    def process_slice(self, sl: slice, dim: int) -> slice:
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start is None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop is None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step is None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs: int | slice | tuple[int | slice, ...]) -> "NDArray":
        """Return a strided view of the subset selected by idxs.

        Integers are treated as length-one slices, so the result always keeps
        the same number of dimensions and shares memory with self.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        slices = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(slices) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = []
        new_strides = list(self.strides)
        new_offset = self._offset

        for i in range(len(slices)):
            s = slices[i]

            new_shape.append((s.stop - s.start + s.step - 1) // s.step)
            new_strides[i] *= s.step
            new_offset += s.start * self.strides[i]

        return NDArray.make(
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            device=self._device,
            handle=self._handle,
            offset=new_offset
        )

    def __setitem__(self, idxs: int | slice | tuple[int | slice, ...], other: Union["NDArray", float]) -> None:
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(
        self,
        other: Union["NDArray", float],
        ewise_func: Callable[[Any, Any, Any], None],
        scalar_func: Callable[[Any, Any, Any], None],
    ) -> "NDArray":
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other: Union["NDArray", float]) -> "NDArray":
        return self + (-other)

    def __rsub__(self, other: Union["NDArray", float]) -> "NDArray":
        return other + (-self)

    def __mul__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self) -> "NDArray":
        return self * (-1)

    def __pow__(self, other: float) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other: Union["NDArray", float]) -> "NDArray":
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other: Any) -> "NDArray":
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other: Any) -> "NDArray":  # type: ignore[override]
        return 1 - (self == other)

    def __gt__(self, other: Any) -> "NDArray":
        return (self >= other) * (self != other)

    def __lt__(self, other: Any) -> "NDArray":
        return 1 - (self >= other)

    def __le__(self, other: Any) -> "NDArray":
        return 1 - (self > other)

    ### Elementwise functions

    def log(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self) -> "NDArray":
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other: "NDArray") -> "NDArray":
        """Multiply two 2D arrays.

        When every dimension is a multiple of the backend's tile size, both
        operands are restrided into tiles and handed to the tiled CPU kernel;
        otherwise this falls through to the plain kernel. The CUDA kernel tiles
        internally, so it always takes the plain path.
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis: int | tuple[int, ...] | list[int] | None, keepdims: bool = False) -> tuple["NDArray", "NDArray"]:
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            #out = NDArray.make((1,) * self.ndim, device=self.device)
            out = NDArray.make((1,), device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out

    def sum(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis: int | tuple[int, ...] | list[int] | None = None, keepdims: bool = False) -> "NDArray":
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out


def array(a: Any, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape: tuple[int, ...], dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape: tuple[int, ...], fill_value: float, dtype: str = "float32", device: BackendDevice | None = None) -> NDArray:
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.broadcast_to(new_shape)


def reshape(array: NDArray, new_shape: tuple[int, ...]) -> NDArray:
    return array.reshape(new_shape)


def maximum(a: NDArray, b: NDArray | float) -> NDArray:
    return a.maximum(b)


def log(a: NDArray) -> NDArray:
    return a.log()


def exp(a: NDArray) -> NDArray:
    return a.exp()


def tanh(a: NDArray) -> NDArray:
    return a.tanh()


def sum(a: NDArray, axis: int | tuple[int] | list[int] | None = None, keepdims: bool = False) -> NDArray:
    return a.sum(axis=axis, keepdims=keepdims)
