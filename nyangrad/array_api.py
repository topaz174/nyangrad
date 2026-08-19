"""The array operations the op layer computes with.

These keep numpy's names and signatures but dispatch on the array type, so one
op implementation covers a plain numpy array and an NDArray living on the cpu or
cuda backend. Where numpy is more permissive than NDArray, for instance implicit
broadcasting or reductions over several axes at once, the NDArray path spells the
extra steps out here rather than in every op.
"""
import numpy

from .ndarray import NDArray, prod


def _strided(a) -> bool:
    return isinstance(a, NDArray)


def reshape(a, shape):
    if _strided(a):
        return a.compact().reshape(_fill_in_wildcard(shape, a.size))
    return numpy.reshape(a, shape)


def _fill_in_wildcard(shape, size):
    """Resolve numpy's -1 placeholder, which NDArray shapes do not take."""
    shape = tuple(shape)
    if -1 not in shape:
        return shape
    known = prod(s for s in shape if s != -1)
    return tuple(size // known if s == -1 else s for s in shape)


def broadcast_to(a, shape):
    shape = tuple(shape)
    if not _strided(a):
        return numpy.broadcast_to(a, shape)
    if a.shape == shape:
        return a
    # numpy lines shapes up from the right; NDArray wants matching rank
    if a.ndim < len(shape):
        a = a.compact().reshape((1,) * (len(shape) - a.ndim) + a.shape)
    return a.broadcast_to(shape)


def _align(a, b):
    """Broadcast two arrays to a common shape the way numpy would."""
    if not (_strided(a) and _strided(b)):
        return a, b
    if a.shape == b.shape:
        return a, b
    shape = numpy.broadcast_shapes(a.shape, b.shape)
    return broadcast_to(a, shape), broadcast_to(b, shape)


def add(a, b):
    a, b = _align(a, b)
    return a + b


def multiply(a, b):
    a, b = _align(a, b)
    return a * b


def divide(a, b):
    a, b = _align(a, b)
    return a / b


def power(a, b):
    if _strided(a) or _strided(b):
        raise NotImplementedError(
            "elementwise power has no kernel on the native backends; "
            "raise to a scalar power instead"
        )
    return a ** b


def swapaxes(a, axis1, axis2):
    if not _strided(a):
        return numpy.swapaxes(a, axis1, axis2)
    axes = list(range(a.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return a.permute(tuple(axes))


def _reduce(a, axis, keepdims, name):
    """Apply a single-axis NDArray reduction repeatedly, one axis at a time."""
    if axis is None:
        axes = tuple(range(a.ndim))
    elif isinstance(axis, int):
        axes = (axis % a.ndim,)
    else:
        axes = tuple(sorted(ax % a.ndim for ax in axis))

    out = a
    for ax in axes:
        out = getattr(out, name)(axis=ax, keepdims=True)
    if keepdims:
        return out
    kept = tuple(s for i, s in enumerate(out.shape) if i not in axes)
    return out.compact().reshape(kept if kept else (1,))


def sum(a, axis=None, keepdims=False):
    if _strided(a):
        return _reduce(a, axis, keepdims, "sum")
    return numpy.sum(a, axis=axis, keepdims=keepdims)


def max(a, axis=None, keepdims=False):
    if _strided(a):
        return _reduce(a, axis, keepdims, "max")
    return numpy.max(a, axis=axis, keepdims=keepdims)


def log(a):
    return a.log() if _strided(a) else numpy.log(a)


def exp(a):
    return a.exp() if _strided(a) else numpy.exp(a)


def tanh(a):
    return a.tanh() if _strided(a) else numpy.tanh(a)


def maximum(a, b):
    return a.maximum(b) if _strided(a) else numpy.maximum(a, b)


def matmul(a, b):
    if not _strided(a) and not _strided(b):
        return a @ b
    if a.ndim == 2 and b.ndim == 2:
        return a @ b

    m, n, p = a.shape[-2], a.shape[-1], b.shape[-1]
    if b.ndim == 2:
        # one shared right operand, so the batch can fold into the row dimension
        flat = a.compact().reshape((prod(a.shape[:-1]), n))
        return (flat @ b).compact().reshape(a.shape[:-2] + (m, p))

    lead = numpy.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a = broadcast_to(a, lead + (m, n))
    b = broadcast_to(b, lead + (n, p))
    out = NDArray.make(lead + (m, p), device=a.device)
    for index in numpy.ndindex(lead):
        block = index + (slice(None), slice(None))
        out[block] = a[block].compact().reshape((m, n)) @ b[block].compact().reshape((n, p))
    return out
