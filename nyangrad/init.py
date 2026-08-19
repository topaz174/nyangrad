"""Tensor initializers."""

import math
from typing import Any

import nyangrad as nyan


def rand(
    *shape,
    low=0.0,
    high=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
):
    """Sample a tensor uniformly from [low, high)."""
    device = nyan.default_device() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return nyan.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(
    *shape,
    mean=0.0,
    std=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
):
    """Sample a tensor from a normal distribution."""
    device = nyan.default_device() if device is None else device
    array = device.randn(*shape) * std + mean
    return nyan.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(
    *shape,
    c=1.0,
    device=None,
    dtype="float32",
    requires_grad=False,
):
    """Create a tensor filled with one value."""
    device = nyan.default_device() if device is None else device
    array = device.full(shape, c, dtype=dtype)
    return nyan.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="float32", requires_grad=False):
    """Sample a float mask whose entries are one with probability p."""
    device = nyan.default_device() if device is None else device
    array = device.rand(*shape) <= p
    return nyan.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """Encode integer indices along a new axis of length n."""
    device = nyan.default_device() if device is None else device
    return nyan.Tensor(
        device.one_hot(n, i.numpy().astype("int32"), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    return zeros(
        *array.shape,
        dtype=array.dtype,
        device=array.device if device is None else device,
        requires_grad=requires_grad,
    )


def ones_like(array, *, device=None, requires_grad=False):
    return ones(
        *array.shape,
        dtype=array.dtype,
        device=array.device if device is None else device,
        requires_grad=requires_grad,
    )


def xavier_uniform(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any
) -> "nyan.Tensor":
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def xavier_normal(
    fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any
) -> "nyan.Tensor":
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)


def kaiming_uniform(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    **kwargs: Any,
) -> "nyan.Tensor":
    if nonlinearity != "relu":
        raise ValueError("only relu is supported")
    bound = math.sqrt(2.0) * math.sqrt(3.0 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)


def kaiming_normal(
    fan_in: int,
    fan_out: int,
    nonlinearity: str = "relu",
    **kwargs: Any,
) -> "nyan.Tensor":
    if nonlinearity != "relu":
        raise ValueError("only relu is supported")
    std = math.sqrt(2.0 / fan_in)
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
