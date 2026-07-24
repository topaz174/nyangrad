"""The module.
"""
from typing import Any
from nyangrad.autograd import Tensor
from nyangrad import ops
import nyangrad.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((1, out_features)))

    def forward(self, X: Tensor) -> Tensor:
        XW = X.matmul(self.weight)
        return XW + self.bias.broadcast_to(XW.shape)


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        flattened_dim = 1

        for i in range(1, len(X.shape)):
            flattened_dim *= X.shape[i]

        return X.reshape((X.shape[0], flattened_dim))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)

        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        y_one_hot = init.one_hot(logits.shape[1], y)
        lsm = ops.logsoftmax(logits)

        return -(ops.summation(lsm * y_one_hot) / logits.shape[0])


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        m = x.shape[0]
        d = x.shape[1]

        if self.training:
            mean = ((ops.summation(x, axes=0)) / m)
            mean_b = mean.reshape((1, d)).broadcast_to((m, d))

            var = ((ops.summation((x - mean_b) ** 2, axes=0)) / m)
            var_b = var.reshape((1, d)).broadcast_to((m, d))

            self.running_mean.data = ((1 - self.momentum) * self.running_mean.data + self.momentum * mean.data)
            self.running_var.data = ((1 - self.momentum) * self.running_var.data + self.momentum * var.data)

            norm = (x - mean_b) / ((var_b + self.eps) ** (1/2))

        else:
            norm = (x - self.running_mean.broadcast_to((m, d))) / ((self.running_var.broadcast_to((m, d)) + self.eps) ** (1/2))


        w_b = self.weight.reshape((1, d)).broadcast_to((m, d))
        b_b = self.bias.reshape((1, d)).broadcast_to((m, d))
        return w_b * norm + b_b


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        m = x.shape[0]
        d = x.shape[1]

        mean = (ops.summation(x, axes=1) / d).reshape((m, 1)).broadcast_to((m, d))
        var = (ops.summation((x - mean) ** 2, axes=1) / d).reshape((m, 1)).broadcast_to((m, d))

        norm = (x - mean) / ((var + self.eps) ** (1/2))

        w_b = self.weight.reshape((1, d)).broadcast_to((m, d))
        b_b = self.bias.reshape((1, d)).broadcast_to((m, d))
        return w_b * norm + b_b


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p)

            x = x * mask

            x /= 1.0 - self.p

        return x


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x
