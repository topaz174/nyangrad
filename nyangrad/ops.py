"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy
import numpy as array_api

BACKEND = "np"

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a ** b
        
    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad * b * (a ** (b - 1)), out_grad * (a ** b) * log(a)

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return (a ** self.scalar).astype("float32")

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return (a / b).astype("float32")

    def gradient(self, out_grad, node):
        a, b = node.inputs
        return out_grad / b, out_grad * (-a / (b ** 2))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes:
            return numpy.swapaxes(a, *self.axes)
        else:
            return numpy.swapaxes(a, -1, -2)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad.reshape(a.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return numpy.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        adims = len(a.shape)
        ogdims = len(out_grad.shape)
        diff = ogdims - adims
        
        agrad = Tensor.sum(out_grad, axes=tuple(range(diff)))

        for i in range(adims):
            if out_grad.shape[i] > a.shape[i]:
                agrad = Tensor.sum(agrad, axes=i)
        
        return agrad.reshape(a.shape)



def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return numpy.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        padding = [] 

        if self.axes is None:
            axesTuple = tuple(range(len(a.shape)))
        elif isinstance(self.axes, int):
            axesTuple = (self.axes,)
        else:
            axesTuple = self.axes

        for i in range(len(a.shape)):
            if i in axesTuple:
                padding.append(1)
            else:
                padding.append(a.shape[i])

        out_grad_padded = reshape(out_grad, padding)
        return Tensor.broadcast_to(out_grad_padded, a.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        adims = len(a.shape)
        bdims = len(b.shape)
        diff = abs(adims - bdims)

        if adims == bdims:
            agrad = out_grad @ b.transpose()
            bgrad = a.transpose() @ out_grad
        elif adims > bdims:
            agrad = out_grad @ b.transpose()
            bgrad = Tensor.sum(a.transpose() @ out_grad, axes=tuple(range(diff))).reshape(b.shape)
        else:
            agrad = Tensor.sum(out_grad @ b.transpose(), axes=tuple(range(diff))).reshape(a.shape)
            bgrad = a.transpose() @ out_grad
        
        return agrad, bgrad



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return numpy.log(a)

    def gradient(self, out_grad, node):
        a = node.inputs[0]
        return out_grad * a ** -1


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return numpy.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * node


def exp(a):
    return Exp()(a)

# class IsPositive(TensorOp):
#     def compute(self, a):
#         a[a > 0] = 1
#         return a
#     def gradient(self, out_grad, node):
#         return None


# def is_positive(a):
#     return IsPositive()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return numpy.maximum(0, a)

    def gradient(self, out_grad, node):
        node_numpy = node.numpy()
        node_grad_numpy = (node_numpy > 0).astype(numpy.float32)

        node_grad = Tensor(node_grad_numpy, dtype="float32")

        return out_grad * node_grad


def relu(a):
    return ReLU()(a)



class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        maxes = array_api.max(Z, axis=1, keepdims=True)
        Z_lse = maxes + array_api.log(array_api.sum(array_api.exp(Z - maxes), axis=1, keepdims=True))

        return Z - Z_lse

    def gradient(self, out_grad: Tensor, node: Tensor):
        softmax = exp(node)

        return out_grad - (summation(out_grad, axes=1).reshape((node.shape[0], 1)) * softmax)


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        maxes = array_api.max(Z, axis=self.axes, keepdims=True)
        maxes_collapsed = array_api.max(Z, axis=self.axes)
        
        return maxes_collapsed + array_api.log(array_api.sum(array_api.exp(Z - maxes), axis=self.axes))

    def gradient(self, out_grad: Tensor, node: Tensor):
        h = node.inputs[0]

        dims = list(h.shape)

        if self.axes:
            for axis in self.axes:
                dims[axis] = 1
        else:
            dims[:] = [1] * len(dims)

        node_reshaped = node.reshape(tuple(dims))

        softmax = exp(h - node_reshaped)

        return out_grad.reshape(tuple(dims)) * softmax


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)

class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)
