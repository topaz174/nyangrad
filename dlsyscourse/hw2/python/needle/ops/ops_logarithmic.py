from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxes = array_api.max(Z, axis=1, keepdims=True)
        Z_lse = maxes + array_api.log(array_api.sum(array_api.exp(Z - maxes), axis=1, keepdims=True))

        return Z - Z_lse
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        softmax = exp(node)

        return out_grad - (summation(out_grad, axes=1).reshape((node.shape[0], 1)) * softmax)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        maxes = array_api.max(Z, axis=self.axes, keepdims=True)
        maxes_collapsed = array_api.max(Z, axis=self.axes)
        
        return maxes_collapsed + array_api.log(array_api.sum(array_api.exp(Z - maxes), axis=self.axes))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
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
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)