"""Dynamic computation graphs and reverse-mode automatic differentiation."""

from functools import reduce
from operator import add
from typing import Dict, List, Optional, Tuple, Union

import numpy

import nyangrad
from .backend import (
    Device,
    all_devices,
    cpu,
    cpu_numpy,
    cuda,
    default_device,
    numpy_device,
    set_default_device,
)
from .ndarray import NDArray as _StridedArray

from nyangrad import init

LAZY_MODE = False
TENSOR_COUNTER = 0

import numpy as array_api

# Ops work on either NumPy arrays or nyangrad's strided arrays.
NDArray = Union[numpy.ndarray, _StridedArray]


class Op:
    """The forward and backward definition of one graph operation."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Run the forward pass on raw arrays and return the result."""
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Given the gradient w.r.t. this op's output, return the gradient(s)
        w.r.t. each of its inputs, expressed as ops on Tensors so the whole
        thing stays differentiable."""
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Normalize single- and multi-input gradients to a tuple."""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """An operation with a single Tensor output."""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """An operation with a TensorTuple output."""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    op: Optional[Op]
    inputs: List["Value"]
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Compute this value once, then reuse the cached result."""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class TensorTuple(Value):
    """A graph value containing a flat tuple of tensors."""

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return nyangrad.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "nyangrad.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return nyangrad.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        return device.array(numpy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy arrays and scalars carry no device of their own
        if isinstance(data, _StridedArray):
            return data.device
        return numpy_device()

    def to(self, device: Device) -> "Tensor":
        """Return a copy of this tensor on another device."""
        if device == self.device:
            return self
        return Tensor(self.numpy(), device=device, requires_grad=self.requires_grad)

    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "nyangrad.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if isinstance(data, _StridedArray):
            return data.numpy()
        return numpy.array(data)

    def __add__(self, other):
        if isinstance(other, Tensor):
            return nyangrad.ops.EWiseAdd()(self, other)
        else:
            return nyangrad.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return nyangrad.ops.EWiseMul()(self, other)
        else:
            return nyangrad.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return nyangrad.ops.EWisePow()(self, other)
        else:
            return nyangrad.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return nyangrad.ops.EWiseAdd()(self, nyangrad.ops.Negate()(other))
        else:
            return nyangrad.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return nyangrad.ops.EWiseDiv()(self, other)
        else:
            return nyangrad.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return nyangrad.ops.MatMul()(self, other)

    def matmul(self, other):
        return nyangrad.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return nyangrad.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return nyangrad.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return nyangrad.ops.Reshape(shape)(self)

    def __neg__(self):
        return nyangrad.ops.Negate()(self)

    def transpose(self, axes=None):
        return nyangrad.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__




def compute_gradient_of_variables(output_tensor, out_grad):
    """Accumulate reverse-mode gradients from the output tensor into the graph."""
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {
        output_tensor: [out_grad]
    }
    reverse_topo_order = reversed(find_topo_sort([output_tensor]))

    for node in reverse_topo_order:
        node.grad = sum_node_list(node_to_output_grads_list[node])
        input_grads = node.op.gradient_as_tuple(node.grad, node) if node.op else ()
        for input_node, input_grad in zip(node.inputs, input_grads):
            node_to_output_grads_list.setdefault(input_node, []).append(input_grad)



def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Return the graph nodes in input-to-output topological order."""
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)

    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Append one graph branch using a post-order depth-first traversal."""
    for input_node in node.inputs:
        if input_node not in visited:
            topo_sort_dfs(input_node, visited, topo_order)

    topo_order.append(node)
    visited.add(node)


def sum_node_list(node_list):
    """Add gradient contributions without Python's leading zero."""
    return reduce(add, node_list)
