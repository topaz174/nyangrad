from . import ops
from .ops import *
from .autograd import Tensor

from . import ndarray
from .ndarray import NDArray
from .backend import (
    all_devices,
    cpu,
    cpu_numpy,
    cuda,
    default_device,
    numpy_device,
    set_default_device,
)

from . import init
from .init import ones, zeros, zeros_like, ones_like

from . import data
from . import nn
from . import optim
