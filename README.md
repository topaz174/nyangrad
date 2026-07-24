# nyangrad

nyangrad is a small deep learning framework I wrote from scratch on top of NumPy. It has its own
reverse-mode automatic differentiation engine, a tensor type that records a computation graph as you
use it, and enough of a neural network library on top (layers, losses, initializers, optimizers, a
data loader) to actually train models end to end. The MNIST MLP-ResNet in `examples/` trains with it.

The whole point was to understand how a framework like PyTorch works underneath instead of treating
`loss.backward()` as magic. So the code leans towards being explicit and readable rather than fast.
Everything is plain NumPy, no autodiff libraries, and every gradient is written out by hand and checked
numerically in the tests.

If you want the short version: it is a tensor library with autograd, a small `nn` module, `optim`,
`init`, and `data`, plus a couple of training scripts.


## What is in here

- `nyangrad/autograd.py` - the core. The `Value`/`Tensor` types, the computation graph, the topological
  sort, and the reverse-mode autodiff pass (`backward`).
- `nyangrad/ops.py` - every tensor operation and its gradient. Elementwise add/mul/div/pow (and scalar
  versions), matmul, reshape, transpose, broadcast, summation, negate, log, exp, relu, and the
  numerically stable logsumexp / logsoftmax. Also the small tuple ops.
- `nyangrad/nn.py` - the module system and the layers: Linear, Flatten, ReLU, Sequential, SoftmaxLoss,
  BatchNorm1d, LayerNorm1d, Dropout, Residual.
- `nyangrad/init.py` - weight initializers: xavier and kaiming, uniform and normal, plus the basic
  random/constant/one-hot helpers.
- `nyangrad/optim.py` - SGD (with momentum and weight decay) and Adam (with bias correction and weight
  decay).
- `nyangrad/data.py` - the Dataset / DataLoader abstractions, a couple of image transforms, and the
  MNIST loader.
- `nyangrad/backend.py` - a tiny device abstraction. Right now there is only a NumPy CPU device, but the
  tensor code goes through this layer so a different backend could be dropped in later.
- `examples/` - training scripts. `mlp_resnet.py` builds and trains the MLP-ResNet on MNIST,
  `two_layer_net.py` is a plain two layer network written directly against the autograd engine.
- `tests/` - the correctness tests I used while building this. They check forward values, and check every
  gradient against a finite-difference numerical gradient.
- `sandbox/` - earlier warmup work, including a C++/pybind11 version of a training step. Kept around
  because it is where a lot of the ideas started, but it is not part of the core library.


## The autograd engine

This is the part I care about the most, so here is roughly how it works.

Every tensor is a node in a graph. A node either is a leaf (it was created directly from data) or it was
produced by an operation applied to other nodes. Each node holds a reference to the op that made it and
the input nodes that went into it, so the graph is built implicitly just by doing math on tensors.

- An `Op` is an object with two methods: `compute`, which does the actual forward NumPy computation on
  raw arrays, and `gradient`, which, given the gradient flowing in from the output side, returns the
  gradient with respect to each input. Splitting it this way means the forward and backward for each
  operation live right next to each other and you can read them together.
- When you call something like `a + b`, a new tensor is made from an `EWiseAdd` op with `a` and `b` as
  inputs. The forward result is computed and cached. Nothing about the graph is thrown away, so it is
  available later for the backward pass.
- `backward()` starts from the output, does a post-order depth-first traversal to get a topological
  ordering of the graph, and then walks it in reverse. For each node it sums the gradient contributions
  coming from everywhere it was used, then pushes gradients back to that node's inputs using the op's
  `gradient` method. This is the standard reverse-mode accumulation, written out plainly.

A few things that were more subtle than I expected:

- Broadcasting. NumPy will happily broadcast a `(4,)` bias across a `(32, 4)` batch in the forward pass,
  but in the backward pass that means the gradient has to be summed back down to the original shape. The
  gradients for broadcast, summation, and matmul all have to carefully undo whatever broadcasting NumPy
  did, which is where most of the fiddly reshaping in `ops.py` comes from.
- Matmul with batched / mismatched ranks. The gradient has to handle the case where the two operands
  have different numbers of dimensions and sum over the leading batch axes that only one side had.
- Numerical stability. A naive softmax or logsumexp overflows the moment the logits get large, so those
  are done with the standard max-subtraction trick. There is a test that throws values like 1e10 at it
  to make sure it does not blow up.
- Keeping the graph light. Optimizers use detached tensors / `.data` when updating parameters so that the
  update step does not itself get recorded into the graph and leak memory. There is a test that counts
  live tensors to catch this.


## The nn layer

The module system is deliberately simple. A `Module` finds its parameters and child modules by looking
through its own `__dict__`, recursing into lists, tuples, and dicts. So you just assign layers and
parameters as attributes and `.parameters()` finds them. `train()` / `eval()` flip a flag that layers
like Dropout and BatchNorm read.

On top of that the usual pieces are implemented from their definitions: Linear with kaiming init,
BatchNorm1d with running statistics, LayerNorm1d, Dropout with inverted scaling, a Residual wrapper, and
a SoftmaxLoss that uses the stable logsoftmax. `examples/mlp_resnet.py` wires these into a residual MLP.


## Usage

Install it in editable mode so the package is importable:

```bash
pip install -r requirements.txt
pip install -e .
```

A quick taste of the autograd:

```python
import nyangrad as ndl
import numpy as np

x = ndl.Tensor(np.random.randn(3, 4))
w = ndl.Tensor(np.random.randn(4, 2))

y = ndl.relu(x @ w).sum()
y.backward()

print(x.grad.shape)  # (3, 4)
print(w.grad.shape)  # (4, 2)
```

Training the MLP-ResNet on MNIST:

```bash
python examples/mlp_resnet.py
```


## Tests

The tests are the thing I trust. Run them with:

```bash
pytest tests/
```

Most of them work the same way: build a small function out of the ops, compute the analytic gradient
through the engine, compute a finite-difference numerical gradient, and assert they match. The nn and
optimizer tests go further and check exact forward and backward values, running statistics, training and
eval mode behavior, and that a few small models actually train down in loss. Division is done in float32,
so its numerical gradient check runs at a float32-appropriate tolerance.


## Notes

- It is CPU and NumPy only. It is built for clarity, not speed, and there is no GPU backend (though the
  device layer is there so one could be added).
- This project grew out of working through CMU's 10-714 (Deep Learning Systems). I used the course as the
  roadmap for what to build and in what order, then wrote and organized the implementation as its own
  standalone framework.
