"""Every op and layer should give the same answer whichever device it runs on.

The numpy device is the reference: each case is run twice from identical inputs,
once on numpy and once on the device under test, and both the forward value and
the gradients have to match to float32 tolerance.
"""
import numpy as np
import pytest

import nyangrad as nyan
import nyangrad.nn as nn

_DEVICES = [
    pytest.param(
        nyan.cpu(),
        marks=pytest.mark.skipif(not nyan.cpu().enabled(), reason="run make first"),
    ),
    pytest.param(
        nyan.cuda(),
        marks=pytest.mark.skipif(not nyan.cuda().enabled(), reason="No GPU"),
    ),
]

ATOL = 1e-4


def _inputs(shapes, device, scale=1.0, positive=False):
    np.random.seed(0)
    arrays = [np.random.randn(*shape).astype("float32") * scale for shape in shapes]
    if positive:
        arrays = [np.abs(a) + 0.1 for a in arrays]
    return [nyan.Tensor(a, device=device, dtype="float32") for a in arrays]


def check(fn, *shapes, device, positive=False, scale=1.0, **kwargs):
    """Compare fn's forward pass and gradients against the numpy device."""
    ref = _inputs(shapes, nyan.numpy_device(), scale=scale, positive=positive)
    got = _inputs(shapes, device, scale=scale, positive=positive)

    out_ref = fn(*ref, **kwargs)
    out_got = fn(*got, **kwargs)
    np.testing.assert_allclose(
        out_got.numpy().reshape(out_ref.shape), out_ref.numpy(), rtol=1e-4, atol=ATOL
    )

    out_ref.sum().backward()
    out_got.sum().backward()
    for expected, actual in zip(ref, got):
        np.testing.assert_allclose(
            actual.grad.numpy().reshape(expected.grad.shape),
            expected.grad.numpy(),
            rtol=1e-4,
            atol=ATOL,
        )


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_ewise_ops(device):
    check(nyan.add, (5, 4), (5, 4), device=device)
    check(nyan.multiply, (5, 4), (5, 4), device=device)
    check(nyan.divide, (5, 4), (5, 4), device=device, positive=True)
    check(lambda a: nyan.add_scalar(a, 2.5), (5, 4), device=device)
    check(lambda a: nyan.mul_scalar(a, 2.5), (5, 4), device=device)
    check(lambda a: nyan.divide_scalar(a, 2.5), (5, 4), device=device)
    check(lambda a: nyan.power_scalar(a, 2), (5, 4), device=device)
    check(nyan.negate, (5, 4), device=device)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_ewise_functions(device):
    check(nyan.log, (5, 4), device=device, positive=True)
    check(nyan.exp, (5, 4), device=device)
    check(nyan.relu, (5, 4), device=device)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_broadcasting_ops_share_shapes_with_numpy(device):
    # the ewise kernels need equal shapes, so the op layer has to line these up
    check(lambda a, b: a * b, (5, 4), (5, 1), device=device)
    check(lambda a, b: a + b, (5, 4), (1, 4), device=device)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_view_ops(device):
    check(nyan.reshape, (5, 4), device=device, shape=(4, 5))
    check(nyan.transpose, (5, 4), device=device)
    check(nyan.transpose, (3, 5, 4), device=device, axes=(1, 2))
    check(nyan.broadcast_to, (5, 1), device=device, shape=(5, 4))
    check(nyan.broadcast_to, (4,), device=device, shape=(3, 4))


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_reductions(device):
    check(nyan.summation, (5, 4), device=device)
    check(nyan.summation, (5, 4), device=device, axes=1)
    check(nyan.summation, (5, 4), device=device, axes=0)
    check(nyan.summation, (5, 4, 3), device=device, axes=(0, 2))
    check(nyan.logsumexp, (5, 4), device=device, axes=(1,))
    check(nyan.logsoftmax, (5, 4), device=device)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_matmul(device):
    check(nyan.matmul, (5, 4), (4, 3), device=device)
    check(nyan.matmul, (16, 16), (16, 16), device=device)  # hits the tiled path
    check(nyan.matmul, (6, 6, 5, 4), (6, 6, 4, 3), device=device)
    check(nyan.matmul, (6, 6, 5, 4), (4, 3), device=device)
    check(nyan.matmul, (5, 4), (6, 6, 4, 3), device=device)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_nn_layers(device):
    np.random.seed(1)
    w = np.random.randn(4, 6).astype("float32")
    b = np.random.randn(1, 6).astype("float32")

    def linear(x):
        layer = nn.Linear(4, 6, device=x.device)
        # the layers init their own weights, so pin them to compare across devices
        layer.weight = nn.Parameter(nyan.Tensor(w, device=x.device, dtype="float32"))
        layer.bias = nn.Parameter(nyan.Tensor(b, device=x.device, dtype="float32"))
        return layer(x)

    check(linear, (8, 4), device=device)
    check(lambda x: nn.LayerNorm1d(4, device=x.device)(x), (8, 4), device=device)
    check(lambda x: nn.BatchNorm1d(4, device=x.device)(x), (8, 4), device=device)
    check(lambda x: nn.Flatten()(x), (8, 2, 3), device=device)
    check(lambda x: nn.ReLU()(x), (8, 4), device=device)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_softmax_loss(device):
    np.random.seed(2)
    logits = np.random.randn(8, 5).astype("float32")
    labels = np.random.randint(0, 5, size=(8,)).astype("float32")

    losses = []
    for dev in (nyan.numpy_device(), device):
        x = nyan.Tensor(logits, device=dev, dtype="float32")
        y = nyan.Tensor(labels, device=dev, dtype="float32")
        loss = nn.SoftmaxLoss()(x, y)
        loss.backward()
        losses.append((float(loss.numpy().sum()), x.grad.numpy()))

    np.testing.assert_allclose(losses[1][0], losses[0][0], rtol=1e-4, atol=ATOL)
    np.testing.assert_allclose(losses[1][1], losses[0][1], rtol=1e-4, atol=ATOL)


@pytest.mark.parametrize("device", _DEVICES, ids=str)
@pytest.mark.parametrize("optimizer", [nyan.optim.SGD, nyan.optim.Adam], ids=["sgd", "adam"])
def test_training_step_matches_numpy(device, optimizer):
    np.random.seed(3)
    data = np.random.randn(16, 6).astype("float32")
    labels = np.random.randint(0, 3, size=(16,)).astype("float32")
    weights = [np.random.randn(6, 3).astype("float32"), np.random.randn(1, 3).astype("float32")]

    trajectories = []
    for dev in (nyan.numpy_device(), device):
        layer = nn.Linear(6, 3, device=dev)
        layer.weight = nn.Parameter(nyan.Tensor(weights[0], device=dev, dtype="float32"))
        layer.bias = nn.Parameter(nyan.Tensor(weights[1], device=dev, dtype="float32"))
        opt = optimizer(layer.parameters(), lr=0.05)
        x = nyan.Tensor(data, device=dev, dtype="float32")
        y = nyan.Tensor(labels, device=dev, dtype="float32")

        losses = []
        for _ in range(5):
            opt.reset_grad()
            loss = nn.SoftmaxLoss()(layer(x), y)
            loss.backward()
            opt.step()
            losses.append(float(loss.numpy().sum()))
        trajectories.append(losses)

    np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=1e-3, atol=1e-3)
    assert trajectories[1][-1] < trajectories[1][0]


@pytest.mark.parametrize("device", _DEVICES, ids=str)
def test_tensor_moves_between_devices(device):
    data = np.random.randn(4, 5).astype("float32")
    x = nyan.Tensor(data, dtype="float32")

    moved = x.to(device)
    assert moved.device == device
    np.testing.assert_allclose(moved.numpy(), data)
    np.testing.assert_allclose(moved.to(nyan.numpy_device()).numpy(), data)
    assert moved.to(device) is moved
