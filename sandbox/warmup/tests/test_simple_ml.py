from pathlib import Path
import sys

import numdifftools as nd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import simple_ml


def test_add_handles_scalars_and_arrays():
    assert simple_ml.add(5, 6) == 11
    np.testing.assert_array_equal(
        simple_ml.add(np.array([1, 2]), np.array([3, 4])),
        np.array([4, 6]),
    )


def test_softmax_loss_at_uniform_logits():
    logits = np.zeros((8, 4), dtype=np.float32)
    labels = np.arange(8) % 4
    np.testing.assert_allclose(simple_ml.softmax_loss(logits, labels), np.log(4))


def test_softmax_regression_epoch_matches_numerical_gradient():
    np.random.seed(0)
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.randint(3, size=20).astype(np.uint8)
    theta = np.zeros((5, 3), dtype=np.float32)
    expected = -nd.Gradient(
        lambda value: simple_ml.softmax_loss(X @ value.reshape(5, 3), y)
    )(theta)

    simple_ml.softmax_regression_epoch(X, y, theta, lr=1.0, batch=len(X))

    np.testing.assert_allclose(theta, expected.reshape(5, 3), rtol=1e-4, atol=1e-4)


def test_two_layer_epoch_reduces_loss():
    np.random.seed(1)
    X = np.random.randn(40, 5).astype(np.float32)
    y = np.random.randint(3, size=40).astype(np.uint8)
    W1 = np.random.randn(5, 8).astype(np.float32) / np.sqrt(8)
    W2 = np.random.randn(8, 3).astype(np.float32) / np.sqrt(3)
    before = simple_ml.softmax_loss(np.maximum(X @ W1, 0) @ W2, y)

    simple_ml.nn_epoch(X, y, W1, W2, lr=0.1, batch=len(X))

    after = simple_ml.softmax_loss(np.maximum(X @ W1, 0) @ W2, y)
    assert after < before


@pytest.mark.skipif(
    not hasattr(simple_ml, "softmax_regression_epoch_cpp"),
    reason="warm-up C++ extension is not built",
)
def test_cpp_epoch_matches_python_epoch():
    np.random.seed(2)
    X = np.random.randn(20, 5).astype(np.float32)
    y = np.random.randint(3, size=20).astype(np.uint8)
    python_theta = np.zeros((5, 3), dtype=np.float32)
    cpp_theta = python_theta.copy()

    simple_ml.softmax_regression_epoch(
        X, y, python_theta, lr=0.2, batch=len(X)
    )
    simple_ml.softmax_regression_epoch_cpp(
        X, y, cpp_theta, lr=0.2, batch=len(X)
    )

    np.testing.assert_allclose(cpp_theta, python_theta, rtol=1e-5, atol=1e-5)
