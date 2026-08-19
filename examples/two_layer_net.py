"""A two-layer MNIST classifier written directly with Tensor operations."""

import gzip
import struct

import numpy as np

import nyangrad as nyan


def parse_mnist(image_filename, label_filename):
    """Load flattened, normalized MNIST images and their labels."""
    with gzip.open(image_filename, "rb") as images_file:
        magic, count, rows, cols = struct.unpack(">IIII", images_file.read(16))
        if magic != 2051:
            raise ValueError(f"invalid MNIST image magic number: {magic}")
        images = np.frombuffer(
            images_file.read(count * rows * cols), dtype=np.uint8
        )
        images = images.reshape(count, rows * cols).astype(np.float32) / 255.0

    with gzip.open(label_filename, "rb") as labels_file:
        magic, label_count = struct.unpack(">II", labels_file.read(8))
        if magic != 2049:
            raise ValueError(f"invalid MNIST label magic number: {magic}")
        labels = np.frombuffer(labels_file.read(label_count), dtype=np.uint8)

    if len(images) != len(labels):
        raise ValueError("MNIST image and label counts do not match")
    return images, labels


def softmax_loss(logits, y_one_hot):
    """Average cross-entropy loss for one-hot targets."""
    log_probs = nyan.logsoftmax(logits)
    return -(log_probs * y_one_hot).sum() / logits.shape[0]


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run one ordered SGD epoch for a bias-free two-layer ReLU network."""
    num_classes = W2.shape[1]

    for start in range(0, X.shape[0], batch):
        X_batch = nyan.Tensor(X[start : start + batch])
        labels = y[start : start + batch]
        y_one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
        y_one_hot[np.arange(len(labels)), labels] = 1

        logits = nyan.relu(X_batch @ W1) @ W2
        loss = softmax_loss(logits, nyan.Tensor(y_one_hot))
        loss.backward()

        W1.data = W1.data - lr * W1.grad.data
        W2.data = W2.data - lr * W2.grad.data

    return W1, W2


def loss_err(logits, labels):
    """Return cross-entropy loss and classification error."""
    y_one_hot = np.zeros((len(labels), logits.shape[-1]), dtype=np.float32)
    y_one_hot[np.arange(len(labels)), labels] = 1
    loss = softmax_loss(logits, nyan.Tensor(y_one_hot)).numpy()
    error = np.mean(logits.numpy().argmax(axis=1) != labels)
    return loss, error
