import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except ImportError:
    pass


def add(x, y):
    return x + y


def parse_mnist(image_filename, label_filename):

    with gzip.open(image_filename, 'rb') as f:
        header = f.read(16)
        magic, nums, rows, cols = struct.unpack('>IIII', header)

        data = f.read(nums * rows * cols)
        X_uint8 = np.frombuffer(data, dtype=np.uint8).reshape(nums, rows * cols)
        X = X_uint8.astype(np.float32)
        X /= 255
       
    with gzip.open(label_filename, 'rb') as f:
        header = f.read(8)
        magic, nums = struct.unpack('>II', header)

        data = f.read(nums)
        y = np.frombuffer(data, dtype=np.uint8)

    return X,y


def softmax_loss(Z, y):
    return np.mean(-Z[np.arange(Z.shape[0]), y] + np.log(np.sum(np.exp(Z), axis=1)))


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    # theta := theta - a(Z-I_y)X^T
    # use slice operator [:] ?
    # X is (m x n)
    # Z is (m x k)
    # I_y is (m x k)
    # transpose X -> n x m x m x k = (n x k)
    # h(x) = x * theta where x is (m x n), theta is (n x k) = (m x k)

    m = X.shape[0]
    k = theta.shape[1]

    for i in range(0, m, batch):
        X_batch = X[i : i + batch] # (B,n)
        y_batch = y[i : i + batch] # (B,)

        h = X_batch @ theta # (B,n) * (n,k) = (B,k)
        exp_h = np.exp(h)
        sums = np.sum(exp_h, axis = 1, keepdims = True)
        Z = exp_h / sums # (B,k)

        I_y = np.zeros((batch, k)) # (B,k)
        I_y[np.arange(batch), y_batch] = 1

        step = (lr / batch) * X_batch.T @ (Z - I_y) # (n,B) * (B,k) = (n,k)

        theta[:] -= step



def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):

    m = X.shape[0]
    k = W2.shape[1]

    for i in range(0, m, batch):
        X_batch = X[i : i + batch] # (B,n)
        y_batch = y[i : i + batch] # (B,)

        # forward pass

        intermediate = X_batch @ W1 # (B,n) * (n,d) = (B,d)
        Z2 = np.maximum(0, intermediate) # (B,d)
        h = Z2 @ W2 # logits: (B,d) * (d,k) = (B,k)

        # backprop for W2
        exp_h = np.exp(h)
        sums = np.sum(exp_h, axis=1, keepdims=True)
        S = exp_h / sums # (B,k)

        I_y = np.zeros((batch, k))
        I_y[np.arange(batch), y_batch] = 1

        W2_grad = (Z2.T @ (S - I_y)) / batch

        # backprop for W1
        relugrad = np.sign(Z2)

        W1_grad = (X_batch.T @ (((S - I_y) @ W2.T) * relugrad)) / batch

        W2 -= lr * W2_grad
        W1 -= lr * W1_grad
        

def loss_err(h,y):
    """Return loss and classification error."""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """Train softmax regression and print one row per epoch."""
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """Train the two-layer network and print one row per epoch."""
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
