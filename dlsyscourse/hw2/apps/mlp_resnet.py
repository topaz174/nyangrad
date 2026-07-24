import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    lin_1 = nn.Linear(dim, hidden_dim)
    norm_1 = norm(hidden_dim)
    relu_1 = nn.ReLU()
    dropout = nn.Dropout(drop_prob)
    lin_2 = nn.Linear(hidden_dim, dim)
    norm_2 = norm(dim)

    seq = nn.Sequential(lin_1, norm_1, relu_1, dropout, lin_2, norm_2)

    res = nn.Residual(seq)

    return nn.Sequential(res, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    lin_1 = nn.Linear(dim, hidden_dim)
    relu_1 = nn.ReLU()

    seq_list = [lin_1, relu_1]
    
    for _ in range(num_blocks):
        seq_list.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm=norm, drop_prob=drop_prob))
    
    lin_2 = nn.Linear(hidden_dim, num_classes)
    seq_list.append(lin_2)

    return nn.Sequential(*seq_list)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION

    smloss = nn.SoftmaxLoss()
    losses = []
    wrong = 0
    m = len(dataloader.dataset)

    if opt: 
        model.train()
    else:
        model.eval()

    for X, y in dataloader:
        logits = model(X)

        wrong += (np.argmax(logits.numpy(), axis=1) != y.numpy()).sum()

        loss = smloss(logits, y)
        losses.append(loss.numpy())

        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()

    err_rate = wrong / m
    avg_loss = np.mean(losses)
    
    return err_rate, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dset = nn.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_dset = nn.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    resnet = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10)

    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)

    train_dloader = nn.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    test_dloader = nn.DataLoader(test_dset, batch_size=batch_size)

    for _ in range(epochs):
        train_err, train_loss = epoch(train_dloader, resnet, opt)

    test_err, test_loss = epoch(test_dloader, resnet, None)

    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
