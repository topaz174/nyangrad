"""Optimization module"""
import nyangrad as nyan
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            grad_decay = param.grad.data + self.weight_decay * param.data

            if param not in self.u:
                self.u[param] = (1 - self.momentum) * grad_decay
            else:
                self.u[param].data = self.momentum * self.u[param].data + (1 - self.momentum) * grad_decay

            param.data -= self.lr * self.u[param].data

    def clip_grad_norm(self, max_norm=0.25):
        """Clips gradient norm of parameters."""
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for param in self.params:
            grad_decay = param.grad.data + self.weight_decay * param.data

            if param not in self.m:
                self.m[param] = (1 - self.beta1) * grad_decay.data
            else:
                self.m[param].data = self.beta1 * self.m[param].data + (1 - self.beta1) * grad_decay.data

            if param not in self.v:
                self.v[param] = (1 - self.beta2) * (grad_decay.data ** 2)
            else:
                self.v[param].data = self.beta2 * self.v[param].data + (1 - self.beta2) * (grad_decay.data ** 2)

            m_bias = self.m[param].data / (1 - self.beta1 ** self.t)
            v_bias = self.v[param].data / (1 - self.beta2 ** self.t)

            param.data = param.data - self.lr * m_bias / (v_bias ** 0.5 + self.eps)

