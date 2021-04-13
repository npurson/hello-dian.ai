import numpy as np

from .modules import Module


class Sigmoid(Module):
    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, delta, eta, **kwargs):
        return self.y * (1 - self.y)


class Tanh(Module):
    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, delta, eta, **kwargs):
        return delta * (1 - self.x ** 2)


class ReLU(Module):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta, eta, **kwargs):
        return np.where(self.x > 0, delta, 0)


class Softmax(Module):

    def forward(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)


class CrossEntropyLoss(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        return -np.sum(np.eye(self.n_classes)[targets] * np.log(probs))  # + (1 - np.eye(self.n_classes)[targets]) * np.log(1 - probs))
