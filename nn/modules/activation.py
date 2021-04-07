import numpy as np

from .module import Module


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


class LeakyReLU(Module):
    def forward(self, x):
        self.x = x
        ...

    def backward(self, delta, eta, **kwargs):
        ...


class ELU(Module):
    def forward(self, x):
        self.x = x
        ...

    def backward(self, delta, eta, **kwargs):
        ...
