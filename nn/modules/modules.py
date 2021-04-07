import numpy as np

from . import Module


class Linear(Module):

    def __init__(self, in_length: int, out_length: int) -> None:
        self.w = np.random.normal(loc=0.0, scale=0.01, size=(out_length, in_length + 1))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w[:, 1:].T) + self.w[:, 0]    # (B, Din) x (Din, Dout) + (Dout) => (B, Dout)

    def backward(self, delta, eta, reg_lambda):
        delta_ = np.dot(delta, self.w[:, 1:])               # (B, Dout) x (Dout, Din)
        self.w[:, 1:] *= 1 - eta * reg_lambda
        self.w[:, 1:] -= eta * np.dot(delta.T, self.x)      # (Dout, Din) - (Dout, B) x (B, Din)
        self.w[:, 0] -= eta * np.sum(delta, axis=0)         # (Dout) - ((B, Dout) => (Dout))
        return delta_


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9) -> None:
        self.mean = np.zeros((length,))
        self.var = np.zeros((length,))
        self.gamma = np.ones((length,))
        self.beta = np.zeros((length,))
        self.momentum = momentum

    def forward(self, x):
        self.mean_ = np.mean(x, axis = 0)
        self.var_ = np.var(x, axis = 0)
        self.mean = self.momentum * self.mean + (1 - self.momentum) * self.mean_
        self.var = self.momentum * self.var + (1 - self.momentum) * self.var_
        self.x = (x - self.mean_) / np.sqrt(self.var_ + 1e-5)
        return self.gamma * self.x + self.beta

    def backward(self, delta, eta, **kwargs):
        N = self.x.shape[0]
        delta_ = delta * self.gamma
        delta_ = N * delta_ - np.sum(delta_, axis=0) - self.x * np.sum(delta_ * self.x, axis=0)

        self.gamma -= eta * np.sum(delta * self.x, axis=0)
        self.beta -= eta * np.sum(delta, axis=0)
        return delta_ / N / np.sqrt(self.var + 1e-5)
