import numpy as np

from nn.modules.module import Module
import torch.nn as nn
import torch
import torch.nn.functional as F

class Linear(Module):

    def __init__(self, in_length: int, out_length: int, w: None) -> None:
        if w is None:
            self.w = np.random.normal(loc=0.0, scale=0.01, size=(out_length, in_length + 1))
        else:
            self.w = w

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w[:, 1:].T) + self.w[:, 0]    # (B, Din) x (Din, Dout) + (Dout) => (B, Dout)

    def backward(self, delta, eta=0, reg_lambda=0):
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
        
        self.cache = (self.x, self.gamma, self.x - self.mean_, self.var_ + 1e-5)
        return self.gamma * self.x + self.beta

    def backward(self, delta, eta=0, **kwargs):
        N = self.x.shape[0]
        x_, gamma, x_minus_mean, var_plus_eps = self.cache
        dgamma = np.sum(x_ * delta, axis=0)
        dbeta = np.sum(delta, axis=0)
        dx_ = np.matmul(np.ones((N,1)), gamma.reshape((1, -1))) * delta
        dx = N * dx_ - np.sum(dx_, axis=0) - x_*np.sum(dx_ * x_, axis=0)
        dx *= (1.0/N) / np.sqrt(var_plus_eps)


        # delta_ = delta * self.gamma
        # delta_ = N * delta_ - np.sum(delta_, axis=0) - self.x * np.sum(delta_ * self.x, axis=0)

        self.gamma -= eta * np.sum(delta * self.x, axis=0)
        self.beta -= eta * np.sum(delta, axis=0)
        return dx


if __name__ == '__main__':

    def test_maxPool():
        input = np.random.rand(2, 2, 4, 4)
        print(input)
        torch_input = torch.Tensor(input)
        torch_input.requires_grad = True
        kernal_size = 2
        maxPool_torch = torch.nn.MaxPool2d(kernel_size=kernal_size)
        maxPool_torch_output = maxPool_torch(torch_input)
        maxPool_torch_output_sum = maxPool_torch_output.sum()
        maxPool_torch_output_sum.backward()
        print(maxPool_torch(torch_input))
        print(torch_input.grad)

    # test_maxPool()







