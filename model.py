import numpy as np

import nn
import nn.functional as F


class DNN(nn.Module):

    def __init__(self, lengths: list, actv: str='ReLU') -> None:
        Activation = getattr(nn.activation, actv)
        self.layers = []
        for i in range(len(lengths) - 1):
            self.layers.append(nn.Linear(lengths[i], lengths[i + 1]))
            self.layers.append(nn.BatchNorm1d(lengths[i + 1]))
            self.layers.append(Activation() if i != len(lengths) - 2 else F.Softmax())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, delta, eta, reg_lambda=0):
        for layer in reversed(self.layers):
            delta = layer.backward(delta, eta, reg_lambda=reg_lambda)

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)


if __name__ == '__main__':
    nn = DNN((2, 4, 4, 2))
    X = np.zeros((12, 2))
    y = np.zeros((12, 2))
    pred = nn.forward(X)
    nn.backward(y - pred, 0.1)
