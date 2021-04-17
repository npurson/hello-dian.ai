import numpy as np

from .modules import Module


class Sigmoid(Module):

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, delta):
        return self.y * (1 - self.y)


class Tanh(Module):

    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, delta):
        return delta * (1 - self.x ** 2)


class ReLU(Module):

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        return np.where(self.x > 0, delta, 0)


class Softmax(Module):

    def forward(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, delta):
        ...


class Loss(object):
    """
    >>> criterion = CrossEntropyLoss(n_classes)
    >>> ...
    >>> for epoch in n_epochs:
    ...     ...
    ...     probs = model(x)
    ...     loss = criterion(probs, target)
    ...     model.backward(loss.backward())
    ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes
    
    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        ...
    
    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):
        super(SoftmaxLoss, self).__call__(probs, targets)
        ...
        return self

    def backward(self, delta):
        return self.probs - self.targets


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):
        super(SoftmaxLoss, self).__call__(probs, targets)
        ...
        return self

    def backward(self):
        ...
