import numpy as np
from .modules import Module


class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.

        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.

        ...

        # End of todo


class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.

        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.

        ...

        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.

        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.

        ...

        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of Softmax function.

        ...

        # End of todo

    def backward(self, dy):

        # Omitted.
        ...


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
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
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.

        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.

        ...

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.

        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.

        ...

        # End of todo
