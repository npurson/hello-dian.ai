from .tensor import Tensor
from .modules import Module


class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)

    def _step_module(self, module):
        for attr in vars(module).values():
            if isinstance(attr, Tensor):
                if hasattr(attr, 'grad'):
                    self._update_weight(attr)
            if isinstance(attr, Module):
                self._step_module(attr)
            if isinstance(attr, list):
                for item in attr:
                    self._step_module(item)

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum

    def _update_weight(self, tensor):
        tensor.v = self.momentum * tensor.v + self.lr * tensor.grad \
                   if 'v' in vars(tensor) else self.lr * tensor.grad
        tensor -= tensor.v


class Adam(Optim):

    def __init__(self, module, lr):
        super(Adam, self).__init__(module, lr)
        ...

    def _update_weight(self, tensor):
        ...