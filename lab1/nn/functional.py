# %load functional.py
import numpy as np
from .modules import Module


class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        self.forw = 1 / (1 + np.exp(-x))
        return self.forw
        
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.
        dy = dy * self.forw * (np.ones(self.forw.shape) - self.forw)
        return dy
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
        self.mask = (x <= 0)   # 得到关于y大于小于0的真值的矩阵
        z = y.copy()       # 深度拷贝一个y矩阵
        z[self.mask] = 0   # 将小于零的值赋为0
        return z   # 返回矩阵
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.
        dy[self.mask] = 0
        dz = dy
        return dz
        ...

        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        
        y = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)   # 返回softmax处理后矩阵，利于进一步计算损失函数
        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.

        ...

        # End of todo


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
        self.loss = None
        ...
        return self

    def backward(self):
        ...


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate softmax loss.
        z = Softmax(probs)    # 使用激活函数将输出矩阵归一化
        batch_size = z.shape[0]   # 得到batch_size
        loss = -np.sum(np.log(z[np.arange(batch_size), label])) / batch_size    # 求出平均损失误差值，使用交叉熵，利用one-hot特性得到每组输入的log值
#                 loss = -np.sum(np.log(z[np.arange(batch_size), t] + 1e-7)) / batch_size
        self.loss = loss
        self.probs = z
        self.targets = targets
        
        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.
        batch_size = self.probs.shape[0]  # 得到batch_size
        dz = np.copy(self.probs)       # 深拷贝
        for label_, z_ in zip(self.probs, dz):   # 由求梯度+onehot编码推出仅需在真实值所在位置减1即得梯度
            z_[label_] -= 1
        dz /= batch_size   # 取平均
        return dz   # 返回梯度
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













# import numpy as np
# from .modules import Module


# class Sigmoid(Module):

#     def forward(self, x):

#         # TODO Implement forward propogation
#         # of sigmoid function.

#         ...

#         # End of todo

#     def backward(self, dy):

#         # TODO Implement backward propogation
#         # of sigmoid function.

#         ...

#         # End of todo


# class Tanh(Module):

#     def forward(self, x):

#         # TODO Implement forward propogation
#         # of tanh function.

#         ...

#         # End of todo

#     def backward(self, dy):

#         # TODO Implement backward propogation
#         # of tanh function.

#         ...

#         # End of todo


# class ReLU(Module):

#     def forward(self, x):

#         # TODO Implement forward propogation
#         # of ReLU function.

#         ...

#         # End of todo

#     def backward(self, dy):

#         # TODO Implement backward propogation
#         # of ReLU function.

#         ...

#         # End of todo


# class Softmax(Module):

#     def forward(self, x):

#         # TODO Implement forward propogation
#         # of ReLU function.

#         ...

#         # End of todo

#     def backward(self, dy):

#         # TODO Implement backward propogation
#         # of ReLU function.

#         ...

#         # End of todo


# class Loss(object):
#     """
#     Usage:
#         >>> criterion = Loss(n_classes)
#         >>> ...
#         >>> for epoch in n_epochs:
#         ...     ...
#         ...     probs = model(x)
#         ...     loss = criterion(probs, target)
#         ...     model.backward(loss.backward())
#         ...     ...
#     """
#     def __init__(self, n_classes):
#         self.n_classes = n_classes

#     def __call__(self, probs, targets):
#         self.probs = probs
#         self.targets = targets
#         ...
#         return self

#     def backward(self):
#         ...


# class SoftmaxLoss(Loss):

#     def __call__(self, probs, targets):

#         # TODO Calculate softmax loss.

#         ...

#         # End of todo

#     def backward(self):

#         # TODO Implement backward propogation
#         # of softmax loss function.

#         ...

#         # End of todo


# class CrossEntropyLoss(Loss):

#     def __call__(self, probs, targets):

#         # TODO Calculate cross-entropy loss.

#         ...

#         # End of todo

#     def backward(self):

#         # TODO Implement backward propogation
#         # of cross-entropy loss function.

#         ...

#         # End of todo




