import numpy as np
import itertools

from . import tensor
from .tensor import Tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return delta

    def train(self) -> None:
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self) -> None:
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """
        self.w = Tensor((in_length + 1, out_length))

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """
        self.x = x
        return np.dot(x, self.w[1:]) + self.w[0]

    def backward(self, delta):
        """Backward propagation of linear module.

        Args:
            delta: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """
        self.w.grad = np.vstack((np.sum(delta, axis=0), np.dot(self.x.T, delta)))
        return np.dot(delta, self.w[1:].T)


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()
        self.mean = np.zeros((length,))
        self.var = np.zeros((length,))
        self.gamma = tensor.ones((length,))
        self.beta = tensor.zeros((length,))
        self.momentum = momentum
        self.eps = 1e-5

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """
        if self.training:
            self.mean = np.mean(x, axis = 0)
            self.var = np.var(x, axis = 0)
            self.running_mean = self.momentum * self.mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.var + (1 - self.momentum) * self.var
            self.x = (x - self.mean) / np.sqrt(self.var + self.eps)
        else:
            self.x = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * self.x + self.beta

    def backward(self, delta):
        """Backward propagation of batch norm module.

        Args:
            delta: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """
        self.gamma.grad = np.sum(delta * self.x, axis=0)
        self.beta.grad = np.sum(delta, axis=0)
        N = delta.shape[0]
        delta *= self.gamma
        delta = N * delta - np.sum(delta, axis=0) - self.x * np.sum(delta * self.x, axis=0)
        return delta / N / np.sqrt(self.var + self.eps)


class Conv2d(Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=False):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            out_channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """
        self.channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = tensor.rand((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = tensor.zeros(out_channels) if bias else None

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """
        B, C, H, W = x.shape
        Hp, Wp = map(lambda i : (i - self.kernel_size + 2 * self.padding) // self.stride + 1, (H, W))
        out = np.ndarray((B, self.channels, Hp, Wp))
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.x = x

        for b, c, h, w in itertools.product(*tuple([range(d) for d in out.shape])):
            out[b, c, h, w] = np.sum(self.kernel[c] * x[b, :, h * self.stride : h * self.stride + self.kernel_size,
                                                              w * self.stride : w * self.stride + self.kernel_size])

        # # Method 2
        # shape = (B, C, Hp, Wp, self.kernel_size, self.kernel_size)
        # strides = (*x.strides[:-2], x.strides[-2] * self.stride, x.strides[-1] * self.stride, *x.strides[-2:])
        # xp = np.lib.stride_tricks.as_strided(out, shape=shape, strides=strides, writeable=False)
        # x = np.tensordot(xp, self.kernel, axes=((1, -2, -1), (1, 2, 3))).transpose((0, 3, 1, 2))
        return (out + self.bias) if self.bias else out

    def backward(self, delta):
        """Backward propagation of convolution module.

        Args:
            delta: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """
        dx = np.zeros_like(self.x)
        self.kernel.grad = np.zeros_like(self.kernel)

        for b, c, h, w in itertools.product(*tuple([range(d) for d in delta.shape])):
            dx[b, :, h * self.stride : h * self.stride + self.kernel_size,
                     w * self.stride : w * self.stride + self.kernel_size] += self.kernel[c] * delta[b, c, h, w]

            self.kernel.grad[c] += delta[b, c, h, w] * self.x[b, :, h * self.stride : h * self.stride + self.kernel_size,
                                                                    w * self.stride : w * self.stride + self.kernel_size]
        if self.bias:
            self.bias.grad = np.sum(delta, axis=(0, 2, 3))
        if self.padding:
            dx = dx[..., self.padding:-self.padding, self.padding:-self.padding]
        return dx


class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """
        B, C, H, W = x.shape
        Hp, Wp = map(lambda i : (i - self.kernel_size + 2 * self.padding) // self.stride + 1, (H, W))
        out = np.ndarray((B, C, Hp, Wp))
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.x = x

        for h, w in itertools.product(range(Hp), range(Wp)):
            out[..., h, w] = np.mean(x[..., h * self.stride : h * self.stride + self.kernel_size,
                                            w * self.stride : w * self.stride + self.kernel_size], axis=(2, 3))
        return out

    def backward(self, delta):
        """Backward propagation of average pooling module.

        Args:
            delta: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """
        dx = np.zeros_like(self.x)
        B, C, H, W = delta.shape
        for h, w in itertools.product(range(H), range(W)):
            dx[..., h * self.stride : h * self.stride + self.kernel_size,
                    w * self.stride : w * self.stride + self.kernel_size] \
                += (np.expand_dims(delta[..., h, w], 2).repeat(self.kernel_size ** 2, 2) / \
                   (self.kernel_size ** 2)).reshape(B, C, self.kernel_size, self.kernel_size)
        if self.padding:
            dx = dx[..., self.padding:-self.padding, self.padding:-self.padding]
        return dx


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """
        B, C, H, W = x.shape
        Hp, Wp = map(lambda i : (i - self.kernel_size + 2 * self.padding) // self.stride + 1, (H, W))
        out = np.zeros((B, C, Hp, Wp))
        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        self.x = x

        for h, w in itertools.product(range(Hp), range(Wp)):
            out[..., h, w] = np.max(x[..., h * self.stride : h * self.stride + self.kernel_size,
                                           w * self.stride : w * self.stride + self.kernel_size], axis=(2, 3))
        return out

        # # Method 2
        # shape = (B, C, Hp, Wp, self.kernel_size, self.kernel_size)
        # strides = (*x.strides[:-2], x.strides[-2] * self.stride, x.strides[-1] * self.stride, *x.strides[-2:])
        # out = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)
        # return np.max(out, axis=(-2,-1))

    def backward(self, delta):
        """Backward propagation of max pooling module.

        Args:
            delta: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """
        dx = np.zeros_like(self.x)
        B, C, H, W = delta.shape

        for h, w in itertools.product(range(H), range(W)):
            max = np.max(self.x[..., h * self.stride : h * self.stride + self.kernel_size,
                                     w * self.stride : w * self.stride + self.kernel_size], axis=(2, 3))
            mask = self.x[..., h * self.stride:h * self.stride + self.kernel_size,
                               w * self.stride:w * self.stride + self.kernel_size] == max
            dx[:, :, h * self.stride : h * self.stride + self.kernel_size,
                     w * self.stride : w * self.stride + self.kernel_size] \
                += mask * np.expand_dims(delta[..., h, w], 2).repeat(self.kernel_size ** 2, axis=2) \
                                                             .reshape(B, C, self.kernel_size, self.kernel_size)

        if self.padding:
            dx = dx[..., self.padding:-self.padding, self.padding:-self.padding]
        return dx


class Dropout(Module):

    def __init__(self, p: float=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            self.mask = np.array(np.random.binomial(1, 1 - self.p, x.shape))
            return x * self.mask / self.p
        else:
            return x

    def backard(self, delta):
        return delta * self.mask


if __name__ == '__main__':
    import pdb; pdb.set_trace()
