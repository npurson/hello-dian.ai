import numpy as np

import tensor
from tensor import Tensor


class Module(object):

    def __init__(self):
        self.training = True

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        ...

    def backward(self, delta):
        return delta

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class Linear(Module):

    def __init__(self, in_length: int, out_length: int) -> None:
        self.w = Tensor((out_length, in_length + 1))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w[:, 1:].T) + self.w[:, 0]

    def backward(self, delta):
        self.w.grad = np.vstack((np.sum(delta, axis=0), np.dot(delta.T, self.x)))
        return np.dot(delta, self.w[:, 1:])


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9) -> None:
        super(Linear, self).__init__()
        self.mean = np.zeros((length,))
        self.var = np.zeros((length,))
        self.gamma = tensor.ones((length,))
        self.beta = tensor.zeros((length,))
        self.momentum = momentum
        self.eps = 1e-5

    def forward(self, x):
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
        self.gamma.grad = np.sum(delta * self.x, axis=0)
        self.beta.grad = np.sum(delta, axis=0)
        N = delta.shape[0]
        delta *= self.gamma
        delta = N * delta - np.sum(delta, axis=0) - self.x * np.sum(delta * self.x, axis=0)
        return delta / N / np.sqrt(self.var + self.eps)


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int=3, stride: int=1, padding: int=0,
                 bias: bool = False) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel = np.random.normal(loc=0.0, scale=0.01, size=(
            out_channels, in_channels, kernel_size, kernel_size))
        self.bias = np.zeros(out_channels) if bias else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        assert len(
            x.shape) == 4, "Input must be a four dimensional(B,C,H,W) ndarray"
        # (Batch_size,Channel,Height,Width)
        B, C, H, W = x.shape

        assert (H - self.kernel_size + 2 *
                self.padding) % self.stride == 0, "Error occurs in parameter setting"
        assert (W - self.kernel_size + 2 *
                self.padding) % self.stride == 0, "Error occurs in parameter setting"

        H_new = int((H - self.kernel_size + 2 * self.padding) / self.stride + 1)
        W_new = int((W - self.kernel_size + 2 * self.padding) / self.stride + 1)
        x_new = np.zeros((B, self.out_channels, H_new, W_new))

        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                           (self.padding, self.padding)), mode="constant")

        self.x = x



        # for batch in range(B):
        #     for out_channel in range(self.out_channels):
        #         for h in range(H_new):
        #             for w in range(W_new):
        #                 multi = self.kernel[out_channel] * self.x[batch,:, h * self.stride:h * self.stride + self.kernel_size, w * self.stride:w * self.stride + self.kernel_size]
        #                 x_new[batch, out_channel, h, w] = np.sum(multi)
        #                 if self.bias:
        #                     x_new[batch, out_channel, h, w] = x_new[batch,out_channel, h, w] + self.bias


        # return x_new

        # (B,Ci,H,W) -> (B,Ci,H,W,K,K)
        shape = (B,C,H_new,W_new,self.kernel_size,self.kernel_size)
        strides = (*x.strides[:-2],x.strides[-2]*self.stride,x.strides[-1]*self.stride,*x.strides[-2:])
        px = np.lib.stride_tricks.as_strided(x_new,shape = shape,strides=strides,writeable=False)

        # (B,Ci,H,W,K,K) * (Co,Ci,K,K) -> (B,Co,H,W)
        x = np.tensordot(px,self.kernel,axes=((1,-2,-1),(1,2,3))).transpose((0,3,1,2))
        print(x.shape)
        if self.bias:
            x += self.bias
        return x

    @Timer
    def backward(self, delta: np.ndarray):
        grad_kernel = np.zeros_like(self.kernel)
        grad_x = np.zeros_like(self.x)
        if self.bias:
            grad_bias = np.zeros_like(self.bias)

        B, C, H, W = delta.shape
        for b in range(B):
            for out_channel in range(self.out_channels):
                for h in range(H):
                    for w in range(W):
                        # (C,K,K) <-> () * (C,K,K)
                        grad_kernel[out_channel] += delta[b, out_channel, h, w] * self.x[b, :,
                                                                                  h * self.stride:h * self.stride + self.kernel_size,
                                                                                  w * self.stride:w * self.stride + self.kernel_size]
                        # (1,C,K,K) <-> () * (C,K,K)
                        grad_x[b, :, h * self.stride: h * self.stride + self.kernel_size,
                        w * self.stride: w * self.stride + self.kernel_size] += delta[b, out_channel, h, w] * \
                                                                                self.kernel[out_channel]
                        if self.bias:
                            grad_bias[out_channel] += np.sum(delta[b, out_channel])

        if self.padding:
            grad_x = grad_x[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_x, grad_kernel, grad_bias


class AvgPool(Module):
    def __init__(self, kernel_size: int = 2,
                 stride: int = 2, padding: int = 0):

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert len(x.shape) == 4, "Input must be a four dimensional(B,C,H,W) ndarray"
        # (Batch_size,Channel,Height,Width)
        B, C, H, W = x.shape
        assert (H - self.kernel_size + 2 *
                self.padding) % self.stride == 0, "Error occurs in parameter setting"
        assert (W - self.kernel_size + 2 *
                self.padding) % self.stride == 0, "Error occurs in parameter setting"

        H_new = int( (H - self.kernel_size + 2 * self.padding) / self.stride + 1)
        W_new = int( (W - self.kernel_size + 2 * self.padding) / self.stride + 1)
        x_new = np.zeros((B, C, H_new, W_new))

        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                           (self.padding, self.padding)), mode="constant")
        self.x = x
        for h in range(H_new):
            for w in range(W_new):
                x_new[:, :, h, w] = np.mean(x[:, :, h * self.stride:h * self.stride + self.kernel_size,
                                           w * self.stride:w * self.stride + self.kernel_size], axis=(2, 3))
        #################################################################################
        return x_new

    @Timer
    def backward(self, delta: np.ndarray):
        """
               Avgpool层的反向传播，使用范例如下
               >>> Pool1 = Avgpool_naive(kernel_size)                # 初始化
               >>> x = np.random.rand(B,in_channels,H,W)                         # 随机生成一个变量
               >>> retval = Pool1(x)                                             # 前向传播
               >>> delta = np.random.rand(*retval.shape)                         # 随机生成一个上游梯度
               >>> grad = Pool1.backward(delta)                                 # 反向传播

                   :param delta: 上游传回的梯度矩阵，和前向传播的输出形状应该相同
                   :return: 返回x的梯度值
               """
        delta_ = np.zeros_like(self.x)
        B, C, H, W = delta.shape
        # TODO:实现Avgpool的反向传播
        for h in range(H):
            for w in range(W):
                delta_[:, :, h * self.stride:h * self.stride + self.kernel_size,
                w * self.stride:w * self.stride + self.kernel_size] += (
                            np.expand_dims(delta[:, :, h, w], 2).repeat(self.kernel_size ** 2, 2) / (
                                self.kernel_size ** 2)).reshape(B, C, self.kernel_size, self.kernel_size)
        if self.padding:
            delta_ = delta_[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return delta_


class MaxPool(Module):
    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        assert len(x.shape) == 4, "Input must be a four dimensional(B,C,H,W) ndarray"
        # (Batch_size,Channel,Height,Width)
        B, C, H, W = x.shape
        assert (H - self.kernel_size + 2 *
                self.padding) % self.stride == 0, "Error occurs in parameter setting"
        assert (W - self.kernel_size + 2 *
                self.padding) % self.stride == 0, "Error occurs in parameter setting"

        H_new = int( (H - self.kernel_size + 2 * self.padding) / self.stride + 1)
        W_new = int( (W - self.kernel_size + 2 * self.padding) / self.stride + 1)


        if self.padding:
            x = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                           (self.padding, self.padding)), mode="constant")
        self.x = x
        shape = (B, C, H_new, W_new, self.kernel_size, self.kernel_size)
        strides = (*x.strides[:-2], x.strides[-2] * self.stride, x.strides[-1] * self.stride, *x.strides[-2:])
        x_new = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides, writeable=False)

        return np.max(x_new,axis=(-2,-1))

    def backward(self, delta: np.ndarray):
        delta_ = np.zeros_like(self.x)
        B, C, H, W = delta.shape

        for h in range(H):
            for w in range(W):
                # (B,C)
                max = np.max(self.x[:, :, h * self.stride:h * self.stride + self.kernel_size,
                             w * self.stride:w * self.stride + self.kernel_size], axis=(1, 2))
                # (B,C,kernel_size,kernel_size)
                mask = self.x[:, :, h * self.stride:h * self.stride + self.kernel_size,
                       w * self.stride:w * self.stride + self.kernel_size] == max
                # (B,C,kernel_size,kernel_size)
                delta_[:, :, h * self.stride:h * self.stride + self.kernel_size, w * self.stride:w * self.stride + self.kernel_size] += mask * np.expand_dims(delta[:, :, h, w],2).repeat(self.kernel_size ** 2, axis=2).reshape(B, C, self.kernel_size, self.kernel_size)

        if self.padding:
            delta_ = delta_[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return delta_


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
