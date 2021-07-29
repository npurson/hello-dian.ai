import numpy as np
from itertools import product
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True
        self.forw = None

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

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


# class Linear(Module):

#     def __init__(self, in_length: int, out_length: int):
#         """Module which applies linear transformation to input.

#         Args:
#             in_length: L_in from expected input shape (N, L_in).
#             out_length: L_out from output shape (N, L_out).
#         """

#         # TODO Initialize the weight
#         # of linear module.
        
#         ...

#         # End of todo

#     def forward(self, x):
#         """Forward propagation of linear module.

#         Args:
#             x: input of shape (N, L_in).
#         Returns:
#             out: output of shape (N, L_out).
#         """

#         # TODO Implement forward propogation
#         # of linear module.

#         ...

#         # End of todo


#     def backward(self, dy):
#         """Backward propagation of linear module.

#         Args:
#             dy: output delta of shape (N, L_out).
#         Returns:
#             dx: input delta of shape (N, L_in).
#         """

#         # TODO Implement backward propogation
#         # of linear module.

#         ...

#         # End of todo

class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # TODO Initialize the weight
        # of linear module.
        print('hello world')
        self.x = None
        self.w = tensor.random((in_length, out_length))   # 初始化权重，注意权重还有一个参数为grad
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.
        self.x = x
        return np.dot(x, self.w)
        ...

        # End of todo


    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        self.w.grad = np.dot(self.x.T, dy)
        return self.w.grad
        ...

        # End of todo






# class BatchNorm1d(Module):

#     def __init__(self, length: int, momentum: float=0.9):
#         """Module which applies batch normalization to input.

#         Args:
#             length: L from expected input shape (N, L).
#             momentum: default 0.9.
#         """
#         super(BatchNorm1d, self).__init__()

#         # TODO Initialize the attributes
#         # of 1d batchnorm module.

#         ...

#         # End of todo

#     def forward(self, x):
#         """Forward propagation of batch norm module.

#         Args:
#             x: input of shape (N, L).
#         Returns:
#             out: output of shape (N, L).
#         """

#         # TODO Implement forward propogation
#         # of 1d batchnorm module.

#         ...

#         # End of todo

#     def backward(self, dy):
#         """Backward propagation of batch norm module.

#         Args:
#             dy: output delta of shape (N, L).
#         Returns:
#             dx: input delta of shape (N, L).
#         """

#         # TODO Implement backward propogation
#         # of 1d batchnorm module.

#         ...

#         # End of todo

class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()
        
        # TODO Initialize the attributes
        # of 1d batchnorm module.
        L = length
        self.gamma = tensor.ones(L) # initialize the parameters
        self.beta = tensor.zeros(L)  # same as up
        self.momentum = momentum
        self.eps = 1e-5
        self.bn_param = {}
        self.bn_param['running_mean'] = tensor.zeros(L)
        self.bn_param['running_var'] = tensor.ones(L)
        self.x_hat = None
        self.var = None
        self.avg = None
        self.vareps = None
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.
        # https://blog.csdn.net/weixin_39228381/article/details/107896863
        # https://zhuanlan.zhihu.com/p/196277511
        
        running_mean = self.bn_param['running_mean']
        running_var = self.bn_param['running_var']
        
        N, L = x.shape  # get the batch and the length of a sample
        self.avg = np.sum(x, axis=0) / N # get the every sample's average
        self.var = np.sum((x - np.tile(self.avg, (N, 1))) ** 2, axis=0) / N  # get the every sample's variance
        self.xmu = x - self.avg
        self.vareps = (self.var + self.eps) ** 0.5   # get the Denominators
        self.x_hat = (x - np.tile(self.avg, (N, 1))) / np.tile(self.vareps, (N, 1))  # get the normalized sequence 
        
        out = self.gamma * self.x_hat + self.beta
        
        running_mean = self.momentum * running_mean + (1 - self.momentum) * self.avg
        running_var = self.momentum * running_var + (1 - self.momentum) * self.var
        self.bn_param['running_mean'] = running_mean
        self.bn_param['running_var'] = running_var
        
        return out
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.
#         N, L = dy.shape
#         var_plus_eps = self.vareps
#         self.gamma.grad = np.sum(self.x_hat * dy, axis=0)
#         self.beta.grad = np.sum(dy, axis=0)
        
#         dx_hat = dy * self.gamma   # x_hat's grad
#         x_hat = self.x_hat
# #         dx = N * dx_hat - np.sum(dx_hat, axis=0) + (1.0/N) * np.sum(dx_hat, axis=0) * np.sum(dx_hat * x_hat, axis=0) - x_hat * np.sum(dx_hat * x_hat, axis=0) 
# #         dx *= (1 - 1.0/N) / var_plus_eps
#         dx = dx_hat * (1 - 1. / N) * (1. / var_plus_eps) * (1 - 1. / (N * self.var) * self.xmu ** 2)

#         return dx
        xhat,gamma,xmu,ivar,sqrtvar,var,eps = self.x_hat, self.gamma, self.xmu, self.vareps, 1 / self.vareps, self.var, self.eps

        #get the dimensions of the input/output
        dout = dy
        N,D = dout.shape

        #step9
        dbeta = np.sum(dout, axis=0)
        dgammax = dout #not necessary, but more understandable

        #step8
        dgamma = np.sum(dgammax*xhat, axis=0)
        dxhat = dgammax * gamma

        #step7
        divar = np.sum(dxhat*xmu, axis=0)
        dxmu1 = dxhat * ivar

        #step6
        dsqrtvar = -1. /(sqrtvar**2) * divar

        #step5
        dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

        #step4
        dsq = 1. /N * np.ones((N,D)) * dvar

        #step3
        dxmu2 = 2 * xmu * dsq

        #step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)


        #step1
        dx2 = 1. /N * np.ones((N,D)) * dmu
        
        #step0
        dx = dx1 + dx2

        return dx
        ...

        # End of todo




# class Conv2d(Module):

#     def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
#                  stride: int=1, padding: int=0, bias: bool=False):
#         """Module which applies 2D convolution to input.

#         Args:
#             in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
#             channels: C_out from output shape (B, C_out, H_out, W_out).
#             kernel_size: default 3.
#             stride: default 1.
#             padding: default 0.
#         """

#         # TODO Initialize the attributes
#         # of 2d convolution module.

#         ...

#         # End of todo

#     def forward(self, x):
#         """Forward propagation of convolution module.

#         Args:
#             x: input of shape (B, C_in, H_in, W_in).
#         Returns:
#             out: output of shape (B, C_out, H_out, W_out).
#         """

#         # TODO Implement forward propogation
#         # of 2d convolution module.

#         ...

#         # End of todo

#     def backward(self, dy):
#         """Backward propagation of convolution module.

#         Args:
#             dy: output delta of shape (B, C_out, H_out, W_out).
#         Returns:
#             dx: input delta of shape (B, C_in, H_in, W_in).
#         """

#         # TODO Implement backward propogation
#         # of 2d convolution module.

#         ...

#         # End of todo


# class Conv2d_im2col(Conv2d):

#     def forward(self, x):

#         # TODO Implement forward propogation of
#         # 2d convolution module using im2col method.

#         ...

#         # End of todo

class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=False):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """
        # TODO Initialize the attributes
        # of 2d convolution module.
        r'''
        卷积层的初始化

        Parameter:
        - W: numpy.array, (C_out, C_in, K_h, K_w)
        - b: numpy.array, (C_out)
        - stride: int
        - pad: int
        '''
        self.W = tensor(np.random.randn(channels, in_channels, kernel_size, kernel_size))
        # self.b = b
        self.stride = stride
        self.pad = padding
        self.kernel_size = kernel_size
        self.x = None
        self.col = None
        self.col_W = None
        # self.dW = None   self.W.grad
        # self.db = None
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = Conv2d_im2col(x)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        
        self.x = x
        self.col = col
        self.col_W = col_W
        
        return out
        
        
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.
        
        def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
            N, C, H, W = input_shape
            out_h = (H + 2 * pad - filter_h) // stride + 1
            out_w = (W + 2 * pad - filter_w) // stride + 1
            col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

            img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
            for y in range(filter_h):
                y_max = y + stride * out_h
                for x in range(filter_w):
                    x_max = x + stride * out_w
                    img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

            return img[:, :, pad:H + pad, pad:W + pad]
        
        
        FN, C, FH, FW = self.W.shape
        dout = dy
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        # self.b.grad = np.sum(dout, axis=0)
        self.W.grad = np.dot(self.col.T, dout)
        self.W.grad = self.W.grad.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx
        ...

        # End of todo


class Conv2d_im2col(Conv2d):

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        input_data = x
        filter_h, filter_w = self.kernel_size, self.kernel_size
        stride = self.stride
        pad = self.pad
        N, C, H, W = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col
        ...

        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.

        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module.

        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        ...

        # End of todo


# class MaxPool(Module):

#     def __init__(self, kernel_size: int=2,
#                  stride: int=2, padding: int=0):
#         """Module which applies max pooling to input.

#         Args:
#             kernel_size: default 2.
#             stride: default 2.
#             padding: default 0.
#         """

#         # TODO Initialize the attributes
#         # of maximum pooling module.

#         ...

#         # End of todo

#     def forward(self, x):
#         """Forward propagation of max pooling module.

#         Args:
#             x: input of shape (B, C, H_in, W_in).
#         Returns:
#             out: output of shape (B, C, H_out, W_out).
#         """

#         # TODO Implement forward propogation
#         # of maximum pooling module.

#         ...

#         # End of todo

#     def backward(self, dy):
#         """Backward propagation of max pooling module.

#         Args:
#             dy: output delta of shape (B, C, H_out, W_out).
#         Returns:
#             out: input delta of shape (B, C, H_in, W_in).
#         """

#         # TODO Implement backward propogation
#         # of maximum pooling module.

#         ...

#         # End of todo

class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.
        self.pool_h = kernel_size
        self.pool_w = kernel_size
        self.stride = stride
        self.pad = padding
        ...

        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.
        
        # 因为对类的继承这些关系不是很清楚，不知道可不可以使用上文用到的方法来进行im2col的转化，索性在这里重新定义一个新的函数：
        def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
            N, C, H, W = input_data.shape
            out_h = (H + 2 * pad - filter_h) // stride + 1
            out_w = (W + 2 * pad - filter_w) // stride + 1

            img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
            col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

            for y in range(filter_h):
                y_max = y + stride * out_h
                for x in range(filter_w):
                    x_max = x + stride * out_w
                    col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

            col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
            return col

        
        N, C, H, W = x.shape
        FN, C, FH, FW = 1, C, self.pool_h, self.pool_w
        out_h = (H + 2 * self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1
        col = im2col(x, FH, FW, self.stride, self.pad)
        col = col.reshape((N*out_h*out_w*C, -1))
        col = np.max(col, axis=-1)
        col = col.reshape(N, out_h, out_w, C)
        col = col.transpose(0, 3, 1, 2)
        return col
        ...

        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.
        def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
            N, C, H, W = input_shape
            out_h = (H + 2 * pad - filter_h) // stride + 1
            out_w = (W + 2 * pad - filter_w) // stride + 1
            col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

            img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
            for y in range(filter_h):
                y_max = y + stride * out_h
                for x in range(filter_w):
                    x_max = x + stride * out_w
                    img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

            return img[:, :, pad:H + pad, pad:W + pad]
        
        
        dout = dy
        dout = dout.transpose(0, 2, 3, 1)
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        return dx

        ...

        # End of todo






class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.

        ...

        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.

        ...

        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.

        ...

        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()
