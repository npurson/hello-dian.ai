import numpy as np
import torch
import torch.nn.functional as F

import nn


class TestBase(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.types = ['maxPooling', 'Conv2D', 'BN', 'FC']

        assert self.kwargs.get('type', None) is not None, "未指定测试模块类型，请添加'type'关键字"
        self.module_type = self.kwargs.get('type')
        assert self.module_type in self.types, "指定模块无效"

        # 判断类型选择不同初始化方式
        self.input_numpy = None
        if self.module_type == 'maxPooling':
            self.input_numpy = np.random.rand(2, 2, 4, 4)
        else:
            self.input_numpy = np.random.rand(5, 4)
        self.input_tensor = torch.Tensor(self.input_numpy)
        self.input_tensor.requires_grad = True

        # 根据不同的网络选择方式，预留不同的打印信息的方式
        self.w = None
        self.model_tensor = None
        self.model_numpy = None

    def forward(self):
        self.output_tensor = self.model_tensor(self.input_tensor)
        self.output_numpy = self.model_numpy(self.input_numpy)
        if self.module_type == 'BN':
            self.output_tensor.backward(self.output_tensor_delta)
        else:
            self.output_tensor_delta = self.output_tensor.sum()
            self.output_numpy_delta = np.ones_like(self.output_numpy)
            self.output_tensor_delta.backward()

        self.printInfo()

    def printInfo(self):
        print("Input shape is: ===============>>>>>\t", self.input_numpy.shape)
        print("\033[1;34;43mThe input matrix is:\033[0m")
        print(self.input_numpy)
        if self.model_numpy == "FC":
            print("The W matrix shape is: ===============>>>>>\t", self.w.shape)
            print("The W matrix is:")
            print(self.w)
        print("{:*^60}".format(''))
        print("{:*^71}".format('\033[0;31m' + self.module_type + ' Layer Test\033[0m'))
        print("{:*^60}".format(''))

        print("1. Using your own Linear code.....\n")
        print(self.output_numpy)
        print("{:*^50}".format("The grad is as follows:"))
        print(self.model_numpy.backward(self.output_numpy_delta))
        print()

        print("2. Here is the official code.....\n")
        print(self.output_tensor)
        print("{:*^50}".format("The grad is as follows:"))
        print(self.input_tensor.grad)


class TestModule(TestBase):

    def __init__(self, **kwargs):
        super(TestModule, self).__init__(**kwargs)
        # 偏置矩阵初始化
        if self.module_type == "FC":
            self.FCInit()
        elif self.module_type == "BN":
            self.BNInit()

    def FCInit(self):
        in_length = self.input_numpy.shape[1]
        out_length = 4
        self.w = np.random.normal(loc=0.0, scale=0.1, size=(out_length, in_length + 1))
        self.w_tensor = torch.Tensor(self.w)
        # 初始化torch层
        self.model_tensor = torch.nn.Linear(in_features=in_length, out_features=out_length, bias=True)
        self.model_tensor.bias.data = self.w_tensor[:, 0]
        self.model_tensor.weight.data = self.w_tensor[:, 1:]
        # 初始化numpy层
        self.model_numpy = nn.Linear(in_length=in_length, out_length=out_length, w=self.w)

    def BNInit(self):
        self.output_numpy_delta = np.random.rand(self.input_numpy.shape[0], self.input_numpy.shape[1])
        self.output_tensor_delta = torch.tensor(self.output_numpy_delta, requires_grad=True)
        self.model_tensor = torch.nn.BatchNorm1d(num_features=self.input_numpy.shape[1], eps=1e-5, momentum=0.9, affine=True)
        self.model_numpy = nn.BatchNorm1d(length=self.input_numpy.shape[1])

    def __call__(self):
        self.forward()


if __name__ == '__main__':
    t = TestModule(type='BN')
    t()
