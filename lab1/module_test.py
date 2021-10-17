import random
import argparse
import functools
import inspect
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import nn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tests', nargs='*')
    return parser.parse_args()


def randnint(n, a=8, b=16):
    """Return N random integers."""
    return (random.randint(a, b) for _ in range(n))


isclose = functools.partial(np.isclose, rtol=1.e-5, atol=1.e-5)


class TestBase(object):
    def __init__(self, module, input_shape, module_params=None):
        self.module = module.split('.')[-1]
        module_params = module_params.split(',') \
                        if module_params is not None else []
        input_shape = input_shape.split('x')
        keys = set(module_params + input_shape)
        args = {k: v for k, v in zip(keys, randnint(len(keys)))}

        self.nnt = nn.tensor.tensor(tuple(args[k] for k in input_shape))
        self.ptt = torch.Tensor(self.nnt)
        self.ptt.requires_grad = True
        if '.' in module:
            self.nnm = getattr(nn.functional, self.module)(*tuple(args[k] for k in module_params))
        else:
            self.nnm = getattr(nn, module)(*tuple(args[k] for k in module_params))

    def forward_test(self) -> bool:
        # self.pt_out = ...
        self.nn_out = self.nnm(self.nnt)
        if self.nn_out is None:
            return False
        return isclose(self.nn_out,
                       self.pt_out.detach().numpy()).all().item()

    def backward_test(self) -> bool:
        if self.nn_out is None:
            return False
        self.nn_grad = self.nnm.backward(nn.tensor.ones_like(self.nn_out))
        if self.nn_grad is None:
            return False
        self.pt_out.sum().backward()
        self.pt_grad = self.ptt.grad
        return isclose(self.nn_grad, self.pt_grad.detach().numpy()).all().item()

    def __call__(self):
        def statstr(s):
            return '\033[32mpass\033[0m' if s else \
                   '\033[31mfail\033[0m'

        indent = 10
        output = (self.module if len(self.module) < indent - 2 else
                  self.module[:indent - 2]) + ': '
        output += ' ' * (indent - len(output))
        output += 'forward ' + '.' * 32 + ' ' + \
                  statstr(self.forward_test()) + \
                  '\n' + ' ' * indent + \
                  'backward ' + '.' * 31 + ' ' + \
                  statstr(self.backward_test())
        print(output)


class LinearTest(TestBase):
    def __init__(self):
        super().__init__('Linear', input_shape='BxL', module_params='L,C')

    def forward_test(self):
        self.pt_wgt = torch.Tensor(self.nnm.w[1:]).transpose(0, 1)
        self.pt_wgt.requires_grad = True
        self.pt_bias = torch.Tensor(self.nnm.w[0])
        self.pt_bias.requires_grad = True
        self.pt_out = F.linear(input=self.ptt, weight=self.pt_wgt,
                               bias=self.pt_bias)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        s &= isclose(self.nnm.w.grad[1:], self.pt_wgt.grad.transpose(0, 1)
                     .detach().numpy()).all().item()
        s &= isclose(self.nnm.w.grad[0], self.pt_bias.grad
                     .detach().numpy()).all().item()
        return s


class Conv2dTest(TestBase):
    def __init__(self):
        super().__init__('Conv2d', input_shape='BxCxHxW', module_params='C,Cp')

    def forward_test(self):
        self.pt_wgt = torch.Tensor(self.nnm.kernel)
        self.pt_wgt.requires_grad = True
        self.pt_bias = torch.Tensor(self.nnm.bias)
        self.pt_bias.requires_grad = True
        self.pt_out = F.conv2d(input=self.ptt, weight=self.pt_wgt,
                               bias=self.pt_bias)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        s &= isclose(self.nnm.kernel.grad, self.pt_wgt.grad
                     .detach().numpy()).all().item()
        s &= isclose(self.nnm.bias.grad, self.pt_bias.grad
                     .detach().numpy()).all().item()
        return s


class Conv2dIm2colTest(TestBase):
    def __init__(self):
        super().__init__('Conv2d_im2col', input_shape='BxCxHxW', module_params='C,Cp')

    def forward_test(self):
        self.pt_wgt = torch.Tensor(self.nnm.kernel)
        self.pt_wgt.requires_grad = True
        self.pt_bias = torch.Tensor(self.nnm.bias)
        self.pt_bias.requires_grad = True
        self.pt_out = F.conv2d(input=self.ptt, weight=self.pt_wgt,
                               bias=self.pt_bias)
        t1 = time.time()
        s = super().forward_test()
        t2 = time.time()
        if t2 - t1 > 0.05:
            print('\033[31mWarning: The speed of your implementation of the im2col '
                  'version of Conv2d does\'t meet our expectation!\033[0m')
        return s

    def backward_test(self):
        return True


class AvgPoolTest(TestBase):
    def __init__(self):
        super().__init__('AvgPool', input_shape='BxCxHxW')

    def forward_test(self):
        self.pt_out = F.avg_pool2d(input=self.ptt, kernel_size=2)
        return super().forward_test()

    def backward_test(self):
        return super().backward_test()


class MaxPoolTest(TestBase):
    def __init__(self):
        super().__init__('MaxPool', input_shape='BxCxHxW')

    def forward_test(self):
        self.pt_out = F.max_pool2d(input=self.ptt, kernel_size=2)
        return super().forward_test()

    def backward_test(self):
        return super().backward_test()


class SigmoidTest(TestBase):
    def __init__(self):
        super().__init__('functional.Sigmoid', input_shape='BxL')

    def forward_test(self):
        self.pt_out = torch.sigmoid(input=self.ptt)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        return s


class TanhTest(TestBase):
    def __init__(self):
        super().__init__('functional.Tanh', input_shape='BxL')

    def forward_test(self):
        self.pt_out = torch.tanh(input=self.ptt)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        return s


class ReLUTest(TestBase):
    def __init__(self):
        super().__init__('functional.ReLU', input_shape='BxL')

    def forward_test(self):
        self.pt_out = torch.relu(input=self.ptt)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        return s


class SoftmaxTest(TestBase):
    def __init__(self):
        super().__init__('functional.Softmax', input_shape='BxL')

    def forward_test(self):
        self.pt_out = torch.softmax(input=self.ptt, dim=1)
        return super().forward_test()

    def backward_test(self):
        return True


if __name__ == '__main__':
    args = parse_args()
    if args.tests:
        args.tests = [t + 'Test' for t in args.tests]
    else:
        modules = inspect.getmembers(sys.modules['__main__'], inspect.isclass)
        args.tests = [m[0] for m in modules if m[0] != 'TestBase']
    for test in args.tests:
        test_module = globals()[test]()
        test_module()
