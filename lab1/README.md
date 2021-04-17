# Lab 1: Vanilla Neural Network

## Reference

### Numpy

Numpy 是一个具有强大的矩阵运算功能的 Python 包，深度学习的各类实现以及框架、包括本次作业都基于其实现。

Numpy 文档：<https://www.numpy.org.cn/>

### nn

nn 是本次作业搭建的简易深度学习框架，现对其中部分内容作简要介绍。代码中的文档有更详细的说明。

#### Tensor

`Tensor` 继承于 `np.ndarray` 类型，添加了 `grad` 属性，**可直接应用 Numpy 中的各类运算操作**。网络中所有**可通过反向传播更新的参数都应以 `Tensor` 类型创建**。

创建 `Tensor` 包括以下方法：
```python
>>> Tensor(shape)           # 继承于 np.ndarray 的构造方法
>>> tensor.zeros(shape)     # 类似 np.zeros()
>>> tensor.ones(shape)      # 类似 np.ones()
>>> tensor.from_array(arr)  # 从 np.ndarray 构造 Tensor
```

`Tensor` 的梯度应在反向传播时进行更新：
```python
>>> tensor.grad = ...
```

#### Module

`Module` 应为网络中所有模块的基类。构造其派生类时应重写其 `forward()` 和 `backward()` 方法。

对 `Module` 进行前向传播
```python
output = module(input)
```

对 `Module` 进行反向传播
```python
dx = module.backward(delta)
```

## Helloworld

实现 `nn/modules.py` 中的 `Linear` 类的 `forward()` 和 `backward()` 方法。通过 `test_module.py` 测试通过后，可运行 `helloworld.py` 查看采用一个线性层的神经网络进行简单的坐标点分类。`plot_clf.py` 会绘制分类器的决策边界，如果你的运行环境不支持图片显示，请通过 `plt.savefig()` 保存后查看。

可以看到，单层神经网络对于上述简单的任务已经可以达到很好的效果。接下来我们会实现更多的模块，构造更强大的网络实现更难的任务。

Tips：VSCode 用户可以通过代码中加入单行 `# %%` 将代码变成 Jupyter Notebook

## lab 1

实现以下模块：
- [ ] BatchNorm1d, Conv2d, AvgPool, MaxPool, DropOut
- [ ] Sigmoid, ReLU, ...
- [ ] ...
