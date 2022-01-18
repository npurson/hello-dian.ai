# Handbook (Detailed Version) **[Preview]**

## 学习路径

- 机器学习简介 [https://www.bilibili.com/video/BV1FT4y1E74V?p=2](https://www.bilibili.com/video/BV1FT4y1E74V?p=2)
- 线性分类
   - [https://www.bilibili.com/video/BV1nJ411z7fe?p=6&spm_id_from=333.788.b_6d756c74695f70616765.6](https://www.bilibili.com/video/BV1nJ411z7fe?p=6&spm_id_from=333.788.b_6d756c74695f70616765.6)
   - [https://cs231n.github.io/linear-classify/](https://cs231n.github.io/linear-classify/)
- 损失函数与优化
   - [https://www.bilibili.com/video/BV1nJ411z7fe?p=7&spm_id_from=333.788.b_6d756c74695f70616765.7](https://www.bilibili.com/video/BV1nJ411z7fe?p=7&spm_id_from=333.788.b_6d756c74695f70616765.7)
- 反向传播与神经网络
   - [https://www.bilibili.com/video/BV1nJ411z7fe?p=8&spm_id_from=333.788.b_6d756c74695f70616765.8](https://www.bilibili.com/video/BV1nJ411z7fe?p=8&spm_id_from=333.788.b_6d756c74695f70616765.8)
   - [https://www.bilibili.com/video/BV1nJ411z7fe?p=9&spm_id_from=333.788.b_6d756c74695f70616765.9](https://www.bilibili.com/video/BV1nJ411z7fe?p=9&spm_id_from=333.788.b_6d756c74695f70616765.9)
   - [https://cs231n.github.io/neural-networks-1/](https://cs231n.github.io/neural-networks-1/)
   - [https://cs231n.github.io/neural-networks-2/](https://cs231n.github.io/neural-networks-2/)

## Lab1 Helloworld 讲解

根据上述内容，我们可以总结出神经网络如下的训练流程：通过线性层等 Module 将输入数据 map 到输出数据（class scores in classification），通过 Loss 量化预测结果与真实标签的正确程度，再通过反向传播更新 Module 的参数。
从自顶向下的角度设计 Module 模块，我们需要 `forward()` 方法对输入数据进行计算并输出传给下一层，也就是进行前向传播；而通过 `backward()` 方法将对 Loss 进行梯度下降优化传入的梯度用以更新当前模块的梯度，同时将梯度回传给前面的模块。
假设当前只有一个 Module，可以大致写出下面这样的伪代码：

```python
module = Module(...)  # init
while training:
    output = module.forward(input)  # forward propagation
    dy = loss(output, target).backward()
    module.backward(dx)
```
​
于是我们有了 `nn/modules.py` 中的基类设计（本 Lab 中的注释都很重要，请仔细阅读）：

```python
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

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy
```
​
以线性层为例，我们若要实现可以写出如下的**伪**代码（并不能跑）：

```python
class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.
        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """

        # TODO Initialize the weight
        # of linear module.

        self.w = Tensor((in_length + 1, out_length))

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
        return  self.w[1:] * x + self.w[0]  # 伪代码，numpy 的矩阵乘不是这么使用的

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

        self.w.grad = ...  # 计算该参数用以优化的梯度，Tips：用上 forward() 时保存的 self.x
        return ...  # 传给下一层的梯度

        # End of todo
```

上述设计也是事实上各类深度学习框架的基本实现（除了有些通过动态图反向传播）。

类似的，我们可以完成框架中的剩余部分，用以实现一系列基础的神经网络！
