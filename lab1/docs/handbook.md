# Handbook

## 1. Helloworld

学习以下内容：
- 机器学习的分类与回归基本任务
- 全连接神经网络
- 梯度下降等优化方法

Tips: You can refer to ***Reference part of [README](../README.md)*** for learning materials.

代码中所有需要实现的部分采用如下标注，可以搜索 “TODO” 查找所有待完成部分。

```Python
# TODO xxxx

...

# End of todo
```

实现 `nn/modules.py` 中的 `Linear` 类的 `forward()` 和 `backward()` 方法。通过 `module_test.py` 测试后，可运行 `helloworld.py` 查看采用一个线性层的神经网络进行简单的坐标点分类。`plot_clf.py` 会绘制分类器的决策边界，如果你的运行环境不支持图片显示，请通过 `plt.savefig()` 保存后查看。

可以看到，单层神经网络对于上述简单的任务已经可以达到很好的效果。接下来我们会实现更多的模块，构造更强大的网络以实现更难的任务。

Tips：如果需要，VSCode 用户可以通过代码中加入单行 `# %%` 将 Python 文件变成 Jupyter Notebook

## 2. MNIST classification

学习以下内容：
- 反向传播
- 神经网络的基本模块：BatchNorm, Convolution, Pooling, Activation, etc.
- 损失函数

实现所有剩余代码文件中的 TODO，并通过 `module_test.py` 测试。设计分类器并在 MNIST 数据集上训练。
