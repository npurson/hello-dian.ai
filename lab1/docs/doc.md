# Doc for NN

NN 是本次作业搭建的简易深度学习框架，现对其中部分内容作简要介绍。代码中的文档有更详细的说明。

### Tensor

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

### Module

`Module` 应为网络中所有模块的基类。构造其派生类时应重写其 `forward()` 和 `backward()` 方法。

对 `Module` 进行前向传播
```python
>>> output = module(input)
```

对 `Module` 进行反向传播
```python
>>> dx = module.backward(dy)
```
