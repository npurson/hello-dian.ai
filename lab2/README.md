# Lab 2: Detector

在 [Tiny VID](http://xinggangw.info/data/tiny_vid.zip) 数据集上进行目标检测。对于数据集中每个类的样本，前 150 张用于训练，剩余用于测试。

## Prequisity

1. 看完 CS231n 目标检测及之前的内容
1. 阅读论文：Faster R-CNN
1. PyTorch 官方教程：<https://pytorch.org/tutorials/>；只需看 *Introduction to PyTorch* 和 *Learning PyTorch* 其中的**部分**，不需要看全部。

## Handbook

1. 完成 `tvid.py` 中 `torch.utils.data.Dataset` 的实现
1. 完成 `detector.py` 中目标检测网络的实现，不需要达到 Faster R-CNN 的复杂程度，但可以尝试加入 Anchor 等，或自由发挥。
1. 完成 `main.py`，使用 `utils.compute_iou` 计算 IoU，acc 指标为**分类正确且 IoU > 0.5 的样本数 / 总数**，要求 acc 达到 0.7 以上
1. 可视化目标检测结果，不论实现方式
1. 实验预计时间：3 天（不包括 Prequisity）
