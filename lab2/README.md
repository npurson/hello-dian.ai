# Lab 2: Single Object Detection

Implement a single object detector with PyTorch, and train on [Tiny-VID](http://xinggangw.info/data/tiny_vid.zip) dataset.
For each classes, use 150 samples for training and the rest for testing.

## Prequisity

1. Watch all parts till **Object Detection** of CS231n.
2. Paper reading: Faster R-CNN, [Optional] YOLO
3. **PyTorch Tutorials**: <https://pytorch.org/tutorials>. Only ***Introduction to PyTorch*** and ***Learning PyTorch*** parts are needed.

## Handbook

1. Complete `TvidDataset`, `Detector`, `compute_iou`, and the rest in `main.py`.
   Tips: Faster R-CNN is not required.
2. Train on [Tiny-VID](http://xinggangw.info/data/tiny_vid.zip) dataset.
   Accuracy, defined as $\frac{samples \ classified \ correctly \ \wedge \ IoU \ > \ 0.5}{total}$, should be at least 70%.
3. Visualize detection results.
4. Expected completion time: 2~3 days.
