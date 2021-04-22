import numpy as np
import torch


def compute_iou(bbox1, bbox2):
    if isinstance(bbox1, torch.Tensor):
        bbox1 = bbox1.detach().numpy()
    if isinstance(bbox2, torch.Tensor):
        bbox2 = bbox2.detach().numpy()
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    ix1 = np.maximum(bbox1[:, 0], bbox2[:, 0])
    ix2 = np.minimum(bbox1[:, 2], bbox2[:, 2])
    iy1 = np.maximum(bbox1[:, 1], bbox2[:, 1])
    iy2 = np.minimum(bbox1[:, 3], bbox2[:, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    return inter / (area1 + area2 - inter)
