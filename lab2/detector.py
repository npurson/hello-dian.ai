import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet


class BoxHead(nn.Module):
    def __init__(self, lengths, num_classes):
        super(BoxHead, self).__init__()
        self.cls_score = nn.Sequential(*tuple([
                            nn.Sequential(nn.Linear(lengths[i], lengths[i + 1]),
                            nn.ReLU()) for i in range(len(lengths) - 1)] +
                            [nn.Linear(lengths[-1], num_classes)]))

        self.bbox_pred = nn.Sequential(*tuple([
                            nn.Sequential(nn.Linear(lengths[i], lengths[i + 1]),
                            nn.ReLU()) for i in range(len(lengths) - 1)] +
                            [nn.Linear(lengths[-1], 4)]))

    def forward(self, x):
        logits = self.cls_score(x)
        bbox = self.bbox_pred(x)
        # pred = torch.argmax(logits, axis=1)
        # return logits, torch.cat([b[p * 4:(p + 1) * 4] for b, p in zip(bbox, pred)]).view(-1, 4)
        return logits, bbox


class Detector(nn.Module):
    def __init__(self, backbone, lengths, num_classes):
        super(Detector, self).__init__()
        self.backbone = getattr(resnet, backbone)(pretrained=True)
        self.box_head = BoxHead(lengths, num_classes)

    def forward(self, x):
        x = self.backbone(x)        # B, 2048, 4, 4
        x = x.flatten(1)
        logits, bbox = self.box_head(x)
        return logits, bbox
