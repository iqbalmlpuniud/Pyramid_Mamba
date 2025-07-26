import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from . import LOSS


@LOSS.register_module
class L1Loss(nn.Module):
    def __init__(self, lam=1):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += self.loss(in1, in2) * self.lam
        return loss


@LOSS.register_module
class L2Loss(nn.Module):
    def __init__(self, lam=1):
        super(L2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')
        self.lam = lam

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            loss += self.loss(in1, in2) * self.lam
        return loss


@LOSS.register_module
class CosLoss(nn.Module):
    def __init__(self, avg=True, flat=True, lam=1):
        super(CosLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity()
        self.lam = lam
        self.avg = avg
        self.flat = flat

    def forward(self, input1, input2):
        input1 = input1 if isinstance(input1, list) else [input1]
        input2 = input2 if isinstance(input2, list) else [input2]
        loss = 0
        for in1, in2 in zip(input1, input2):
            if self.flat:
                loss += (1 - self.cos_sim(in1.contiguous().view(in1.shape[0], -1), in2.contiguous().view(in2.shape[0], -1))).mean() * self.lam
            else:
                loss += (1 - self.cos_sim(in1.contiguous(), in2.contiguous())).mean() * self.lam
        return loss / len(input1) if self.avg else loss






class CSUMLoss(nn.Module):
    def __init__(self, lam=1):
        super(CSUMLoss, self).__init__()
        self.lam = lam

    def forward(self, input):
        loss = 0
        for instance in input:
            _, _, h, w = instance.shape
            loss += torch.sum(instance) / (h * w) * self.lam
        return loss


