#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :se_attention.py
@Author :CodeCat
@Date   :2025/5/7 19:12
"""
import torch
import torch.nn as nn
from torch.nn import init


class SEAttention(nn.Module):
    """
    paper: Squeeze-and-Excitation Attention
    ref: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, in_dim, reduction=16):
        """
        :param in_dim: input dimension
        :param reduction: reduction ratio
        """
        super(SEAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim // reduction),
            nn.ReLU(),
            nn.Linear(in_features=in_dim // reduction, out_features=in_dim),
            nn.Sigmoid()
        )
        self.init_weights()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


if __name__ == '__main__':
    inp = torch.randn(4, 512, 7, 7)
    se = SEAttention(in_dim=512, reduction=16)
    out = se(inp)
    print(out.shape)   # torch.Size([4, 512, 7, 7])