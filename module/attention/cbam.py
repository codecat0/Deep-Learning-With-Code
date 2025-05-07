#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :cbam.py
@Author :CodeCat
@Date   :2025/5/7 19:20
"""
import torch
import torch.nn as nn
from torch.nn import init


class ChannelAttention(nn.Module):
    """
    Channel Attention
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.ce = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.avg_pool(x)
        maxout = self.max_pool(x)
        avgout = self.ce(avgout)
        maxout = self.ce(maxout)
        out = self.sigmoid(avgout + maxout)
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    paper: CBAM: Convolutional Block Attention Module
    ref: https://arxiv.org/abs/1807.06521
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio=ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)
        self.init_weights()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.ca(x) * x
        out = self.sa(out) * out
        return out

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
    cbam = CBAM(in_planes=512, ratio=16, kernel_size=7)
    out = cbam(inp)
    print(out.shape)   # torch.Size([4, 512, 7, 7])