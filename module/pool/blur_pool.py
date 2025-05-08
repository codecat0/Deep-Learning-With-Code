#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :blur_pool.py
@Author :CodeCat
@Date   :2025/5/8 16:42
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Type


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1):
    """
    The fill size is calculated based on the convolution kernel size, step size, and expansion rate.

    Args:
        kernel_size (int): Convolution kernel size。
        stride (int, optional): Stride size, Default: 1。
        dilation (int, optional): Dilation size, Default: 1。

    Returns:
        int: padding size。

    """
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d(nn.Module):
    """
    Blur Pooling layer.
    paper: Making Convolutional Networks Shift-Invariant Again
    ref: https://arxiv.org/pdf/1904.11486.pdf
    """

    def __init__(self,
                 channels: Optional[int],
                 filt_size: int = 3,
                 stride: int = 2,
                 pad_mode: str = 'reflect') -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.pad_mode = pad_mode
        self.padding = [get_padding(filt_size, stride=stride, dilation=1)] * 4

        coeffs = torch.tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(np.float32))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :]
        if channels is not None:
            blur_filter = blur_filter.repeat(channels, 1, 1, 1)
        self.register_buffer('filt', blur_filter, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, mode=self.pad_mode)
        if self.channels is None:
            channels = x.shape[1]
            weight = self.filt.expand(channels, 1, self.filt_size, self.filt_size).to(x.device)
        else:
            channels = self.channels
            weight = self.filt.to(x.device)
        out = F.conv2d(x, weight, groups=channels, stride=self.stride)
        return out


if __name__ == '__main__':
    inp = torch.randn(1, 64, 128, 128)
    model = BlurPool2d(64)
    out = model(inp)
    print(out.shape)  # torch.Size([1, 64, 64, 64])
