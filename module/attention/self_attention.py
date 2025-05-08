#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :self_attention.py
@Author :CodeCat
@Date   :2025/5/7 18:51
"""
import torch
import torch.nn as nn
from torch.nn import init
from typing import Optional, Type


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    paper: Attention is All You Need
    ref: https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                 d_model: Optional[int],
                 d_k: Optional[int],
                 d_v: Optional[int],
                 num_heads: int = 4,
                 dropout: float = 0.1) -> None:
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param num_heads: Number of attention heads
        :param dropout: Dropout rate
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, d_k * num_heads)
        self.fc_k = nn.Linear(d_model, d_k * num_heads)
        self.fc_v = nn.Linear(d_model, d_v * num_heads)
        self.fc_o = nn.Linear(d_v * num_heads, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        self.init_weights()

    def forward(self,
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                attn_mask: torch.Tensor = None,
                attn_weights: torch.Tensor = None) -> torch.Tensor:
        """
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attn_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking. False indicates no mask. Default: None.
        :param attn_weights: Multiplicative weights for attention values (b_s, h, nq, nk). Default: None.
        :return: Output from Self-Attention layer (b_s, nq, d_model)
        """
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.num_heads, self.d_k).permute(0, 2, 1, 3)   # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.num_heads, self.d_k).permute(0, 2, 3, 1)      # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.num_heads, self.d_v).permute(0, 2, 1, 3)    # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / self.d_k ** 0.5   # (b_s, h, nq, nk)
        if attn_weights is not None:
            att = att * attn_weights
        if attn_mask is not None:
            att = att.masked_fill(attn_mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.d_v * self.num_heads)  # (b_s, h, nq, d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
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
    inp = torch.randn(8, 50, 512)
    sa = ScaledDotProductAttention(d_model=512, d_k=64, d_v=64, num_heads=8)
    out = sa(inp, inp, inp)
    print(out.shape)   # torch.Size([8, 50, 512])
