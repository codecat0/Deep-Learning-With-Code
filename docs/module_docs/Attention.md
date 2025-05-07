# Attention 模块
## 1. Self-Attention
### 1.1 简介
Self-Attention 是 Transformer 模型的核心组件，它允许模型在处理序列数据时能够关注到序列中不同位置的信息。与传统循环神经网络（RNN）或卷积神经网络（CNN）相比，Self-Attention 能够并行地处理所有输入位置的关联信息，从而提高了模型的效率和性能。
### 1.2 工作原理
Self-Attention 通过计算查询向量（Query）、键向量（Key）和值向量（Value）之间的加权平均来实现对序列数据的注意力机制。具体步骤如下：
1. **生成 Q、K、V**：首先，将输入的序列通过线性变换得到三个不同的表示矩阵 Q、K 和 V。这三个矩阵分别代表了查询、键和值的维度空间。
   
   - $Q = W_q \cdot X$
   - $K = W_k \cdot X$
   - $V = W_v \cdot X$
   其中，$X$ 是输入序列，$W_q, W_k, W_v$ 是可学习的权重矩阵。

2. **计算相似度得分**：然后，使用点积操作来计算每个Q与每个K之间的相似度得分。
3. **应用 Softmax 归一化**：对相似度得分进行Softmax处理，以得到每个Q与所有K的注意力权重。
4. **加权求和**：最后，将得到的注意力权重应用于V上，得到最终的输出。
5. **输出结果**
### 1.3 模型架构图
![](https://pic1.imgdb.cn/item/681b45c558cb8da5c8e38749.png)
### 1.4 使用方法
```python
from attention.self_attention import ScaledDotProductAttention
import torch

inp = torch.randn(8, 50, 512)
sa = ScaledDotProductAttention(d_model=512, d_k=64, d_v=64, num_heads=8)
out = sa(inp, inp, inp)
print(out.shape)   # torch.Size([8, 50, 512])
```

## 2. SE-Attention
### 2.1 简介
SE-Attention（Squeeze-and-Excitation Attention）是一种用于增强深度神经网络特征表示的注意力机制。它通过自适应地重新校准通道维度上的特征响应，使得模型能够更加关注于重要的信息。
### 2.2 工作原理
SE-Attention 主要由两个部分组成：**Squeeze操作**和**Excitation操作**。
1. **Squeeze操作**：将空间维度压缩到单个值，通常是通过全局平均池化实现的。
   
   - $s = \text{AvgPool}(X)$
   其中，$X$ 是输入特征图。

2. **Excitation操作**：使用全连接层对Squeeze操作的输出进行变换，并通过Sigmoid激活函数生成权重向量。
   
   - $z = f(W_1 s + b_1)$
   - $\alpha = \sigma(W_2 z + b_2)$
   其中，$\sigma$是Sigmoid函数，$f$可以是ReLU或其他非线性激活函数，$W_1, W_2$ 和 $b_1, b_2$ 是可学习的参数
   - 最终的权重向量 $\alpha$ 用于重新校准输入特征图。
   - 输出 $Y = X \times \alpha$

### 2.3 模型架构图
![](https://pic1.imgdb.cn/item/681b472658cb8da5c8e38799.png)

### 2.4 使用方法
```python
from attention.se_attention import SEAttention
import torch

inp = torch.randn(4, 512, 7, 7)
se = SEAttention(in_dim=512, reduction=16)
out = se(inp)
print(out.shape)   # torch.Size([4, 512, 7, 7])
```

## 3. CBAM-Attention
### 3.1 简介
CBAM（Convolutional Block Attention Module）是一种结合了通道注意力和空间注意力机制的模块，旨在提高深度神经网络对特征表示的关注能力。
### 3.2 工作原理
CBAM 由两个子模块组成：**Channel Attention Module (CAM)** 和 **Spatial Attention Module (SAM)**。
1. **Channel Attention Module (CAM)**：类似于SE-Attention，但使用了额外的卷积层来增强特征的表达能力。
   
   - 使用全局平均池化和最大池化得到两个不同的描述符。
   - 通过全连接和Sigmoid激活函数生成权重向量 $\alpha$。

2. **Spatial Attention Module (SAM)**：使用卷积操作来生成空间注意力图。

   - 输入特征图通过卷积层和Sigmoid激活函数生成空间注意力权重。
   - 最终的输出是CAM和SAM的组合。

### 3.3 模型架构图
![](https://pic1.imgdb.cn/item/681b481758cb8da5c8e387ce.png)

### 3.4 使用方法
```python
from attention.cbam_attention import CBAM
import torch

inp = torch.randn(4, 512, 7, 7)
cbam = CBAM(in_planes=512, ratio=16, kernel_size=7)
out = cbam(inp)
print(out.shape)   # torch.Size([4, 512, 7, 7])
```