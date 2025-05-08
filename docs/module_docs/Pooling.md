# Pooling模块
## 1. [BlurPooling](https://arxiv.org/pdf/1904.11486.pdf)
### 1.1 简介
BlurPooling是一种用于图像处理的池化操作，它通过模糊（blur）的方式来降低特征图的分辨率。这种方法在保持信息丰富性的同时，减少了计算量和参数数量，适用于深度学习中的卷积神经网络中。
### 1.2 工作原理
BlurPooling的工作原理类似于传统的池化操作，如最大池化（Max Pooling）或平均池化（Average Pooling），但它通过应用一个模糊核来代替直接的降采样。具体步骤如下：
1. **选择模糊核**：通常使用高斯模糊核或其他类型的低通滤波器作为模糊核。
2. **卷积运算**：将选定的模糊核与输入特征图进行卷积运算。这一步的目的是在保持信息丰富性的同时降低分辨率。
3. **可选的下采样**：在某些实现中，为了进一步减少计算量，可以在模糊后的结果上进行下采样操作，例如使用最近邻插值法或双线性插值法减小尺寸。
### 1.3 模型架构图
![](https://pic1.imgdb.cn/item/681c773b58cb8da5c8e58bcf.png)
![](https://pic1.imgdb.cn/item/681c772558cb8da5c8e58bba.png)
### 1.4 使用方法
```python
from pool.blur_pool import BlurPool2d
import torch

inp = torch.randn(1, 64, 128, 128)
model = BlurPool2d(64)
out = model(inp)
print(out.shape)  # torch.Size([1, 64, 64, 64])
```