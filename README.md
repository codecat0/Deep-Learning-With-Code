# Deep Learning With Code

## 1. 简介
本项目致力于实现一个既能让**深度学习小白也能搞懂**，又能**服务科研和工业社区**的代码库。包含但不限于：
- 论文解读，论文核心代码实现，以及配套的教程；
- 计算机视觉、自然语言处理、大模型等领域的经典算法实现；
- 计算机视觉项目、自然语言处理项目、大模型项目的实战案例；
- 常用深度学习框架的教程。

## 2. 目录结构
```yaml
|--DeepLearningWithCode
    |-- README.md：项目说明
    |-- docs：文档说明
    |-- tutorial：常用框架教程
    |-- module：论文核心代码
    |-- projects：项目实战
```

### 2.1 module 核心代码模块
#### [Attention模块](https://github.com/codecat0/Deep-Learning-With-Code/blob/master/docs/module_docs/Attention.md)
- [ScaledDotProductAttention](https://github.com/codecat0/Deep-Learning-With-Code/blob/master/module/attention/self_attention.py)：[Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [SEAttention](https://github.com/codecat0/Deep-Learning-With-Code/blob/master/module/attention/se_attention.py)：[Squeeze-and-Excitation Attention](https://arxiv.org/abs/1709.01507)
- [CBAM](https://github.com/codecat0/Deep-Learning-With-Code/blob/master/module/attention/cbam.py)：[CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)

#### [Pooling模块](https://github.com/codecat0/Deep-Learning-With-Code/blob/master/docs/module_docs/Pooling.md)
- [BlurPooling](https://github.com/codecat0/Deep-Learning-With-Code/blob/master/module/pool/blur_pool.py)：[Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/pdf/1904.11486.pdf)