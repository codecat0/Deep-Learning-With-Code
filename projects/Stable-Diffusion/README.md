# Stable Diffusion
## 数据预处理
1. 将训练数据集按照不同的长宽比（aspect ratio）进行分组（groups）或者分桶（buckets），在训练过程中，每次在buckets中随机选择一个bucket并从中采样Batch个数据进行训练
```bash
python preprocess/bucketing.py -i "data_dir" -o "output_dir"
```
