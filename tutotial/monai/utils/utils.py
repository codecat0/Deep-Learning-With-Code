import os
import shutil
import random
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def adjust_window(image, window_level=35, window_width=85):
    # 计算窗宽和窗位的最小和最大值
    min_value = window_level - window_width // 2
    max_value = window_level + window_width // 2

    # 将图像裁剪到指定的窗宽范围内
    windowed_image = np.clip(image, min_value, max_value)

    # 归一化图像到0-255范围
    windowed_image = ((windowed_image - min_value) / (max_value - min_value) * 255).astype(np.uint8)

    return windowed_image


def visualize(original_data, aug_data=None, ori_idx=20, aug_idx=20):
    original_image = original_data["image"]
    original_mask = original_data["label"]
    if len(original_image.shape) == 4:
        original_image = original_image[0]
    if len(original_mask.shape) == 4:
        original_mask = original_mask[0]
    original_image = original_image.numpy()
    original_image = adjust_window(original_image)
    original_mask = original_mask.numpy()
    original_image = original_image[:, :, ori_idx]
    original_mask = original_mask[:, :, ori_idx]
    if aug_data is not None:
        image = aug_data["image"]
        mask = aug_data["label"]
        if len(image.shape) == 4:
            image = image[0]
        if len(mask.shape) == 4:
            mask = mask[0]
        image = image.numpy()
        image = adjust_window(image)
        image = image[:, :, aug_idx]
        mask = mask.numpy()
        mask = mask[:, :, aug_idx]
        
    
    fontsize = 12

    if aug_data is None:
        f, ax = plt.subplots(1, 2, figsize=(8, 8))

        ax[0].imshow(original_image, cmap='gray')
        ax[1].imshow(original_mask)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image, cmap='gray')
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image, cmap='gray')
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


def print_info(data):
    print(f"image shape: {data['image'].shape}, image type: {type(data['image'])}")
    print(f"label shape: {data['label'].shape}, label type: {type(data['label'])}")