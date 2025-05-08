#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :bucketing.py
@Author :CodeCat
@Date   :2025/5/8 19:21
"""
import os
from tqdm import tqdm
import glob
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import json
import argparse
import numpy as np
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(description='Bucketing')
    parser.add_argument('--input_dir', '-i', type=str, required=True, help="Original dataset directory")
    parser.add_argument('--output_dir', '-o', type=str, required=False, help="Output bucketed dataset directory")
    parser.add_argument('--threads', '-p', required=False, default=12, type=int, help="Number of processes")
    parser.add_argument('--resolution', '-r', required=False, default=768, type=int, help="bucketing resolution")
    parser.add_argument('--min_length', required=False, default=512, type=int, help="bucketing minimum length")
    parser.add_argument('--max_length', required=False, default=1024, type=int, help="bucketing maximum length")
    parser.add_argument('--max_ratio', required=False, default=2.0, type=float, help="bucketing max ratio")
    args = parser.parse_args()
    return args


def make_bucktets(args):
    """
    根据给定的参数生成一系列尺寸和比例合适的图片尺寸桶。

    Args:
        args (argparse.Namespace): 包含生成尺寸桶所需参数的命名空间对象。

    Returns:
        tuple: 包含两个元素的元组，第一个元素为尺寸桶列表（list of tuple），第二个元素为比例列表（list of float）。

    """
    increment = 64
    max_pixels = args.resolution * args.resolution

    buckets = set()
    buckets.add((args.resolution, args.resolution))

    width = args.min_length

    while width <= args.max_length:
        height = min(args.max_length, (max_pixels // width) - (max_pixels // width) % increment)
        ratio = width / height

        if 1 / args.max_ratio <= ratio <= args.max_ratio:
            buckets.add((width, height))
            buckets.add((height, width))

        width += increment

    buckets = list(buckets)
    ratios = [w / h for w, h in buckets]
    buckets = np.array(buckets)[np.argsort(ratios)]
    ratios = np.sort(ratios)
    return buckets, ratios


def resize_image(file, ratios, buckets, args):
    """
    根据给定的比例和尺寸桶，调整图像尺寸并裁剪到合适的尺寸。

    Args:
        file (str): 图像文件的路径。
        ratios (list of float): 比例列表。
        buckets (list of tuple): 尺寸桶列表，每个元素是一个包含宽度和高度的元组。
        args (argparse.Namespace): 包含输出目录等参数的命名空间对象。

    Returns:
        list: 包含两个元素的列表，第一个元素是图像文件的基名（不包含扩展名），第二个元素是调整后的图像尺寸（宽度，高度）。

    """
    image = Image.open(file)
    image = image.convert("RGB")
    ratio = image.width / image.height
    ar_errors = ratios - ratio
    indice = np.argmin(np.abs(ar_errors))
    bucket_width, bucket_height = buckets[indice]
    ar_error = ar_errors[indice]

    if ar_error <= 0:
        tmp_width = int(image.width * (bucket_height / image.height))
        image = image.resize((tmp_width, bucket_height))
        left = (tmp_width - bucket_width) / 2
        right = bucket_width + left
        image = image.crop((left, 0, right, bucket_height))
    else:
        tmp_height = int(image.height * (bucket_width / image.width))
        image = image.resize((bucket_width, tmp_height))
        top = (tmp_height - bucket_height) / 2
        bottom = bucket_height + top
        image = image.crop((0, top, bucket_width, bottom))

    image.save(os.path.join(args.output_dir, os.path.basename(file)))
    return [os.path.splitext(os.path.basename(file))[0], str((bucket_width, bucket_height))]


def main(args):
    files = []
    [files.extend(glob.glob(f'{args.input_dir}' + '/*.' + e)) for e in
     ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'JPG', 'JPEG', 'PNG', 'WEBP', 'BMP']]

    with ProcessPoolExecutor(8) as e:
        results = list(tqdm(e.map(resize_image, files), total=len(files)))

    meta = {}
    for file, bucket in results:
        if bucket in meta:
            meta[bucket].append(file)
        else:
            meta[bucket] = [file]

    for key in meta:
        logger.info(f"{key} : {len(meta[key])}")

    with open(os.path.join(args.output_dir, 'buckets.json'), 'w') as f:
        json.dump(meta, f)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    buckets, ratios = make_bucktets(args)
    main(args)
