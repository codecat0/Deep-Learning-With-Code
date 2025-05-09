#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :latent.py
@Author :CodeCat
@Date   :2025/5/9 18:19
"""
import os
import argparse
import json
import numpy as np
import torch
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Latent Extractor")
    parser.add_argument('--directory', '-d', type=str, required=True, help='Directory to extract latent from')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Output path')
    parser.add_argument('--start', '-s', required=False, default=0, type=int,
                        help='Start index of the images in directory')
    parser.add_argument('--end', '-e', required=False, type=int, help='End index of the images in directory')
    parser.add_argument('--model', '-m', required=False, default="CompVis/stable-diffusion-v1-4", type=str, help='Model name or path for autoencoder model')
    parser.add_argument('--batch_size', '-b', required=False, default=4, type=int, help='Batch size for extraction')
    parser.add_argument('--dtype', '-t', required=False, default="fp32", type=str, choices=['fp32', 'fp16', 'bf16'])
    args = parser.parse_args()
    return args


def check_and_assert_nan_tensor(tensor):
    if torch.isnan(tensor).any().item():
        raise ValueError("NaN Tensor")
    return


def extract_latents_by_dir():
    """
    根据目录提取潜在特征。

    从指定目录中读取图片文件，使用预训练的变分自编码器（VAE）提取每张图片的潜在特征，
    并将这些特征保存到指定的输出目录中。
    """
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae')
    vae.eval()
    vae.to("cuda", dtype=dtype)

    to_tensor_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    path = args.directory.rstrip('/') + '/'
    output_path = args.output_path.rstrip('/') + '/'
    os.makedirs(output_path, exist_ok=True)

    files = os.listdir(args.directory)

    end_id = len(files) if args.end is None else args.end

    image_tensors = []
    files = []
    for file in tqdm(files[args.start:end_id]):
        if "png" not in file and "jpg" not in file:
            continue

        image = Image.open(path + file)
        image_tensor = to_tensor_norm(image).to("cuda", dtype=dtype)
        image_tensors.append(image_tensor)
        files.append(file)
        if len(image_tensors) == args.batch_size:
            input_tensor = torch.stack(image_tensors)
            with torch.no_grad():
                latents = vae.encode(input_tensor).latent_dist.sample()
                check_and_assert_nan_tensor(latents)
                latents = latents.squeeze(0).cpu().numpy()
            for i in range(len(files)):
                np.save(output_path + files[i][:-4] + '.npy', latents[i])
            image_tensors = []
            files = []

    if len(files) > 0:
        input_tensor = torch.stack(image_tensors)
        with torch.no_grad():
            latents = vae.encode(input_tensor).latent_dist.sample()
            check_and_assert_nan_tensor(latents)
            latents = latents.squeeze(0).cpu().numpy()
        for i in range(len(files)):
            np.save(output_path + files[i][:-4] + '.npy', latents[i])


def extract_latents_by_json():
    """
    根据JSON文件提取潜在特征。

    从指定的JSON文件中读取图像文件路径，使用预训练的变分自编码器（VAE）提取图像的潜在特征，
    并将这些特征保存到指定的输出目录中。
    """
    vae = AutoencoderKL.from_pretrained(args.model, subfolder='vae')
    vae.eval()
    vae.to("cuda", dtype=dtype)

    to_tensor_norm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    path = args.directory.rstrip('/') + '/'
    output_path = args.output_path.rstrip('/') + '/'
    os.makedirs(output_path, exist_ok=True)

    with open(path + 'buckets.json', 'r') as f:
        buckets = json.load(f)

    for key in buckets.keys():
        image_tensors = []
        files = []
        for file in tqdm(buckets[key]):
            image = Image.open(path + file + ".png")
            image_tensor = to_tensor_norm(image).to("cuda", dtype=dtype)
            image_tensors.append(image_tensor)
            files.append(file)

            if len(files) == args.batch_size:
                input_tensor = torch.stack(image_tensors)
                with torch.no_grad():
                    latents = vae.encode(input_tensor).latent_dist.sample()
                    check_and_assert_nan_tensor(latents)
                    latents = latents.squeeze(0).cpu().numpy()
                for i in range(len(files)):
                    np.save(output_path + files[i] + '.npy', latents[i])
                image_tensors = []
                files = []

        if len(files) > 0:
            input_tensor = torch.stack(image_tensors)
            with torch.no_grad():
                latents = vae.encode(input_tensor).latent_dist.sample()
                check_and_assert_nan_tensor(latents)
                latents = latents.squeeze(0).cpu().numpy()
            for i in range(len(files)):
                np.save(output_path + files[i] + '.npy', latents[i])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    args = parse_args()
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    else:
        raise ValueError("Must be one of ['fp32', 'fp16', 'bf16']")

    if os.path.exists(os.path.join(args.directory, 'buckets.json')):
        print("Extracting by json")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
                extract_latents_by_json()
    else:
        print("Extracting by dir")
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
                extract_latents_by_dir()

