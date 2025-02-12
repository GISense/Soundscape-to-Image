from __future__ import division
from PIL import Image
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_the_blackborder(image_p):
    image = cv2.imread(image_p)  # 读取图片
    img = cv2.medianBlur(image, 5)  # 中值滤波，去除黑色边际中可能含有的噪声干扰
    b = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)  # 调整裁剪效果
    binary_image = b[1]  # 二值图--具有三通道
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    try:
        edges_y, edges_x = np.where(binary_image == 255)  ##h, w
        bottom = min(edges_y)
        top = max(edges_y)
        height = top - bottom

        left = min(edges_x)
        right = max(edges_x)
        height = top - bottom
        width = right - left

        res_image = image[bottom:bottom + height, left:left + width]

    except:
        res_image=image
        print(image_p)
    return res_image



def load_image(image_path, transform=None, max_size=None, shape=None):
    image=remove_the_blackborder(image_path)
    image=Image.fromarray(image)
    #image = Image.open(image_path)  # 读入图片， 下面是一些图片的预处理操作
    if max_size:
        scale = max_size / max(image.size)
        size = np.array(image.size) * scale
        # print(size)         [400.         306.78733032]
        image = image.resize(size.astype(int), Image.ANTIALIAS)  # 改变图片大小

    if shape:
        image = image.resize(shape, Image.LANCZOS)  # Image.LANCZOS是插值的一种方式

    if transform:
        # print(image)         # PIL的JpegImageFile格式(size=(W，H))
        image = transform(image).unsqueeze(0)
        # print(image.shape)   #   [C, H, W]
    return image.to(device)
