import cv2
import numpy as np
from PIL import Image
import open_clip
import torch
def preprocess_for_painting(image_path, model_name="ViT-H-14", pretrained="laion2b_s32b_b79k", device=None):
    # 读取图像并转换为 RGB 格式
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(image_path)

    # 转换为 RGB 格式
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像从 NumPy 转换为 PIL Image
    pil_img = Image.fromarray(img)

    # 返回 PIL 图像，以便 preprocess 使用
    return pil_img
