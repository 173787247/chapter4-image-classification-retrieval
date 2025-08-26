#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像数据集测试程序
"""

import os
from PIL import Image

def test_image_dataset():
    """测试图像数据集"""
    print("=== 图像数据集测试 ===")
    
    # 检查正类图像
    positive_dir = "data/positive"
    if os.path.exists(positive_dir):
        positive_images = [f for f in os.listdir(positive_dir) if f.endswith(".jpg")]
        print(f" 正类图像（猫）: {len(positive_images)} 张")
        for img in positive_images[:3]:  # 显示前3张
            img_path = os.path.join(positive_dir, img)
            try:
                with Image.open(img_path) as im:
                    print(f"  - {img}: {im.size[0]}x{im.size[1]} pixels")
            except Exception as e:
                print(f"  - {img}: 读取失败 - {e}")
    else:
        print(" 正类图像目录不存在")
    
    # 检查负类图像
    negative_dir = "data/negative"
    if os.path.exists(negative_dir):
        negative_images = [f for f in os.listdir(negative_dir) if f.endswith(".jpg")]
        print(f" 负类图像（狗）: {len(negative_images)} 张")
        for img in negative_images[:3]:  # 显示前3张
            img_path = os.path.join(negative_dir, img)
            try:
                with Image.open(img_path) as im:
                    print(f"  - {img}: {im.size[0]}x{im.size[1]} pixels")
            except Exception as e:
                print(f"  - {img}: 读取失败 - {e}")
    else:
        print(" 负类图像目录不存在")
    
    # 检查查询图像
    query_dir = "data/query"
    if os.path.exists(query_dir):
        query_images = [f for f in os.listdir(query_dir) if f.endswith(".jpg")]
        print(f" 查询图像: {len(query_images)} 张")
        for img in query_images:
            img_path = os.path.join(query_dir, img)
            try:
                with Image.open(img_path) as im:
                    print(f"  - {img}: {im.size[0]}x{im.size[1]} pixels")
            except Exception as e:
                print(f"  - {img}: 读取失败 - {e}")
    else:
        print(" 查询图像目录不存在")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_image_dataset()
