#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像下载脚本
从互联网抓取图像用于测试
"""

import os
import requests
from PIL import Image
import io
import time
from urllib.parse import urlparse
import random

def download_image(url, save_path):
    """下载单张图像"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 检查是否为图像
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return False, "不是图像文件"
        
        # 保存图像
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        return True, "下载成功"
        
    except Exception as e:
        return False, str(e)

def create_sample_images():
    """创建示例图像数据集"""
    
    # 创建目录
    os.makedirs("data/positive", exist_ok=True)
    os.makedirs("data/negative", exist_ok=True)
    os.makedirs("data/query", exist_ok=True)
    
    # 正类图像：猫
    cat_images = [
        "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1529778873920-4da4926a72c2?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1533738363-b7f0a2134eae?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1543852786-1cf6624b3d12?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1557246565-8a3d3ab5d7f6?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1561948955-570b270e7c36?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1596854407944-b4c6c67e06e9?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1606216794074-735e91aa2c92?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1615789591457-74a63395c990?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1622227922680-52091f6e0b0a?w=400&h=400&fit=crop"
    ]
    
    # 负类图像：狗
    dog_images = [
        "https://images.unsplash.com/photo-1547407139-3c921a641edc?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1568393691622-c7ba131d63b4?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1576201836106-db1758fd1c97?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1591160690555-5debfba289f0?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1605897472359-2481e0a50a1a?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1615789591457-74a63395c990?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1628009368231-7bbbbd3a7ac6?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400&h=400&fit=crop",
        "https://images.unsplash.com/photo-1633722715463-d30f4f325e24?w=400&h=400&fit=crop"
    ]
    
    print("开始下载图像...")
    
    # 下载正类图像（猫）
    print("\n下载正类图像（猫）...")
    for i, url in enumerate(cat_images):
        filename = f"cat_{i+1:02d}.jpg"
        save_path = os.path.join("data/positive", filename)
        
        success, message = download_image(url, save_path)
        if success:
            print(f" {filename}: {message}")
        else:
            print(f" {filename}: {filename}: {message}")
        
        time.sleep(0.5)  # 避免请求过快
    
    # 下载负类图像（狗）
    print("\n下载负类图像（狗）...")
    for i, url in enumerate(dog_images):
        filename = f"dog_{i+1:02d}.jpg"
        save_path = os.path.join("data/negative", filename)
        
        success, message = download_image(url, save_path)
        if success:
            print(f" {filename}: {message}")
        else:
            print(f" {filename}: {filename}: {message}")
        
        time.sleep(0.5)  # 避免请求过快
    
    # 复制一些图像到查询目录
    print("\n准备查询图像...")
    import shutil
    
    # 复制几张猫的图像作为查询图像
    for i in range(3):
        src = os.path.join("data/positive", f"cat_{i+1:02d}.jpg")
        dst = os.path.join("data/query", f"query_cat_{i+1}.jpg")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f" 复制查询图像: query_cat_{i+1}.jpg")
    
    # 复制几张狗的图像作为查询图像
    for i in range(2):
        src = os.path.join("data/negative", f"dog_{i+1:02d}.jpg")
        dst = os.path.join("data/query", f"query_dog_{i+1}.jpg")
        print(f" 复制查询图像: query_dog_{i+1}.jpg")
    
    print("\n图像下载完成！")
    
    # 统计下载结果
    positive_count = len([f for f in os.listdir("data/positive") if f.endswith(".jpg")])
    negative_count = len([f for f in os.listdir("data/negative") if f.endswith(".jpg")])
    query_count = len([f for f in os.listdir("data/negative") if f.endswith(".jpg")])
    
    print(f"\n下载统计:")
    print(f"  正类图像（猫）: {positive_count} 张")
    print(f"  负类图像（狗）: {negative_count} 张")
    print(f"  查询图像: {query_count} 张")
    print(f"  总计: {positive_count + negative_count + query_count} 张")

if __name__ == "__main__":
    create_sample_images()
