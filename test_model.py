#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的ResNet二分类器
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

def load_trained_model(model_path):
    """加载训练好的模型"""
    # 创建模型架构
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    # 加载训练好的权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f" 模型加载成功: {model_path}")
        return model
    else:
        print(f" 模型文件不存在: {model_path}")
        return None

def predict_image(model, image_path, transform):
    """预测单张图像"""
    try:
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
        
        # 预测
        model.eval()
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        class_names = ["狗", "猫"]
        return class_names[predicted_class], confidence
        
    except Exception as e:
        print(f"预测失败 {image_path}: {e}")
        return None, 0.0

def main():
    print("=== 测试训练好的ResNet二分类器 ===")
    
    # 加载模型
    model_path = "models/resnet_classifier.pth"
    model = load_trained_model(model_path)
    
    if model is None:
        return
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 测试查询图像
    query_dir = "data/query"
    if os.path.exists(query_dir):
        query_images = [f for f in os.listdir(query_dir) if f.endswith(".jpg")]
        
        print(f"\n测试查询图像 ({len(query_images)} 张):")
        print("-" * 50)
        
        for img_name in query_images:
            img_path = os.path.join(query_dir, img_name)
            predicted_class, confidence = predict_image(model, img_path, transform)
            
            if predicted_class:
                print(f"图像: {img_name}")
                print(f"预测: {predicted_class}")
                print(f"置信度: {confidence:.4f}")
                print("-" * 30)
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
