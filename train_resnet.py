#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的ResNet二分类器训练脚本
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np

class CatDogDataset(Dataset):
    def __init__(self, positive_dir, negative_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 加载猫图像
        if os.path.exists(positive_dir):
            for img_name in os.listdir(positive_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(positive_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(1)  # 猫 = 1
        
        # 加载狗图像
        if os.path.exists(negative_dir):
            for img_name in os.listdir(negative_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(negative_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(0)  # 狗 = 0
        
        print(f"数据集加载完成: {len(self.images)} 张图像")
        print(f"  猫: {sum(self.labels)} 张")
        print(f"  狗: {len(self.labels) - sum(self.labels)} 张")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"图像加载失败 {img_path}: {e}")
            default_image = torch.zeros(3, 224, 224)
            return default_image, label

def main():
    print("=== 开始ResNet二分类器训练 ===")
    
    # 检查数据
    positive_dir = "data/positive"
    negative_dir = "data/negative"
    
    if not os.path.exists(positive_dir) or not os.path.exists(negative_dir):
        print(" 数据目录不存在！")
        return
    
    # 创建数据集
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CatDogDataset(positive_dir, negative_dir, transform)
    
    if len(dataset) == 0:
        print(" 数据集为空！")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"验证集: {len(val_dataset)} 张图像")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 创建模型
    print("\n创建ResNet模型...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 冻结预训练层
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后的分类层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 二分类
    
    print("模型创建完成！")
    
    # 训练设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 开始训练
    print("\n开始训练...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += torch.sum(preds == labels)
        
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct.double() / len(train_loader.dataset)
        
        print(f"训练 Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_correct += torch.sum(preds == labels)
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct.double() / len(val_loader.dataset)
        
        print(f"验证 Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    print("\n训练完成！")
    
    # 保存模型
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet_classifier.pth")
    print("模型已保存到: models/resnet_classifier.pth")

if __name__ == "__main__":
    main()
