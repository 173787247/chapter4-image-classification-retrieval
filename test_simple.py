#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试程序
"""

print("=== 简单测试程序 ===")

try:
    import torch
    print(" PyTorch导入成功")
    
    # 添加src目录到路径
    import sys
    sys.path.append("src")
    
    # 测试ResNet分类器
    from resnet_classifier import ResNetBinaryClassifier
    print(" ResNet分类器导入成功")
    
    # 创建模型
    classifier = ResNetBinaryClassifier()
    print(" 模型创建成功")
    
    # 检查CUDA可用性
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f" 使用设备: {device}")
    
    # 模型参数统计
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f" 总参数数量: {total_params:,}")
    print(f" 可训练参数数量: {trainable_params:,}")
    
    print(" 所有测试通过！")
    
except Exception as e:
    print(f" 测试失败: {e}")
    import traceback
    traceback.print_exc()
