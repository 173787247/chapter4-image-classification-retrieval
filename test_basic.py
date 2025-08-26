#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试程序
"""

print("=== 第四章作业测试程序 ===")

# 测试基本导入
try:
    import torch
    print(" PyTorch导入成功")
    print(f"PyTorch版本: {torch.__version__}")
except ImportError as e:
    print(f" PyTorch导入失败: {e}")

try:
    import torchvision
    print(" TorchVision导入成功")
    print(f"TorchVision版本: {torchvision.__version__}")
except ImportError as e:
    print(f" TorchVision导入失败: {e}")

try:
    import PIL
    print(" PIL导入成功")
    print(f"PIL版本: {PIL.__version__}")
except ImportError as e:
    print(f" PIL导入失败: {e}")

try:
    import numpy as np
    print(" NumPy导入成功")
    print(f"NumPy版本: {np.__version__}")
except ImportError as e:
    print(f" NumPy导入失败: {e}")

try:
    import matplotlib
    print(" Matplotlib导入成功")
    print(f"Matplotlib版本: {matplotlib.__version__}")
except ImportError as e:
    print(f" Matplotlib导入失败: {e}")

print("\n=== 测试完成 ===")
