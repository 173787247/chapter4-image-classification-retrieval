# 第四章作业：图像分类与检索系统

## 项目概述

本项目实现了两个核心功能：
1. **ResNet二分类器** - 使用预训练ResNet模型进行图像二分类
2. **CLIP图像检索系统** - 基于CLIP模型的以图搜图功能

## 项目结构

```
chapter4-image-classification-retrieval/
 README.md                    # 项目说明文档
 requirements.txt             # Python依赖包
 src/                        # 源代码目录
    resnet_classifier.py    # ResNet二分类器
    clip_retrieval.py       # CLIP图像检索系统
    data_loader.py          # 数据加载器
    utils.py                # 工具函数
    main.py                 # 主程序
 data/                       # 数据目录
    positive/               # 正类图像
    negative/               # 负类图像
    query/                  # 查询图像
 models/                     # 模型保存目录
 results/                    # 结果输出目录
 notebooks/                  # Jupyter笔记本
     01_resnet_classifier.ipynb
     02_clip_retrieval.ipynb
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
- 在 `data/positive/` 中放入正类图像
- 在 `data/negative/` 中放入负类图像
- 在 `data/query/` 中放入查询图像

### 3. 运行Jupyter笔记本
```bash
jupyter notebook notebooks/
```

## 功能详情

### 练习1：ResNet二分类器
- **模型**：预训练ResNet-50
- **任务**：二分类（正类/负类）
- **训练**：支持微调或冻结主干网络
- **输出**：0/1分类结果

### 练习2：CLIP图像检索系统
- **模型**：OpenAI CLIP
- **功能**：以图搜图
- **检索**：基于余弦相似度
- **可视化**：查询图 + Top-K结果

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

MIT License
