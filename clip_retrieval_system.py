#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIP图像检索系统 - 简化版
"""

import os
import torch
import open_clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class CLIPRetrievalSystem:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载CLIP模型
        print("正在加载CLIP模型...")
        self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        print(" CLIP模型加载完成！")
        
        # 图像特征向量库
        self.image_embeddings = {}
        self.image_paths = []
    
    def extract_image_embedding(self, image_path):
        """提取图像特征向量"""
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"特征提取失败 {image_path}: {e}")
            return None
    
    def build_index(self, image_dir):
        """构建图像索引"""
        print(f"构建索引: {image_dir}")
        
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(root, file)
                    print(f"处理: {file}")
                    
                    embedding = self.extract_image_embedding(image_path)
                    if embedding is not None:
                        self.image_embeddings[image_path] = embedding
                        self.image_paths.append(image_path)
        
        print(f"索引构建完成！共 {len(self.image_paths)} 张图像")
    
    def search_similar(self, query_image_path, top_k=5):
        """搜索相似图像"""
        if not self.image_embeddings:
            print("索引库为空！")
            return []
        
        query_embedding = self.extract_image_embedding(query_image_path)
        if query_embedding is None:
            return []
        
        similarities = []
        for image_path, embedding in self.image_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1), 
                embedding.reshape(1, -1)
            )[0][0]
            similarities.append((image_path, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def visualize_results(self, query_path, results):
        """可视化搜索结果"""
        if not results:
            return
        
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 3))
        
        # 查询图像
        query_img = Image.open(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title("查询图像")
        axes[0].axis("off")
        
        # 搜索结果
        for i, (result_path, similarity) in enumerate(results):
            result_img = Image.open(result_path)
            axes[i + 1].imshow(result_img)
            axes[i + 1].set_title(f"相似度: {similarity:.3f}")
            axes[i + 1].axis("off")
        
        plt.tight_layout()
        
        # 保存结果
        os.makedirs("results", exist_ok=True)
        save_path = f"results/clip_retrieval_{os.path.basename(query_path)}"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"结果已保存: {save_path}")
        
        plt.show()

def main():
    print("=== CLIP图像检索系统 ===")
    
    # 创建检索系统
    retrieval_system = CLIPRetrievalSystem()
    
    # 构建索引
    print("\n1. 构建图像索引...")
    retrieval_system.build_index("data")
    
    # 测试检索
    print("\n2. 测试图像检索...")
    query_dir = "data/query"
    
    if os.path.exists(query_dir):
        query_images = [f for f in os.listdir(query_dir) if f.endswith(".jpg")]
        
        for query_img in query_images[:2]:  # 测试前2张
            query_path = os.path.join(query_dir, query_img)
            print(f"\n查询图像: {query_img}")
            
            # 搜索相似图像
            results = retrieval_system.search_similar(query_path, top_k=5)
            
            if results:
                print("Top-5 相似图像:")
                for i, (path, similarity) in enumerate(results):
                    name = os.path.basename(path)
                    print(f"  {i+1}. {name} (相似度: {similarity:.3f})")
                
                # 可视化
                retrieval_system.visualize_results(query_path, results)
    
    print("\n CLIP检索系统测试完成！")

if __name__ == "__main__":
    main()
