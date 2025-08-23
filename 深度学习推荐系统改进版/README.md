# 基于深度学习的智能商户推荐系统改进版

## 项目概述

本项目是对原有"星图·言迹·群荐"系统的深度改进版本，引入了先进的深度学习技术和现代推荐算法，构建了一个更加智能和精准的商户推荐系统。

## 核心改进

### 1. 深度学习技术集成
- **BERT文本编码**：使用预训练的BERT模型进行评论文本深度理解
- **图神经网络(GNN)**：利用GraphSAGE和GAT进行社交网络建模
- **多模态融合**：结合文本、数值、图结构的多模态深度学习
- **注意力机制**：引入Transformer注意力机制提升特征重要性学习

### 2. 现代推荐算法
- **协同过滤**：基于用户的协同过滤(CF)和基于物品的协同过滤
- **矩阵分解**：SVD++、NMF等矩阵分解方法
- **深度推荐网络**：DeepFM、Wide&Deep等深度推荐模型
- **序列推荐**：基于用户行为序列的推荐

### 3. 系统架构优化
- **模块化设计**：数据加载、特征工程、模型训练、推荐生成分离
- **可扩展性**：支持新用户冷启动和新商户冷启动
- **实时性**：支持增量学习和实时推荐更新

## 技术架构

```
数据层 → 特征工程层 → 深度学习层 → 推荐算法层 → 结果输出层
   ↓           ↓           ↓           ↓           ↓
原始数据   多模态特征    BERT+GNN    混合推荐     个性化推荐
```

## 项目结构

```
深度学习推荐系统改进版/
├── data/                    # 数据文件
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后数据
│   └── embeddings/         # 预训练嵌入
├── src/                    # 源代码
│   ├── data_loader.py      # 数据加载模块
│   ├── feature_engineering.py  # 特征工程
│   ├── models/             # 模型定义
│   │   ├── bert_model.py   # BERT文本模型
│   │   ├── gnn_model.py    # 图神经网络模型
│   │   ├── recommendation.py # 推荐算法
│   │   └── fusion_model.py # 多模态融合模型
│   ├── training.py         # 模型训练
│   └── evaluation.py       # 模型评估
├── notebooks/              # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_recommendation.ipynb
├── configs/                # 配置文件
├── results/                # 结果输出
└── requirements.txt        # 依赖包
```

## 核心算法

### 1. 文本理解模块
- **BERT-Base-Chinese**：中文评论文本编码
- **TextCNN**：文本特征提取
- **BiLSTM+Attention**：序列建模

### 2. 图神经网络模块
- **GraphSAGE**：社交网络节点表示学习
- **GAT**：图注意力网络
- **GraphConv**：图卷积网络

### 3. 推荐算法模块
- **DeepFM**：深度因子分解机
- **Wide&Deep**：宽深模型
- **NCF**：神经协同过滤
- **LightGCN**：轻量图卷积网络

## 使用方法

### 环境配置
```bash
pip install -r requirements.txt
```

### 运行流程
1. **数据预处理**：`python src/data_loader.py`
2. **特征工程**：`python src/feature_engineering.py`
3. **模型训练**：`python src/training.py`
4. **推荐生成**：`python src/recommendation.py`

## 预期效果

- **覆盖率提升**：解决冷启动问题，提升推荐覆盖率
- **多样性提升**：通过多模态融合提升推荐多样性
- **可解释性**：通过注意力机制提供推荐解释

## 参考文献

1. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
2. Hamilton, W., et al. "Inductive Representation Learning on Large Graphs"
3. He, X., et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"
4. Cheng, H., et al. "Wide & Deep Learning for Recommender Systems"
5. He, X., et al. "Neural Collaborative Filtering"
6. He, X., et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
