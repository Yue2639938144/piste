"""
序列相似性聚类模块

该模块提供基于序列相似性的聚类和数据集划分功能，用于避免数据泄露。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.cluster import AgglomerativeClustering, KMeans
from Bio import pairwise2
from Bio.Align import substitution_matrices
import torch


def calculate_similarity_matrix(sequences: List[str], 
                                matrix: Optional[Dict[str, Dict[str, float]]] = None) -> np.ndarray:
    """
    计算序列之间的相似性矩阵
    
    Args:
        sequences: 氨基酸序列列表
        matrix: 替代矩阵，如BLOSUM62，默认为None，使用内置的BLOSUM62
        
    Returns:
        相似性矩阵，形状为(n_sequences, n_sequences)
    """
    n_sequences = len(sequences)
    similarity_matrix = np.zeros((n_sequences, n_sequences))
    
    # 使用BLOSUM62替代矩阵
    if matrix is None:
        matrix = substitution_matrices.load("BLOSUM62")
    
    # 计算每对序列之间的相似性
    for i in range(n_sequences):
        for j in range(i, n_sequences):
            # 对角线上的元素（自身相似性）
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
            
            # 计算序列对齐得分
            alignments = pairwise2.align.globalds(
                sequences[i], 
                sequences[j], 
                matrix, 
                -10, 
                -0.5, 
                one_alignment_only=True
            )
            
            if alignments:
                # 标准化得分
                alignment = alignments[0]
                score = alignment.score
                max_possible_score = max(
                    sum(matrix.get(aa, {}).get(aa, 0) for aa in sequences[i]),
                    sum(matrix.get(aa, {}).get(aa, 0) for aa in sequences[j])
                )
                normalized_score = score / max_possible_score if max_possible_score > 0 else 0
                
                # 填充相似性矩阵（对称）
                similarity_matrix[i, j] = normalized_score
                similarity_matrix[j, i] = normalized_score
    
    return similarity_matrix


def sequence_similarity_clustering(sequences: List[str], 
                                   n_clusters: int = 10, 
                                   method: str = 'hierarchical',
                                   similarity_threshold: float = 0.7) -> np.ndarray:
    """
    基于序列相似性进行聚类
    
    Args:
        sequences: 氨基酸序列列表
        n_clusters: 聚类数量
        method: 聚类方法，可选值为'hierarchical'或'kmeans'
        similarity_threshold: 相似性阈值，用于层次聚类
        
    Returns:
        聚类标签数组
    """
    # 计算相似性矩阵
    similarity_matrix = calculate_similarity_matrix(sequences)
    
    # 将相似性矩阵转换为距离矩阵
    distance_matrix = 1 - similarity_matrix
    
    # 执行聚类
    if method.lower() == 'hierarchical':
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters if n_clusters > 0 else None,
            affinity='precomputed',
            linkage='average',
            distance_threshold=1 - similarity_threshold if n_clusters is None else None
        )
        labels = clustering.fit_predict(distance_matrix)
    elif method.lower() == 'kmeans':
        # 对于k-means，我们需要将距离矩阵转换为特征空间
        # 这里使用多维缩放(MDS)进行降维
        from sklearn.manifold import MDS
        mds = MDS(n_components=min(50, len(sequences) - 1), dissimilarity='precomputed', random_state=42)
        features = mds.fit_transform(distance_matrix)
        
        # 执行k-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
    else:
        raise ValueError("方法必须是'hierarchical'或'kmeans'")
    
    return labels


def split_by_clusters(data_df: pd.DataFrame, 
                     clusters: List[Union[int, str]],
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    基于预先计算的聚类标签划分训练、验证和测试集
    
    Args:
        data_df: 包含序列和标签的DataFrame
        clusters: 聚类标签列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        训练集、验证集和测试集的DataFrame元组
    """
    # 将聚类标签添加到DataFrame
    data_df = data_df.copy()
    data_df['cluster'] = clusters
    
    # 获取唯一的聚类标签
    unique_clusters = list(set(clusters))
    n_clusters = len(unique_clusters)
    
    # 计算每个聚类的样本数
    cluster_counts = data_df['cluster'].value_counts().to_dict()
    
    # 计算训练、验证和测试集的聚类数量
    n_train_clusters = max(1, int(n_clusters * train_ratio))
    n_val_clusters = max(1, int(n_clusters * val_ratio))
    n_test_clusters = n_clusters - n_train_clusters - n_val_clusters
    
    # 确保至少有一个聚类用于每个集合
    if n_test_clusters < 1:
        n_test_clusters = 1
        if n_train_clusters + n_val_clusters + n_test_clusters > n_clusters:
            if n_val_clusters > 1:
                n_val_clusters -= 1
            else:
                n_train_clusters -= 1
    
    # 按聚类大小排序，确保大小平衡
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    
    # 分配聚类到训练、验证和测试集
    train_clusters = []
    val_clusters = []
    test_clusters = []
    
    # 使用贪心算法分配聚类，尽量使每个集合的样本数接近目标比例
    remaining_clusters = [c[0] for c in sorted_clusters]
    
    # 首先分配训练集
    while len(train_clusters) < n_train_clusters and remaining_clusters:
        train_clusters.append(remaining_clusters.pop(0))
    
    # 然后分配验证集
    while len(val_clusters) < n_val_clusters and remaining_clusters:
        val_clusters.append(remaining_clusters.pop(0))
    
    # 剩余的分配给测试集
    test_clusters.extend(remaining_clusters)
    
    # 划分数据集
    train_df = data_df[data_df['cluster'].isin(train_clusters)].copy()
    val_df = data_df[data_df['cluster'].isin(val_clusters)].copy()
    test_df = data_df[data_df['cluster'].isin(test_clusters)].copy()
    
    # 删除聚类列
    train_df.drop(columns=['cluster'], inplace=True)
    val_df.drop(columns=['cluster'], inplace=True)
    test_df.drop(columns=['cluster'], inplace=True)
    
    return train_df, val_df, test_df