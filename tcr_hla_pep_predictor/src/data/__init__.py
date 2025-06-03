"""
数据处理模块

该模块负责TCR-HLA-Pep数据的加载、预处理、序列相似性聚类和数据集划分。
"""

from .preprocessing import preprocess_sequences, encode_amino_acids, generate_masks
from .dataloader import DataLoader, MyDataSet
from .clustering import sequence_similarity_clustering, split_by_clusters

__all__ = [
    'preprocess_sequences', 
    'encode_amino_acids', 
    'generate_masks',
    'DataLoader', 
    'MyDataSet',
    'sequence_similarity_clustering', 
    'split_by_clusters'
] 