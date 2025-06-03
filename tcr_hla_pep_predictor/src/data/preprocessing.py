"""
序列预处理模块

该模块提供氨基酸序列的预处理功能，包括序列清洗、索引化、生化属性编码和掩码生成。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
import torch


# 氨基酸字母表
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# 氨基酸到索引的映射
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
# 索引到氨基酸的映射
IDX_TO_AA = {idx: aa for idx, aa in enumerate(AMINO_ACIDS)}
# 填充标记索引
PAD_IDX = len(AMINO_ACIDS)
# 未知氨基酸索引
UNK_IDX = PAD_IDX + 1

# 氨基酸生化属性编码
# 属性包括：疏水性、极性、电荷、大小、芳香性等
BIOCHEM_PROPERTIES = {
    'A': [1.8, 0.0, 0.0, 0.87, 0.0],  # 丙氨酸
    'C': [2.5, 0.0, 0.0, 1.52, 0.0],  # 半胱氨酸
    'D': [-3.5, 1.0, -1.0, 1.43, 0.0],  # 天冬氨酸
    'E': [-3.5, 1.0, -1.0, 1.77, 0.0],  # 谷氨酸
    'F': [2.8, 0.0, 0.0, 2.08, 1.0],  # 苯丙氨酸
    'G': [-0.4, 0.0, 0.0, 0.60, 0.0],  # 甘氨酸
    'H': [-3.2, 1.0, 0.5, 1.78, 1.0],  # 组氨酸
    'I': [4.5, 0.0, 0.0, 1.56, 0.0],  # 异亮氨酸
    'K': [-3.9, 1.0, 1.0, 1.94, 0.0],  # 赖氨酸
    'L': [3.8, 0.0, 0.0, 1.67, 0.0],  # 亮氨酸
    'M': [1.9, 0.0, 0.0, 1.68, 0.0],  # 甲硫氨酸
    'N': [-3.5, 1.0, 0.0, 1.45, 0.0],  # 天冬酰胺
    'P': [-1.6, 0.0, 0.0, 1.29, 0.0],  # 脯氨酸
    'Q': [-3.5, 1.0, 0.0, 1.75, 0.0],  # 谷氨酰胺
    'R': [-4.5, 1.0, 1.0, 2.38, 0.0],  # 精氨酸
    'S': [-0.8, 1.0, 0.0, 1.13, 0.0],  # 丝氨酸
    'T': [-0.7, 1.0, 0.0, 1.40, 0.0],  # 苏氨酸
    'V': [4.2, 0.0, 0.0, 1.40, 0.0],  # 缬氨酸
    'W': [-0.9, 0.0, 0.0, 2.54, 1.0],  # 色氨酸
    'Y': [-1.3, 1.0, 0.0, 2.20, 1.0],  # 酪氨酸
}


def preprocess_sequences(sequences: List[str]) -> List[str]:
    """
    预处理氨基酸序列，包括去除非标准字符和统一大小写
    
    Args:
        sequences: 氨基酸序列列表
        
    Returns:
        处理后的氨基酸序列列表
    """
    processed_seqs = []
    for seq in sequences:
        # 转换为大写
        seq = seq.upper()
        # 过滤非标准氨基酸字符
        seq = ''.join([aa if aa in AMINO_ACIDS else 'X' for aa in seq])
        processed_seqs.append(seq)
    return processed_seqs


def encode_amino_acids(sequences: List[str], 
                       max_length: Optional[int] = None, 
                       padding: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将氨基酸序列编码为数值索引和生化特征
    
    Args:
        sequences: 预处理后的氨基酸序列列表
        max_length: 序列最大长度，如果为None则使用最长序列的长度
        padding: 是否进行填充
        
    Returns:
        索引编码的张量和生化特征编码的张量
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # 初始化索引编码和生化特征编码
    index_encoding = torch.full((len(sequences), max_length), PAD_IDX, dtype=torch.long)
    biochem_encoding = torch.zeros((len(sequences), max_length, len(next(iter(BIOCHEM_PROPERTIES.values())))))
    
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_length]):
            # 索引编码
            index_encoding[i, j] = AA_TO_IDX.get(aa, UNK_IDX)
            # 生化特征编码
            if aa in BIOCHEM_PROPERTIES:
                biochem_encoding[i, j] = torch.tensor(BIOCHEM_PROPERTIES[aa])
    
    return index_encoding, biochem_encoding


def generate_masks(sequences: List[str], max_length: Optional[int] = None) -> torch.Tensor:
    """
    为序列生成注意力掩码
    
    Args:
        sequences: 氨基酸序列列表
        max_length: 序列最大长度，如果为None则使用最长序列的长度
        
    Returns:
        注意力掩码张量，形状为(batch_size, max_length)，有效位置为1，填充位置为0
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    # 初始化掩码
    masks = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        # 有效位置标记为1
        masks[i, :len(seq)] = 1
    
    return masks 