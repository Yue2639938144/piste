"""
HLA-Pep互作预测模型

该模块实现了HLA-Pep二元互作预测模型，用于预测HLA与抗原肽段的结合能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Union, Optional, Any

from .attention import PhysicalSlidingAttention, DataDrivenAttention, FusedAttention
from .tcr_pep_model import SequenceEmbedding, PositionalEncoding, EncoderLayer


class HLAPepModel(nn.Module):
    """
    HLA-Pep互作预测模型
    
    预测HLA与抗原肽段的结合能力。
    """
    
    def __init__(self, 
                 vocab_size: int = 22, 
                 embedding_dim: int = 128, 
                 hidden_dim: int = 256, 
                 num_heads: int = 8, 
                 num_layers: int = 4, 
                 max_hla_len: int = 34, 
                 max_pep_len: int = 15, 
                 use_biochem: bool = True, 
                 biochem_dim: int = 5,
                 dropout: float = 0.1,
                 attention_type: str = "fused",
                 sigma: float = 1.0,
                 num_iterations: int = 3,
                 fusion_method: str = "weighted_sum",
                 fusion_weights: Optional[List[float]] = None):
        """
        初始化HLA-Pep模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            max_hla_len: 最大HLA序列长度
            max_pep_len: 最大肽段序列长度
            use_biochem: 是否使用生化特征
            biochem_dim: 生化特征维度
            dropout: Dropout比率
            attention_type: 注意力类型，可选值为"physical"、"data_driven"或"fused"
            sigma: 高斯核函数的标准差
            num_iterations: 位置更新迭代次数
            fusion_method: 融合方法，可选值为"weighted_sum"、"concat"或"gated"
            fusion_weights: 融合权重
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_hla_len = max_hla_len
        self.max_pep_len = max_pep_len
        self.attention_type = attention_type.lower()
        
        # 序列嵌入
        self.hla_embedding = SequenceEmbedding(
            vocab_size, embedding_dim, use_biochem=use_biochem, 
            biochem_dim=biochem_dim, dropout=dropout
        )
        self.pep_embedding = SequenceEmbedding(
            vocab_size, embedding_dim, use_biochem=use_biochem, 
            biochem_dim=biochem_dim, dropout=dropout
        )
        
        # 位置编码
        self.hla_pos_encoding = PositionalEncoding(embedding_dim, max_hla_len, dropout)
        self.pep_pos_encoding = PositionalEncoding(embedding_dim, max_pep_len, dropout)
        
        # HLA编码器
        self.hla_encoder = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 肽段编码器
        self.pep_encoder = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # 交叉注意力层
        if attention_type == "physical":
            self.cross_attention = PhysicalSlidingAttention(
                embedding_dim, num_heads, sigma, num_iterations
            )
        elif attention_type == "data_driven":
            self.cross_attention = DataDrivenAttention(
                embedding_dim, num_heads, dropout
            )
        elif attention_type == "fused":
            self.cross_attention = FusedAttention(
                embedding_dim, num_heads, sigma, num_iterations, dropout,
                fusion_method, fusion_weights
            )
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
               hla_idx: torch.Tensor, 
               pep_idx: torch.Tensor, 
               hla_biochem: Optional[torch.Tensor] = None, 
               pep_biochem: Optional[torch.Tensor] = None,
               hla_mask: Optional[torch.Tensor] = None, 
               pep_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            hla_idx: HLA序列索引，形状为(batch_size, hla_len)
            pep_idx: 肽段序列索引，形状为(batch_size, pep_len)
            hla_biochem: HLA生化特征，形状为(batch_size, hla_len, biochem_dim)
            pep_biochem: 肽段生化特征，形状为(batch_size, pep_len, biochem_dim)
            hla_mask: HLA序列掩码，形状为(batch_size, hla_len)
            pep_mask: 肽段序列掩码，形状为(batch_size, pep_len)
            
        Returns:
            包含预测结果和注意力权重的字典
        """
        batch_size = hla_idx.size(0)
        
        # 创建注意力掩码
        if hla_mask is None:
            hla_mask = torch.ones_like(hla_idx, dtype=torch.bool)
        if pep_mask is None:
            pep_mask = torch.ones_like(pep_idx, dtype=torch.bool)
        
        # 计算序列长度
        hla_lengths = hla_mask.sum(dim=1)
        pep_lengths = pep_mask.sum(dim=1)
        
        # 创建自注意力掩码
        hla_attn_mask = ~hla_mask.unsqueeze(1).expand(-1, hla_idx.size(1), -1)
        pep_attn_mask = ~pep_mask.unsqueeze(1).expand(-1, pep_idx.size(1), -1)
        
        # 创建交叉注意力掩码
        cross_attn_mask = ~hla_mask.unsqueeze(2).expand(-1, -1, pep_idx.size(1))
        
        # 序列嵌入
        hla_embed = self.hla_embedding(hla_idx, hla_biochem)
        pep_embed = self.pep_embedding(pep_idx, pep_biochem)
        
        # 位置编码
        hla_embed = self.hla_pos_encoding(hla_embed)
        pep_embed = self.pep_pos_encoding(pep_embed)
        
        # HLA编码
        for layer in self.hla_encoder:
            hla_embed = layer(hla_embed, hla_attn_mask)
        
        # 肽段编码
        for layer in self.pep_encoder:
            pep_embed = layer(pep_embed, pep_attn_mask)
        
        # 交叉注意力
        if self.attention_type == "physical" or self.attention_type == "fused":
            # 物理滑动注意力或融合注意力需要序列长度
            cross_output, cross_attn = self.cross_attention(
                hla_embed, pep_embed, pep_embed, 
                hla_lengths, pep_lengths, 
                'hla_pep', cross_attn_mask
            )
        else:
            # 数据驱动注意力不需要序列长度
            cross_output, cross_attn = self.cross_attention(
                hla_embed, pep_embed, pep_embed, cross_attn_mask
            )
        
        # 全局表示
        hla_global = hla_embed.sum(dim=1) / (hla_lengths.unsqueeze(1).float() + 1e-8)
        cross_global = cross_output.sum(dim=1) / (hla_lengths.unsqueeze(1).float() + 1e-8)
        
        # 连接全局表示
        combined = torch.cat([hla_global, cross_global], dim=1)
        
        # 预测结果
        pred = self.output_layer(combined).squeeze(-1)
        
        # 返回结果和注意力权重
        if isinstance(cross_attn, dict):
            return {
                "pred": pred,
                "attn_weights": cross_attn
            }
        else:
            return {
                "pred": pred,
                "attn_weights": {"attention": cross_attn}
            } 