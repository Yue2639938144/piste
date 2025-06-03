"""
TCR-Pep互作预测模型

该模块实现了TCR-Pep二元互作预测模型，用于预测TCR与抗原肽段的结合能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Union, Optional, Any

from .attention import PhysicalSlidingAttention, DataDrivenAttention, FusedAttention


class SequenceEmbedding(nn.Module):
    """
    序列嵌入模块
    
    将氨基酸序列转换为嵌入向量，支持索引嵌入和生化特征嵌入的融合。
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 padding_idx: int = 20, 
                 use_biochem: bool = True,
                 biochem_dim: int = 5,
                 dropout: float = 0.1):
        """
        初始化序列嵌入模块
        
        Args:
            vocab_size: 词汇表大小（氨基酸种类数）
            embedding_dim: 嵌入维度
            padding_idx: 填充标记的索引
            use_biochem: 是否使用生化特征
            biochem_dim: 生化特征维度
            dropout: Dropout比率
        """
        super().__init__()
        
        self.use_biochem = use_biochem
        
        # 氨基酸索引嵌入
        self.aa_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=padding_idx
        )
        
        # 生化特征投影
        if use_biochem:
            self.biochem_proj = nn.Linear(biochem_dim, embedding_dim)
            self.fusion_layer = nn.Linear(embedding_dim * 2, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, 
               aa_indices: torch.Tensor, 
               biochem_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            aa_indices: 氨基酸索引，形状为(batch_size, seq_len)
            biochem_features: 生化特征，形状为(batch_size, seq_len, biochem_dim)
            
        Returns:
            序列嵌入，形状为(batch_size, seq_len, embedding_dim)
        """
        # 氨基酸索引嵌入
        aa_embed = self.aa_embedding(aa_indices)
        
        # 融合生化特征
        if self.use_biochem and biochem_features is not None:
            biochem_embed = self.biochem_proj(biochem_features)
            embed = self.fusion_layer(torch.cat([aa_embed, biochem_embed], dim=-1))
        else:
            embed = aa_embed
        
        # 应用Dropout和层归一化
        embed = self.dropout(embed)
        embed = self.layer_norm(embed)
        
        return embed


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    为序列添加位置信息，使模型能够感知序列中的位置关系。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)
            
        Returns:
            添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """
    编码器层
    
    包含多头注意力和前馈神经网络的Transformer编码器层。
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 d_ff: int = 2048, 
                 dropout: float = 0.1):
        """
        初始化编码器层
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: 前馈神经网络维度
            dropout: Dropout比率
        """
        super().__init__()
        
        # 自注意力层
        self.self_attn = DataDrivenAttention(d_model, n_heads, dropout)
        
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, 
               x: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为(batch_size, seq_len, d_model)
            mask: 注意力掩码
            
        Returns:
            编码器层输出
        """
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈神经网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


class TCRPepModel(nn.Module):
    """
    TCR-Pep互作预测模型
    
    预测TCR与抗原肽段的结合能力。
    """
    
    def __init__(self, 
                 vocab_size: int = 22, 
                 embedding_dim: int = 128, 
                 hidden_dim: int = 256, 
                 num_heads: int = 8, 
                 num_layers: int = 4, 
                 max_tcr_len: int = 30, 
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
        初始化TCR-Pep模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            max_tcr_len: 最大TCR序列长度
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
        self.max_tcr_len = max_tcr_len
        self.max_pep_len = max_pep_len
        self.attention_type = attention_type.lower()
        
        # 序列嵌入
        self.tcr_embedding = SequenceEmbedding(
            vocab_size, embedding_dim, use_biochem=use_biochem, 
            biochem_dim=biochem_dim, dropout=dropout
        )
        self.pep_embedding = SequenceEmbedding(
            vocab_size, embedding_dim, use_biochem=use_biochem, 
            biochem_dim=biochem_dim, dropout=dropout
        )
        
        # 位置编码
        self.tcr_pos_encoding = PositionalEncoding(embedding_dim, max_tcr_len, dropout)
        self.pep_pos_encoding = PositionalEncoding(embedding_dim, max_pep_len, dropout)
        
        # TCR编码器
        self.tcr_encoder = nn.ModuleList([
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
               tcr_idx: torch.Tensor, 
               pep_idx: torch.Tensor, 
               tcr_biochem: Optional[torch.Tensor] = None, 
               pep_biochem: Optional[torch.Tensor] = None,
               tcr_mask: Optional[torch.Tensor] = None, 
               pep_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            tcr_idx: TCR序列索引，形状为(batch_size, tcr_len)
            pep_idx: 肽段序列索引，形状为(batch_size, pep_len)
            tcr_biochem: TCR生化特征，形状为(batch_size, tcr_len, biochem_dim)
            pep_biochem: 肽段生化特征，形状为(batch_size, pep_len, biochem_dim)
            tcr_mask: TCR序列掩码，形状为(batch_size, tcr_len)
            pep_mask: 肽段序列掩码，形状为(batch_size, pep_len)
            
        Returns:
            包含预测结果和注意力权重的字典
        """
        batch_size = tcr_idx.size(0)
        
        # 创建注意力掩码
        if tcr_mask is None:
            tcr_mask = torch.ones_like(tcr_idx, dtype=torch.bool)
        if pep_mask is None:
            pep_mask = torch.ones_like(pep_idx, dtype=torch.bool)
        
        # 计算序列长度
        tcr_lengths = tcr_mask.sum(dim=1)
        pep_lengths = pep_mask.sum(dim=1)
        
        # 创建自注意力掩码
        tcr_attn_mask = ~tcr_mask.unsqueeze(1).expand(-1, tcr_idx.size(1), -1)
        pep_attn_mask = ~pep_mask.unsqueeze(1).expand(-1, pep_idx.size(1), -1)
        
        # 创建交叉注意力掩码
        cross_attn_mask = ~tcr_mask.unsqueeze(2).expand(-1, -1, pep_idx.size(1))
        
        # 序列嵌入
        tcr_embed = self.tcr_embedding(tcr_idx, tcr_biochem)
        pep_embed = self.pep_embedding(pep_idx, pep_biochem)
        
        # 位置编码
        tcr_embed = self.tcr_pos_encoding(tcr_embed)
        pep_embed = self.pep_pos_encoding(pep_embed)
        
        # TCR编码
        for layer in self.tcr_encoder:
            tcr_embed = layer(tcr_embed, tcr_attn_mask)
        
        # 肽段编码
        for layer in self.pep_encoder:
            pep_embed = layer(pep_embed, pep_attn_mask)
        
        # 交叉注意力
        if self.attention_type == "physical" or self.attention_type == "fused":
            # 物理滑动注意力或融合注意力需要序列长度
            cross_output, cross_attn = self.cross_attention(
                tcr_embed, pep_embed, pep_embed, 
                tcr_lengths, pep_lengths, 
                'tcr_pep', cross_attn_mask
            )
        else:
            # 数据驱动注意力不需要序列长度
            cross_output, cross_attn = self.cross_attention(
                tcr_embed, pep_embed, pep_embed, cross_attn_mask
            )
        
        # 全局表示
        tcr_global = tcr_embed.sum(dim=1) / tcr_lengths.unsqueeze(1).float()
        cross_global = cross_output.sum(dim=1) / tcr_lengths.unsqueeze(1).float()
        
        # 连接全局表示
        combined = torch.cat([tcr_global, cross_global], dim=1)
        
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