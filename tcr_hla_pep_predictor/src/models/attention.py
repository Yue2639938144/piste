"""
注意力机制模块

该模块实现了物理滑动注意力、数据驱动注意力和融合注意力三种机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional, Any


class PhysicalSlidingAttention(nn.Module):
    """
    物理滑动注意力机制
    
    基于生物分子空间位置关系的注意力机制，通过模拟氨基酸残基在空间中的相对位置和互作可能性，
    计算注意力权重。
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 sigma: float = 1.0, 
                 num_iterations: int = 3):
        """
        初始化物理滑动注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            sigma: 高斯核函数的标准差，控制注意力"视野"范围
            num_iterations: 位置更新迭代次数
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.sigma = sigma
        self.num_iterations = num_iterations
        
        # 线性投影层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        # 位置编码学习参数
        self.position_scale = nn.Parameter(torch.ones(1))
        self.position_bias = nn.Parameter(torch.zeros(1))
        
    def initial_positions(self, 
                          interaction_type: str, 
                          query_lengths: torch.Tensor, 
                          key_lengths: torch.Tensor, 
                          max_query_len: int, 
                          max_key_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        初始化氨基酸残基的空间位置
        
        Args:
            interaction_type: 互作类型，'tcr_pep'、'hla_pep'或'trimer'
            query_lengths: 查询序列长度
            key_lengths: 键序列长度
            max_query_len: 最大查询序列长度
            max_key_len: 最大键序列长度
            
        Returns:
            查询和键的初始位置张量
        """
        batch_size = query_lengths.size(0)
        device = query_lengths.device
        
        if interaction_type == 'tcr_pep':
            # TCR-Pep互作的初始位置
            # TCR CDR3区域与肽段中部对齐
            q_pos = torch.zeros(batch_size, max_query_len).to(device)
            k_pos = torch.zeros(batch_size, max_key_len).to(device)
            
            for i in range(batch_size):
                q_len = query_lengths[i].item()
                k_len = key_lengths[i].item()
                
                # TCR CDR3位置：居中分布
                q_pos[i, :q_len] = torch.linspace(-q_len/2, q_len/2, q_len)
                
                # 肽段位置：居中分布
                k_pos[i, :k_len] = torch.linspace(-k_len/2, k_len/2, k_len)
        
        elif interaction_type == 'hla_pep':
            # HLA-Pep互作的初始位置
            # 肽段位于HLA结合槽中
            q_pos = torch.zeros(batch_size, max_query_len).to(device)
            k_pos = torch.zeros(batch_size, max_key_len).to(device)
            
            for i in range(batch_size):
                q_len = query_lengths[i].item()
                k_len = key_lengths[i].item()
                
                # HLA位置：形成结合槽形状
                hla_pos = torch.zeros(q_len)
                
                # 结合槽的α螺旋1（约占1/3长度）
                helix1_len = max(q_len // 3, 1)
                hla_pos[:helix1_len] = torch.linspace(0, helix1_len, helix1_len)
                
                # 结合槽的β片层（约占1/3长度）
                sheet_len = max(q_len // 3, 1)
                hla_pos[helix1_len:helix1_len+sheet_len] = torch.linspace(helix1_len, 0, sheet_len)
                
                # 结合槽的α螺旋2（约占1/3长度）
                helix2_len = q_len - helix1_len - sheet_len
                hla_pos[helix1_len+sheet_len:] = torch.linspace(0, helix2_len, helix2_len)
                
                q_pos[i, :q_len] = hla_pos
                
                # 肽段位置：线性排列在结合槽中央
                k_pos[i, :k_len] = torch.linspace(helix1_len/2, sheet_len, k_len)
        
        elif interaction_type == 'trimer':
            # 三元互作的初始位置
            # 简化模型：HLA-Pep作为一个整体，与TCR互作
            q_pos = torch.zeros(batch_size, max_query_len).to(device)  # TCR
            k_pos = torch.zeros(batch_size, max_key_len).to(device)    # HLA-Pep
            
            for i in range(batch_size):
                q_len = query_lengths[i].item()
                k_len = key_lengths[i].item()
                
                # TCR位置：半圆形分布
                angles = torch.linspace(math.pi/4, 3*math.pi/4, q_len)
                q_pos[i, :q_len] = torch.sin(angles) * 5
                
                # HLA-Pep位置：直线分布
                k_pos[i, :k_len] = torch.linspace(-k_len/2, k_len/2, k_len)
        
        else:
            raise ValueError(f"不支持的互作类型: {interaction_type}")
        
        # 应用缩放和偏置
        q_pos = q_pos * self.position_scale + self.position_bias
        k_pos = k_pos * self.position_scale + self.position_bias
        
        # 将无效位置（填充位置）设为一个大值
        q_mask = torch.arange(max_query_len).unsqueeze(0).expand(batch_size, -1).to(device)
        q_mask = q_mask >= query_lengths.unsqueeze(1)
        q_pos.masked_fill_(q_mask, 1e4)
        
        k_mask = torch.arange(max_key_len).unsqueeze(0).expand(batch_size, -1).to(device)
        k_mask = k_mask >= key_lengths.unsqueeze(1)
        k_pos.masked_fill_(k_mask, 1e4)
        
        return q_pos, k_pos
    
    def calculate_distance_attention(self, 
                                    q_pos: torch.Tensor, 
                                    k_pos: torch.Tensor) -> torch.Tensor:
        """
        计算基于空间距离的注意力权重
        
        Args:
            q_pos: 查询位置，形状为(batch_size, query_len)
            k_pos: 键位置，形状为(batch_size, key_len)
            
        Returns:
            距离注意力权重，形状为(batch_size, query_len, key_len)
        """
        batch_size = q_pos.size(0)
        query_len = q_pos.size(1)
        key_len = k_pos.size(1)
        
        # 计算欧氏距离的平方
        q_pos_expanded = q_pos.unsqueeze(2).expand(-1, -1, key_len)
        k_pos_expanded = k_pos.unsqueeze(1).expand(-1, query_len, -1)
        distances = (q_pos_expanded - k_pos_expanded).pow(2)
        
        # 使用高斯核函数计算注意力权重
        attention = torch.exp(-distances / (2 * self.sigma * self.sigma))
        
        # 将无效位置（填充位置）的注意力权重设为0
        mask = (q_pos_expanded >= 1e3) | (k_pos_expanded >= 1e3)
        attention.masked_fill_(mask, 0.0)
        
        return attention
    
    def update_positions(self, 
                        q_pos: torch.Tensor, 
                        k_pos: torch.Tensor, 
                        attention: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于注意力权重更新位置
        
        Args:
            q_pos: 查询位置，形状为(batch_size, query_len)
            k_pos: 键位置，形状为(batch_size, key_len)
            attention: 注意力权重，形状为(batch_size, query_len, key_len)
            
        Returns:
            更新后的查询和键位置
        """
        batch_size = q_pos.size(0)
        query_len = q_pos.size(1)
        key_len = k_pos.size(1)
        
        # 计算注意力权重归一化
        attn_sum = attention.sum(dim=2, keepdim=True).clamp(min=1e-9)
        norm_attention = attention / attn_sum
        
        # 更新查询位置：向关注的键位置移动
        q_pos_new = torch.bmm(norm_attention, k_pos.unsqueeze(2)).squeeze(2)
        
        # 计算键到查询的注意力（转置）
        k2q_attention = attention.transpose(1, 2)
        k_attn_sum = k2q_attention.sum(dim=2, keepdim=True).clamp(min=1e-9)
        k2q_norm_attention = k2q_attention / k_attn_sum
        
        # 更新键位置：向关注它的查询位置移动
        k_pos_new = torch.bmm(k2q_norm_attention, q_pos.unsqueeze(2)).squeeze(2)
        
        # 保持原始的无效位置标记
        q_mask = q_pos >= 1e3
        k_mask = k_pos >= 1e3
        q_pos_new.masked_fill_(q_mask, 1e4)
        k_pos_new.masked_fill_(k_mask, 1e4)
        
        return q_pos_new, k_pos_new
    
    def forward(self, 
               query: torch.Tensor, 
               key: torch.Tensor, 
               value: torch.Tensor, 
               query_lengths: torch.Tensor, 
               key_lengths: torch.Tensor, 
               interaction_type: str,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量，形状为(batch_size, query_len, d_model)
            key: 键张量，形状为(batch_size, key_len, d_model)
            value: 值张量，形状为(batch_size, key_len, d_model)
            query_lengths: 查询序列实际长度
            key_lengths: 键序列实际长度
            interaction_type: 互作类型，'tcr_pep'、'hla_pep'或'trimer'
            mask: 注意力掩码，形状为(batch_size, query_len, key_len)
            
        Returns:
            输出张量和注意力权重
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        # 线性变换
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # 多头分割
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 初始化位置
        q_pos, k_pos = self.initial_positions(
            interaction_type, 
            query_lengths, 
            key_lengths, 
            query_len, 
            key_len
        )
        
        # 计算内积注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        
        # 初始注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        
        # 迭代更新位置和注意力
        for _ in range(self.num_iterations):
            # 计算基于距离的注意力
            distance_attn = self.calculate_distance_attention(q_pos, k_pos)
            
            # 将距离注意力扩展为多头形式
            distance_attn = distance_attn.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            
            # 融合内积注意力和距离注意力
            combined_attn = attn_weights * distance_attn
            
            # 归一化
            combined_attn = combined_attn / (combined_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9))
            
            # 更新位置
            q_pos, k_pos = self.update_positions(q_pos, k_pos, distance_attn[:, 0])
            
            # 更新注意力权重
            attn_weights = combined_attn
        
        # 应用注意力
        context = torch.matmul(attn_weights, v)
        
        # 重塑和连接多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性层
        output = self.out(context)
        
        # 返回输出和最终注意力权重（取平均）
        return output, attn_weights.mean(dim=1)


class DataDrivenAttention(nn.Module):
    """
    数据驱动注意力机制
    
    基于Transformer的多头自注意力机制，从数据中学习序列间的复杂关系。
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 dropout: float = 0.1):
        """
        初始化数据驱动注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性投影层
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
               query: torch.Tensor, 
               key: torch.Tensor, 
               value: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询张量，形状为(batch_size, query_len, d_model)
            key: 键张量，形状为(batch_size, key_len, d_model)
            value: 值张量，形状为(batch_size, key_len, d_model)
            mask: 注意力掩码，形状为(batch_size, query_len, key_len)
            
        Returns:
            输出张量和注意力权重
        """
        batch_size = query.size(0)
        
        # 线性变换
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
        
        # 多头分割
        q = q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
        
        # 注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        context = torch.matmul(attn_weights, v)
        
        # 重塑和连接多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性层
        output = self.out(context)
        
        # 返回输出和注意力权重（取平均）
        return output, attn_weights.mean(dim=1)


class FusedAttention(nn.Module):
    """
    融合注意力机制
    
    结合物理滑动注意力和数据驱动注意力的融合机制。
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 sigma: float = 1.0, 
                 num_iterations: int = 3, 
                 dropout: float = 0.1,
                 fusion_method: str = "weighted_sum",
                 fusion_weights: Optional[List[float]] = None):
        """
        初始化融合注意力
        
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            sigma: 高斯核函数的标准差
            num_iterations: 位置更新迭代次数
            dropout: Dropout比率
            fusion_method: 融合方法，可选值为"weighted_sum"、"concat"或"gated"
            fusion_weights: 融合权重，用于weighted_sum方法
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.fusion_method = fusion_method.lower()
        
        # 初始化物理滑动注意力和数据驱动注意力
        self.physical_attention = PhysicalSlidingAttention(
            d_model, n_heads, sigma, num_iterations
        )
        self.data_attention = DataDrivenAttention(
            d_model, n_heads, dropout
        )
        
        # 融合方法特定参数
        if fusion_method == "weighted_sum":
            if fusion_weights is None:
                fusion_weights = [0.5, 0.5]
            self.register_buffer("fusion_weights", torch.tensor(fusion_weights))
        elif fusion_method == "concat":
            self.fusion_layer = nn.Linear(d_model * 2, d_model)
        elif fusion_method == "gated":
            self.gate = nn.Linear(d_model * 2, d_model)
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")
    
    def forward(self, 
               query: torch.Tensor, 
               key: torch.Tensor, 
               value: torch.Tensor, 
               query_lengths: torch.Tensor, 
               key_lengths: torch.Tensor, 
               interaction_type: str,
               mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            query: 查询张量，形状为(batch_size, query_len, d_model)
            key: 键张量，形状为(batch_size, key_len, d_model)
            value: 值张量，形状为(batch_size, key_len, d_model)
            query_lengths: 查询序列实际长度
            key_lengths: 键序列实际长度
            interaction_type: 互作类型，'tcr_pep'、'hla_pep'或'trimer'
            mask: 注意力掩码，形状为(batch_size, query_len, key_len)
            
        Returns:
            输出张量和包含各种注意力权重的字典
        """
        # 物理滑动注意力
        physical_output, physical_attn = self.physical_attention(
            query, key, value, query_lengths, key_lengths, interaction_type, mask
        )
        
        # 数据驱动注意力
        data_output, data_attn = self.data_attention(
            query, key, value, mask
        )
        
        # 融合输出
        if self.fusion_method == "weighted_sum":
            output = self.fusion_weights[0] * physical_output + self.fusion_weights[1] * data_output
        elif self.fusion_method == "concat":
            output = self.fusion_layer(torch.cat([physical_output, data_output], dim=-1))
        elif self.fusion_method == "gated":
            gate_input = torch.cat([physical_output, data_output], dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            output = gate * physical_output + (1 - gate) * data_output
        
        # 返回输出和注意力权重字典
        attention_weights = {
            "physical": physical_attn,
            "data_driven": data_attn,
            "fused": 0.5 * physical_attn + 0.5 * data_attn  # 简单平均作为融合注意力
        }
        
        return output, attention_weights 