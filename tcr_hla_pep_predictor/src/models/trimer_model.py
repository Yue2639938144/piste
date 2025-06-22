"""
TCR-HLA-Pep三元互作预测模型

该模块实现了TCR-HLA-Pep三元互作预测模型，整合TCR-Pep和HLA-Pep二元模型，实现三元互作预测功能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Union, Optional, Any

from .tcr_pep_model import TCRPepModel, SequenceEmbedding, PositionalEncoding, EncoderLayer
from .hla_pep_model import HLAPepModel
from .attention import PhysicalSlidingAttention, DataDrivenAttention, FusedAttention


class TrimerModel(nn.Module):
    """
    TCR-HLA-Pep三元互作预测模型
    
    整合TCR-Pep和HLA-Pep二元模型，实现TCR、HLA和肽段的三元互作预测。
    """
    
    def __init__(self, 
                 vocab_size: int = 22, 
                 embedding_dim: int = 128, 
                 hidden_dim: int = 256, 
                 num_heads: int = 8, 
                 num_layers: int = 4, 
                 max_tcr_len: int = 30, 
                 max_hla_len: int = 34, 
                 max_pep_len: int = 15, 
                 use_biochem: bool = True, 
                 biochem_dim: int = 5,
                 dropout: float = 0.1,
                 attention_type: str = "fused",
                 sigma: float = 1.0,
                 num_iterations: int = 3,
                 fusion_method: str = "weighted_sum",
                 fusion_weights: Optional[List[float]] = None,
                 tcr_pep_model: Optional[TCRPepModel] = None,
                 hla_pep_model: Optional[HLAPepModel] = None,
                 joint_training: bool = True):
        """
        初始化三元模型
        
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            max_tcr_len: 最大TCR序列长度
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
            tcr_pep_model: 预训练的TCR-Pep模型，如果为None则创建新模型
            hla_pep_model: 预训练的HLA-Pep模型，如果为None则创建新模型
            joint_training: 是否进行联合训练
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_tcr_len = max_tcr_len
        self.max_hla_len = max_hla_len
        self.max_pep_len = max_pep_len
        self.attention_type = attention_type.lower()
        self.joint_training = joint_training
        
        # 初始化或加载TCR-Pep模型
        if tcr_pep_model is None:
            self.tcr_pep_model = TCRPepModel(
                vocab_size, embedding_dim, hidden_dim, num_heads, num_layers,
                max_tcr_len, max_pep_len, use_biochem, biochem_dim, dropout,
                attention_type, sigma, num_iterations, fusion_method, fusion_weights
            )
        else:
            self.tcr_pep_model = tcr_pep_model
        
        # 初始化或加载HLA-Pep模型
        if hla_pep_model is None:
            self.hla_pep_model = HLAPepModel(
                vocab_size, embedding_dim, hidden_dim, num_heads, num_layers,
                max_hla_len, max_pep_len, use_biochem, biochem_dim, dropout,
                attention_type, sigma, num_iterations, fusion_method, fusion_weights
            )
        else:
            self.hla_pep_model = hla_pep_model
        
        # TCR-HLA交叉注意力层
        if attention_type == "physical":
            self.tcr_hla_attention = PhysicalSlidingAttention(
                embedding_dim, num_heads, sigma, num_iterations
            )
        elif attention_type == "data_driven":
            self.tcr_hla_attention = DataDrivenAttention(
                embedding_dim, num_heads, dropout
            )
        elif attention_type == "fused":
            self.tcr_hla_attention = FusedAttention(
                embedding_dim, num_heads, sigma, num_iterations, dropout,
                fusion_method, fusion_weights
            )
        else:
            raise ValueError(f"不支持的注意力类型: {attention_type}")
        
        # 三元互作整合层
        self.trimer_integration = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
               tcr_idx: torch.Tensor, 
               pep_idx: torch.Tensor, 
               hla_idx: torch.Tensor,
               tcr_biochem: Optional[torch.Tensor] = None, 
               pep_biochem: Optional[torch.Tensor] = None,
               hla_biochem: Optional[torch.Tensor] = None,
               tcr_mask: Optional[torch.Tensor] = None, 
               pep_mask: Optional[torch.Tensor] = None,
               hla_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            tcr_idx: TCR序列索引，形状为(batch_size, tcr_len)
            pep_idx: 肽段序列索引，形状为(batch_size, pep_len)
            hla_idx: HLA序列索引，形状为(batch_size, hla_len)
            tcr_biochem: TCR生化特征，形状为(batch_size, tcr_len, biochem_dim)
            pep_biochem: 肽段生化特征，形状为(batch_size, pep_len, biochem_dim)
            hla_biochem: HLA生化特征，形状为(batch_size, hla_len, biochem_dim)
            tcr_mask: TCR序列掩码，形状为(batch_size, tcr_len)
            pep_mask: 肽段序列掩码，形状为(batch_size, pep_len)
            hla_mask: HLA序列掩码，形状为(batch_size, hla_len)
            
        Returns:
            包含预测结果和注意力权重的字典
        """
        batch_size = tcr_idx.size(0)
        
        # 创建注意力掩码
        if tcr_mask is None:
            tcr_mask = torch.ones_like(tcr_idx, dtype=torch.bool)
        if pep_mask is None:
            pep_mask = torch.ones_like(pep_idx, dtype=torch.bool)
        if hla_mask is None:
            hla_mask = torch.ones_like(hla_idx, dtype=torch.bool)
        
        # 计算序列长度
        tcr_lengths = tcr_mask.sum(dim=1)
        pep_lengths = pep_mask.sum(dim=1)
        hla_lengths = hla_mask.sum(dim=1)
        
        # 创建TCR-HLA交叉注意力掩码
        tcr_hla_attn_mask = ~tcr_mask.unsqueeze(2).expand(-1, -1, hla_idx.size(1))
        
        # 获取二元模型的编码表示和注意力权重
        with torch.set_grad_enabled(self.joint_training):
            # TCR-Pep模型前向传播
            tcr_pep_outputs = self.tcr_pep_model(
                tcr_idx, pep_idx, tcr_biochem, pep_biochem, tcr_mask, pep_mask
            )
            
            # HLA-Pep模型前向传播
            hla_pep_outputs = self.hla_pep_model(
                hla_idx, pep_idx, hla_biochem, pep_biochem, hla_mask, pep_mask
            )
        
        # 获取TCR和HLA的编码表示
        tcr_embed = self.tcr_pep_model.tcr_embedding(tcr_idx, tcr_biochem)
        tcr_embed = self.tcr_pep_model.tcr_pos_encoding(tcr_embed)
        for layer in self.tcr_pep_model.tcr_encoder:
            tcr_embed = layer(tcr_embed, ~tcr_mask.unsqueeze(1).expand(-1, tcr_idx.size(1), -1))
        
        hla_embed = self.hla_pep_model.hla_embedding(hla_idx, hla_biochem)
        hla_embed = self.hla_pep_model.hla_pos_encoding(hla_embed)
        for layer in self.hla_pep_model.hla_encoder:
            hla_embed = layer(hla_embed, ~hla_mask.unsqueeze(1).expand(-1, hla_idx.size(1), -1))
        
        # TCR-HLA交叉注意力
        if self.attention_type == "physical" or self.attention_type == "fused":
            # 物理滑动注意力或融合注意力需要序列长度
            tcr_hla_output, tcr_hla_attn = self.tcr_hla_attention(
                tcr_embed, hla_embed, hla_embed, 
                tcr_lengths, hla_lengths, 
                'trimer', tcr_hla_attn_mask
            )
        else:
            # 数据驱动注意力不需要序列长度
            tcr_hla_output, tcr_hla_attn = self.tcr_hla_attention(
                tcr_embed, hla_embed, hla_embed, tcr_hla_attn_mask
            )
        
        # 获取全局表示
        tcr_global = tcr_embed.sum(dim=1) / (tcr_lengths.unsqueeze(1).float() + 1e-8)
        hla_global = hla_embed.sum(dim=1) / (hla_lengths.unsqueeze(1).float() + 1e-8)
        pep_global = self.hla_pep_model.pep_embedding(pep_idx, pep_biochem)
        pep_global = self.hla_pep_model.pep_pos_encoding(pep_global)
        for layer in self.hla_pep_model.pep_encoder:
            pep_global = layer(pep_global, ~pep_mask.unsqueeze(1).expand(-1, pep_idx.size(1), -1))
        pep_global = pep_global.sum(dim=1) / (pep_lengths.unsqueeze(1).float() + 1e-8)
        
        # 整合三元互作信息
        trimer_features = torch.cat([tcr_global, hla_global, pep_global], dim=1)
        trimer_integrated = self.trimer_integration(trimer_features)
        
        # 预测结果
        pred = self.output_layer(trimer_integrated).squeeze(-1)
        
        # 收集所有注意力权重
        attn_weights = {}
        
        # 添加TCR-Pep注意力权重
        if isinstance(tcr_pep_outputs["attn_weights"], dict):
            for key, value in tcr_pep_outputs["attn_weights"].items():
                attn_weights[f"tcr_pep_{key}"] = value
        else:
            attn_weights["tcr_pep"] = tcr_pep_outputs["attn_weights"]
        
        # 添加HLA-Pep注意力权重
        if isinstance(hla_pep_outputs["attn_weights"], dict):
            for key, value in hla_pep_outputs["attn_weights"].items():
                attn_weights[f"hla_pep_{key}"] = value
        else:
            attn_weights["hla_pep"] = hla_pep_outputs["attn_weights"]
        
        # 添加TCR-HLA注意力权重
        if isinstance(tcr_hla_attn, dict):
            for key, value in tcr_hla_attn.items():
                attn_weights[f"tcr_hla_{key}"] = value
        else:
            attn_weights["tcr_hla"] = tcr_hla_attn
        
        # 返回结果和注意力权重
        return {
            "pred": pred,
            "tcr_pep_pred": tcr_pep_outputs["pred"],
            "hla_pep_pred": hla_pep_outputs["pred"],
            "attn_weights": attn_weights
        }
    
    def freeze_binary_models(self):
        """
        冻结二元模型参数
        """
        for param in self.tcr_pep_model.parameters():
            param.requires_grad = False
        for param in self.hla_pep_model.parameters():
            param.requires_grad = False
    
    def unfreeze_binary_models(self):
        """
        解冻二元模型参数
        """
        for param in self.tcr_pep_model.parameters():
            param.requires_grad = True
        for param in self.hla_pep_model.parameters():
            param.requires_grad = True
    
    def get_binary_predictions(self, 
                              tcr_idx: torch.Tensor, 
                              pep_idx: torch.Tensor, 
                              hla_idx: torch.Tensor,
                              tcr_biochem: Optional[torch.Tensor] = None, 
                              pep_biochem: Optional[torch.Tensor] = None,
                              hla_biochem: Optional[torch.Tensor] = None,
                              tcr_mask: Optional[torch.Tensor] = None, 
                              pep_mask: Optional[torch.Tensor] = None,
                              hla_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        获取二元模型的预测结果
        
        Args:
            tcr_idx: TCR序列索引，形状为(batch_size, tcr_len)
            pep_idx: 肽段序列索引，形状为(batch_size, pep_len)
            hla_idx: HLA序列索引，形状为(batch_size, hla_len)
            tcr_biochem: TCR生化特征，形状为(batch_size, tcr_len, biochem_dim)
            pep_biochem: 肽段生化特征，形状为(batch_size, pep_len, biochem_dim)
            hla_biochem: HLA生化特征，形状为(batch_size, hla_len, biochem_dim)
            tcr_mask: TCR序列掩码，形状为(batch_size, tcr_len)
            pep_mask: 肽段序列掩码，形状为(batch_size, pep_len)
            hla_mask: HLA序列掩码，形状为(batch_size, hla_len)
            
        Returns:
            包含二元模型预测结果的字典
        """
        # TCR-Pep模型预测
        tcr_pep_outputs = self.tcr_pep_model(
            tcr_idx, pep_idx, tcr_biochem, pep_biochem, tcr_mask, pep_mask
        )
        
        # HLA-Pep模型预测
        hla_pep_outputs = self.hla_pep_model(
            hla_idx, pep_idx, hla_biochem, pep_biochem, hla_mask, pep_mask
        )
        
        return {
            "tcr_pep_pred": tcr_pep_outputs["pred"],
            "hla_pep_pred": hla_pep_outputs["pred"]
        } 