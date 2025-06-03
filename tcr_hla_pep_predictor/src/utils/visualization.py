"""
可视化工具模块

该模块提供残基互作可视化功能，包括注意力热力图和残基互作网络图。
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def extract_attention_weights(model_outputs: Dict[str, Any], 
                             interaction_type: str) -> np.ndarray:
    """
    从模型输出中提取注意力权重
    
    Args:
        model_outputs: 模型输出字典
        interaction_type: 互作类型，'tcr_pep'、'hla_pep'或'trimer'
        
    Returns:
        注意力权重矩阵
    """
    attn_weights = model_outputs['attn_weights']
    
    if interaction_type == 'tcr_pep':
        if 'tcr_pep_fused' in attn_weights:
            return attn_weights['tcr_pep_fused'].cpu().numpy()
        elif 'tcr_pep_physical' in attn_weights:
            return attn_weights['tcr_pep_physical'].cpu().numpy()
        elif 'tcr_pep_data_driven' in attn_weights:
            return attn_weights['tcr_pep_data_driven'].cpu().numpy()
        else:
            return attn_weights['attention'].cpu().numpy()
    
    elif interaction_type == 'hla_pep':
        if 'hla_pep_fused' in attn_weights:
            return attn_weights['hla_pep_fused'].cpu().numpy()
        elif 'hla_pep_physical' in attn_weights:
            return attn_weights['hla_pep_physical'].cpu().numpy()
        elif 'hla_pep_data_driven' in attn_weights:
            return attn_weights['hla_pep_data_driven'].cpu().numpy()
        else:
            return attn_weights['attention'].cpu().numpy()
    
    elif interaction_type == 'trimer':
        # 提取TCR-HLA注意力权重
        if 'tcr_hla_fused' in attn_weights:
            return attn_weights['tcr_hla_fused'].cpu().numpy()
        elif 'tcr_hla_physical' in attn_weights:
            return attn_weights['tcr_hla_physical'].cpu().numpy()
        elif 'tcr_hla_data_driven' in attn_weights:
            return attn_weights['tcr_hla_data_driven'].cpu().numpy()
        else:
            return attn_weights['tcr_hla'].cpu().numpy()
    
    else:
        raise ValueError(f"不支持的互作类型: {interaction_type}")


def process_attention_weights(attn_weights: np.ndarray, 
                             seq1_mask: np.ndarray, 
                             seq2_mask: np.ndarray) -> np.ndarray:
    """
    处理注意力权重，移除填充位置
    
    Args:
        attn_weights: 注意力权重矩阵，形状为(batch_size, seq1_len, seq2_len)
        seq1_mask: 第一个序列的掩码，形状为(batch_size, seq1_len)
        seq2_mask: 第二个序列的掩码，形状为(batch_size, seq2_len)
        
    Returns:
        处理后的注意力权重矩阵，移除填充位置
    """
    batch_size, seq1_len, seq2_len = attn_weights.shape
    processed_weights = []
    
    for i in range(batch_size):
        # 获取有效长度
        valid_len1 = int(seq1_mask[i].sum())
        valid_len2 = int(seq2_mask[i].sum())
        
        # 提取有效区域的注意力权重
        valid_weights = attn_weights[i, :valid_len1, :valid_len2]
        processed_weights.append(valid_weights)
    
    return processed_weights


def plot_attention_heatmap(attn_weights: np.ndarray, 
                          seq1: str, 
                          seq2: str, 
                          title: str, 
                          output_path: Optional[str] = None, 
                          cmap: str = 'viridis',
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    绘制注意力热力图
    
    Args:
        attn_weights: 注意力权重矩阵，形状为(seq1_len, seq2_len)
        seq1: 第一个序列
        seq2: 第二个序列
        title: 图表标题
        output_path: 输出文件路径，如果为None则不保存
        cmap: 颜色映射
        figsize: 图表大小
        
    Returns:
        matplotlib图表对象
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(attn_weights, 
                cmap=cmap, 
                annot=False, 
                xticklabels=list(seq2), 
                yticklabels=list(seq1),
                ax=ax)
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(f'{seq2} 残基')
    ax.set_ylabel(f'{seq1} 残基')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_interactive_heatmap(attn_weights: np.ndarray, 
                              seq1: str, 
                              seq2: str, 
                              title: str, 
                              output_path: Optional[str] = None) -> go.Figure:
    """
    创建交互式热力图
    
    Args:
        attn_weights: 注意力权重矩阵，形状为(seq1_len, seq2_len)
        seq1: 第一个序列
        seq2: 第二个序列
        title: 图表标题
        output_path: 输出文件路径，如果为None则不保存
        
    Returns:
        plotly图表对象
    """
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=attn_weights,
        x=list(seq2),
        y=list(seq1),
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='%{y} -> %{x}: %{z:.4f}<extra></extra>'
    ))
    
    # 设置标题和标签
    fig.update_layout(
        title=title,
        xaxis_title=f'{seq2} 残基',
        yaxis_title=f'{seq1} 残基',
        width=800,
        height=600
    )
    
    # 保存图表
    if output_path is not None:
        fig.write_html(output_path)
    
    return fig


def identify_key_interactions(attn_weights: np.ndarray, 
                             seq1: str, 
                             seq2: str, 
                             threshold: float = 0.1) -> List[Tuple[str, str, float]]:
    """
    识别关键互作残基对
    
    Args:
        attn_weights: 注意力权重矩阵，形状为(seq1_len, seq2_len)
        seq1: 第一个序列
        seq2: 第二个序列
        threshold: 注意力权重阈值
        
    Returns:
        关键互作残基对列表，每个元素为(残基1, 残基2, 注意力权重)
    """
    key_interactions = []
    
    # 找出高于阈值的注意力权重
    for i in range(len(seq1)):
        for j in range(len(seq2)):
            if attn_weights[i, j] >= threshold:
                key_interactions.append((
                    f"{seq1[i]}{i+1}",  # 残基1，格式为"氨基酸+位置"
                    f"{seq2[j]}{j+1}",  # 残基2，格式为"氨基酸+位置"
                    attn_weights[i, j]   # 注意力权重
                ))
    
    # 按注意力权重降序排序
    key_interactions.sort(key=lambda x: x[2], reverse=True)
    
    return key_interactions


def visualize_tcr_pep_interactions(model_outputs: Dict[str, Any], 
                                  tcr_seq: str, 
                                  pep_seq: str, 
                                  tcr_mask: np.ndarray, 
                                  pep_mask: np.ndarray, 
                                  output_dir: str, 
                                  sample_id: str,
                                  interactive: bool = True) -> Dict[str, Any]:
    """
    可视化TCR-Pep互作
    
    Args:
        model_outputs: 模型输出字典
        tcr_seq: TCR序列
        pep_seq: 肽段序列
        tcr_mask: TCR序列掩码
        pep_mask: 肽段序列掩码
        output_dir: 输出目录
        sample_id: 样本ID
        interactive: 是否生成交互式图表
        
    Returns:
        可视化结果字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取注意力权重
    attn_weights = extract_attention_weights(model_outputs, 'tcr_pep')
    
    # 处理注意力权重
    processed_weights = process_attention_weights(attn_weights, tcr_mask, pep_mask)
    
    # 获取有效序列
    valid_tcr = tcr_seq[:int(tcr_mask.sum())]
    valid_pep = pep_seq[:int(pep_mask.sum())]
    
    # 静态热力图
    static_output_path = os.path.join(output_dir, f"{sample_id}_tcr_pep_heatmap.png")
    fig_static = plot_attention_heatmap(
        processed_weights[0],
        valid_tcr,
        valid_pep,
        f"TCR-Pep 互作热力图 (样本 {sample_id})",
        static_output_path
    )
    
    # 交互式热力图
    interactive_output_path = None
    fig_interactive = None
    if interactive:
        interactive_output_path = os.path.join(output_dir, f"{sample_id}_tcr_pep_heatmap.html")
        fig_interactive = create_interactive_heatmap(
            processed_weights[0],
            valid_tcr,
            valid_pep,
            f"TCR-Pep 互作热力图 (样本 {sample_id})",
            interactive_output_path
        )
    
    # 识别关键互作残基对
    key_interactions = identify_key_interactions(processed_weights[0], valid_tcr, valid_pep)
    
    # 保存关键互作残基对
    interactions_output_path = os.path.join(output_dir, f"{sample_id}_tcr_pep_interactions.csv")
    pd.DataFrame(key_interactions, columns=['TCR_Residue', 'Peptide_Residue', 'Attention_Weight']).to_csv(
        interactions_output_path, index=False
    )
    
    return {
        'static_fig': fig_static,
        'interactive_fig': fig_interactive,
        'static_output_path': static_output_path,
        'interactive_output_path': interactive_output_path,
        'key_interactions': key_interactions,
        'interactions_output_path': interactions_output_path
    }


def visualize_hla_pep_interactions(model_outputs: Dict[str, Any], 
                                  hla_seq: str, 
                                  pep_seq: str, 
                                  hla_mask: np.ndarray, 
                                  pep_mask: np.ndarray, 
                                  output_dir: str, 
                                  sample_id: str,
                                  interactive: bool = True) -> Dict[str, Any]:
    """
    可视化HLA-Pep互作
    
    Args:
        model_outputs: 模型输出字典
        hla_seq: HLA序列
        pep_seq: 肽段序列
        hla_mask: HLA序列掩码
        pep_mask: 肽段序列掩码
        output_dir: 输出目录
        sample_id: 样本ID
        interactive: 是否生成交互式图表
        
    Returns:
        可视化结果字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取注意力权重
    attn_weights = extract_attention_weights(model_outputs, 'hla_pep')
    
    # 处理注意力权重
    processed_weights = process_attention_weights(attn_weights, hla_mask, pep_mask)
    
    # 获取有效序列
    valid_hla = hla_seq[:int(hla_mask.sum())]
    valid_pep = pep_seq[:int(pep_mask.sum())]
    
    # 静态热力图
    static_output_path = os.path.join(output_dir, f"{sample_id}_hla_pep_heatmap.png")
    fig_static = plot_attention_heatmap(
        processed_weights[0],
        valid_hla,
        valid_pep,
        f"HLA-Pep 互作热力图 (样本 {sample_id})",
        static_output_path
    )
    
    # 交互式热力图
    interactive_output_path = None
    fig_interactive = None
    if interactive:
        interactive_output_path = os.path.join(output_dir, f"{sample_id}_hla_pep_heatmap.html")
        fig_interactive = create_interactive_heatmap(
            processed_weights[0],
            valid_hla,
            valid_pep,
            f"HLA-Pep 互作热力图 (样本 {sample_id})",
            interactive_output_path
        )
    
    # 识别关键互作残基对
    key_interactions = identify_key_interactions(processed_weights[0], valid_hla, valid_pep)
    
    # 保存关键互作残基对
    interactions_output_path = os.path.join(output_dir, f"{sample_id}_hla_pep_interactions.csv")
    pd.DataFrame(key_interactions, columns=['HLA_Residue', 'Peptide_Residue', 'Attention_Weight']).to_csv(
        interactions_output_path, index=False
    )
    
    return {
        'static_fig': fig_static,
        'interactive_fig': fig_interactive,
        'static_output_path': static_output_path,
        'interactive_output_path': interactive_output_path,
        'key_interactions': key_interactions,
        'interactions_output_path': interactions_output_path
    }


def visualize_trimer_interactions(model_outputs: Dict[str, Any], 
                                 tcr_seq: str, 
                                 hla_seq: str, 
                                 pep_seq: str, 
                                 tcr_mask: np.ndarray, 
                                 hla_mask: np.ndarray, 
                                 pep_mask: np.ndarray, 
                                 output_dir: str, 
                                 sample_id: str,
                                 interactive: bool = True) -> Dict[str, Any]:
    """
    可视化TCR-HLA-Pep三元互作
    
    Args:
        model_outputs: 模型输出字典
        tcr_seq: TCR序列
        hla_seq: HLA序列
        pep_seq: 肽段序列
        tcr_mask: TCR序列掩码
        hla_mask: HLA序列掩码
        pep_mask: 肽段序列掩码
        output_dir: 输出目录
        sample_id: 样本ID
        interactive: 是否生成交互式图表
        
    Returns:
        可视化结果字典
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化TCR-Pep互作
    tcr_pep_results = visualize_tcr_pep_interactions(
        model_outputs, tcr_seq, pep_seq, tcr_mask, pep_mask, output_dir, sample_id, interactive
    )
    
    # 可视化HLA-Pep互作
    hla_pep_results = visualize_hla_pep_interactions(
        model_outputs, hla_seq, pep_seq, hla_mask, pep_mask, output_dir, sample_id, interactive
    )
    
    # 提取TCR-HLA注意力权重
    tcr_hla_attn_weights = extract_attention_weights(model_outputs, 'trimer')
    
    # 处理注意力权重
    tcr_hla_processed_weights = process_attention_weights(tcr_hla_attn_weights, tcr_mask, hla_mask)
    
    # 获取有效序列
    valid_tcr = tcr_seq[:int(tcr_mask.sum())]
    valid_hla = hla_seq[:int(hla_mask.sum())]
    valid_pep = pep_seq[:int(pep_mask.sum())]
    
    # 静态TCR-HLA热力图
    tcr_hla_static_path = os.path.join(output_dir, f"{sample_id}_tcr_hla_heatmap.png")
    tcr_hla_fig_static = plot_attention_heatmap(
        tcr_hla_processed_weights[0],
        valid_tcr,
        valid_hla,
        f"TCR-HLA 互作热力图 (样本 {sample_id})",
        tcr_hla_static_path
    )
    
    # 交互式TCR-HLA热力图
    tcr_hla_interactive_path = None
    tcr_hla_fig_interactive = None
    if interactive:
        tcr_hla_interactive_path = os.path.join(output_dir, f"{sample_id}_tcr_hla_heatmap.html")
        tcr_hla_fig_interactive = create_interactive_heatmap(
            tcr_hla_processed_weights[0],
            valid_tcr,
            valid_hla,
            f"TCR-HLA 互作热力图 (样本 {sample_id})",
            tcr_hla_interactive_path
        )
    
    # 识别TCR-HLA关键互作残基对
    tcr_hla_key_interactions = identify_key_interactions(tcr_hla_processed_weights[0], valid_tcr, valid_hla)
    
    # 保存TCR-HLA关键互作残基对
    tcr_hla_interactions_path = os.path.join(output_dir, f"{sample_id}_tcr_hla_interactions.csv")
    pd.DataFrame(tcr_hla_key_interactions, columns=['TCR_Residue', 'HLA_Residue', 'Attention_Weight']).to_csv(
        tcr_hla_interactions_path, index=False
    )
    
    # 创建三元互作综合可视化（仅交互式）
    trimer_fig = None
    trimer_output_path = None
    if interactive:
        trimer_output_path = os.path.join(output_dir, f"{sample_id}_trimer_interactions.html")
        
        # 创建三个子图的交互式可视化
        trimer_fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"TCR-Pep 互作热力图 (样本 {sample_id})",
                f"HLA-Pep 互作热力图 (样本 {sample_id})",
                f"TCR-HLA 互作热力图 (样本 {sample_id})",
                f"三元互作预测概率: {model_outputs['pred'].item():.4f}"
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "indicator"}]
            ]
        )
        
        # TCR-Pep热力图
        trimer_fig.add_trace(
            go.Heatmap(
                z=tcr_pep_results['static_fig'].get_axes()[0].get_images()[0].get_array(),
                x=list(valid_pep),
                y=list(valid_tcr),
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='TCR %{y} -> Pep %{x}: %{z:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # HLA-Pep热力图
        trimer_fig.add_trace(
            go.Heatmap(
                z=hla_pep_results['static_fig'].get_axes()[0].get_images()[0].get_array(),
                x=list(valid_pep),
                y=list(valid_hla),
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='HLA %{y} -> Pep %{x}: %{z:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # TCR-HLA热力图
        trimer_fig.add_trace(
            go.Heatmap(
                z=tcr_hla_processed_weights[0],
                x=list(valid_hla),
                y=list(valid_tcr),
                colorscale='Viridis',
                hoverongaps=False,
                hovertemplate='TCR %{y} -> HLA %{x}: %{z:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 预测概率指示器
        trimer_fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(model_outputs['pred'].item()),
                title={'text': "结合概率"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "red"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.5
                    }
                }
            ),
            row=2, col=2
        )
        
        # 更新布局
        trimer_fig.update_layout(
            height=800,
            width=1000,
            title_text=f"TCR-HLA-Pep 三元互作可视化 (样本 {sample_id})"
        )
        
        # 保存图表
        trimer_fig.write_html(trimer_output_path)
    
    # 返回可视化结果
    return {
        'tcr_pep_results': tcr_pep_results,
        'hla_pep_results': hla_pep_results,
        'tcr_hla_static_fig': tcr_hla_fig_static,
        'tcr_hla_interactive_fig': tcr_hla_fig_interactive,
        'tcr_hla_static_path': tcr_hla_static_path,
        'tcr_hla_interactive_path': tcr_hla_interactive_path,
        'tcr_hla_key_interactions': tcr_hla_key_interactions,
        'tcr_hla_interactions_path': tcr_hla_interactions_path,
        'trimer_fig': trimer_fig,
        'trimer_output_path': trimer_output_path
    }


def visualize_sample(model, 
                    sample: Dict[str, torch.Tensor], 
                    aa_vocab: List[str],
                    output_dir: str, 
                    sample_id: str,
                    interactive: bool = True) -> Dict[str, Any]:
    """
    可视化单个样本的互作
    
    Args:
        model: 模型
        sample: 样本字典
        aa_vocab: 氨基酸词汇表
        output_dir: 输出目录
        sample_id: 样本ID
        interactive: 是否生成交互式图表
        
    Returns:
        可视化结果字典
    """
    # 设置为评估模式
    model.eval()
    
    # 准备输入数据
    device = next(model.parameters()).device
    tcr_idx = sample['tcr_idx'].unsqueeze(0).to(device)
    pep_idx = sample['pep_idx'].unsqueeze(0).to(device)
    hla_idx = sample['hla_idx'].unsqueeze(0).to(device)
    tcr_biochem = sample['tcr_biochem'].unsqueeze(0).to(device) if 'tcr_biochem' in sample else None
    pep_biochem = sample['pep_biochem'].unsqueeze(0).to(device) if 'pep_biochem' in sample else None
    hla_biochem = sample['hla_biochem'].unsqueeze(0).to(device) if 'hla_biochem' in sample else None
    tcr_mask = sample['tcr_mask'].unsqueeze(0).to(device) if 'tcr_mask' in sample else None
    pep_mask = sample['pep_mask'].unsqueeze(0).to(device) if 'pep_mask' in sample else None
    hla_mask = sample['hla_mask'].unsqueeze(0).to(device) if 'hla_mask' in sample else None
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            tcr_idx, pep_idx, hla_idx,
            tcr_biochem, pep_biochem, hla_biochem,
            tcr_mask, pep_mask, hla_mask
        )
    
    # 将索引转换为氨基酸序列
    tcr_seq = ''.join([aa_vocab[idx] for idx in tcr_idx[0].cpu().numpy() if idx < len(aa_vocab)])
    pep_seq = ''.join([aa_vocab[idx] for idx in pep_idx[0].cpu().numpy() if idx < len(aa_vocab)])
    hla_seq = ''.join([aa_vocab[idx] for idx in hla_idx[0].cpu().numpy() if idx < len(aa_vocab)])
    
    # 可视化三元互作
    vis_results = visualize_trimer_interactions(
        outputs,
        tcr_seq, hla_seq, pep_seq,
        tcr_mask[0].cpu().numpy(), hla_mask[0].cpu().numpy(), pep_mask[0].cpu().numpy(),
        output_dir, sample_id, interactive
    )
    
    return vis_results