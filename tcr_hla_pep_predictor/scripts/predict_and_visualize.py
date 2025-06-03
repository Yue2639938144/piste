#!/usr/bin/env python
"""
TCR-HLA-Pep预测和可视化脚本

该脚本用于使用训练好的模型进行预测，并生成残基互作可视化。
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import MyDataSet, DataLoader
from src.models.trimer_model import TrimerModel
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.visualization import visualize_sample


def load_model(model_path: str, device: torch.device) -> Tuple[TrimerModel, Dict]:
    """
    加载模型
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        模型和配置字典
    """
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # 创建模型
    model = TrimerModel(
        vocab_size=22,  # 20种氨基酸 + 填充 + 未知
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        max_tcr_len=config['data']['max_tcr_len'],
        max_hla_len=config['data']['max_hla_len'],
        max_pep_len=config['data']['max_pep_len'],
        use_biochem=config['model']['use_biochem_features'],
        biochem_dim=config['model']['biochem_dim'],
        dropout=config['model']['dropout'],
        attention_type=config['attention']['fusion_method'],
        sigma=config['attention']['physical_sliding']['sigma'],
        num_iterations=config['attention']['physical_sliding']['num_iterations'],
        fusion_method=config['attention']['fusion_method'],
        fusion_weights=config['attention']['fusion_weights']
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, config


def predict(model: TrimerModel, 
           dataloader: DataLoader, 
           threshold: float, 
           device: torch.device) -> Dict[str, np.ndarray]:
    """
    使用模型进行预测
    
    Args:
        model: 模型
        dataloader: 数据加载器
        threshold: 分类阈值
        device: 设备
        
    Returns:
        包含预测结果的字典
    """
    model.eval()
    all_preds = []
    all_binary_preds = []
    all_tcr_pep_preds = []
    all_hla_pep_preds = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取数据
            tcr_idx = batch['tcr_idx'].to(device)
            pep_idx = batch['pep_idx'].to(device)
            hla_idx = batch['hla_idx'].to(device)
            tcr_biochem = batch['tcr_biochem'].to(device) if 'tcr_biochem' in batch else None
            pep_biochem = batch['pep_biochem'].to(device) if 'pep_biochem' in batch else None
            hla_biochem = batch['hla_biochem'].to(device) if 'hla_biochem' in batch else None
            tcr_mask = batch['tcr_mask'].to(device) if 'tcr_mask' in batch else None
            pep_mask = batch['pep_mask'].to(device) if 'pep_mask' in batch else None
            hla_mask = batch['hla_mask'].to(device) if 'hla_mask' in batch else None
            
            # 获取样本ID
            sample_ids = batch['id'] if 'id' in batch else [f"sample_{i}" for i in range(tcr_idx.size(0))]
            
            # 前向传播
            outputs = model(
                tcr_idx, pep_idx, hla_idx,
                tcr_biochem, pep_biochem, hla_biochem,
                tcr_mask, pep_mask, hla_mask
            )
            
            # 收集预测结果
            preds = outputs['pred'].detach().cpu().numpy()
            binary_preds = (preds >= threshold).astype(int)
            tcr_pep_preds = outputs['tcr_pep_pred'].detach().cpu().numpy()
            hla_pep_preds = outputs['hla_pep_pred'].detach().cpu().numpy()
            
            all_preds.append(preds)
            all_binary_preds.append(binary_preds)
            all_tcr_pep_preds.append(tcr_pep_preds)
            all_hla_pep_preds.append(hla_pep_preds)
            all_sample_ids.extend(sample_ids)
    
    # 合并预测结果
    all_preds = np.concatenate(all_preds)
    all_binary_preds = np.concatenate(all_binary_preds)
    all_tcr_pep_preds = np.concatenate(all_tcr_pep_preds)
    all_hla_pep_preds = np.concatenate(all_hla_pep_preds)
    
    return {
        'sample_ids': np.array(all_sample_ids),
        'preds': all_preds,
        'binary_preds': all_binary_preds,
        'tcr_pep_preds': all_tcr_pep_preds,
        'hla_pep_preds': all_hla_pep_preds
    }


def save_predictions(predictions: Dict[str, np.ndarray], output_path: str):
    """
    保存预测结果
    
    Args:
        predictions: 预测结果字典
        output_path: 输出路径
    """
    # 创建DataFrame
    df = pd.DataFrame({
        'sample_id': predictions['sample_ids'],
        'prediction': predictions['preds'],
        'binary_prediction': predictions['binary_preds'],
        'tcr_pep_prediction': predictions['tcr_pep_preds'],
        'hla_pep_prediction': predictions['hla_pep_preds']
    })
    
    # 保存为CSV
    df.to_csv(output_path, index=False)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="TCR-HLA-Pep预测和可视化")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--input", type=str, required=True, help="输入数据文件路径")
    parser.add_argument("--output_dir", type=str, default="results", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化")
    parser.add_argument("--visualize_samples", type=int, default=5, help="可视化样本数量")
    parser.add_argument("--interactive", action="store_true", help="是否生成交互式可视化")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("predict", os.path.join(args.output_dir, "predict.log"))
    
    # 加载配置
    logger.info(f"从{args.config}加载配置...")
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    logger.info(f"从{args.model}加载模型...")
    model, model_config = load_model(args.model, device)
    
    # 获取分类阈值
    threshold = model_config.get('threshold', 0.5)
    logger.info(f"使用分类阈值: {threshold}")
    
    # 加载数据
    logger.info(f"从{args.input}加载数据...")
    input_df = pd.read_csv(args.input)
    logger.info(f"数据大小: {input_df.shape}")
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset = MyDataSet(
        input_df, 
        data_type='trimer',
        max_tcr_len=config['data']['max_tcr_len'],
        max_hla_len=config['data']['max_hla_len'],
        max_pep_len=config['data']['max_pep_len']
    )
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 进行预测
    logger.info("进行预测...")
    predictions = predict(model, dataloader, threshold, device)
    
    # 保存预测结果
    output_path = os.path.join(args.output_dir, "predictions.csv")
    logger.info(f"保存预测结果到{output_path}...")
    save_predictions(predictions, output_path)
    
    # 可视化
    if args.visualize:
        logger.info("生成可视化...")
        vis_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 获取氨基酸词汇表
        aa_vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                   'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 
                   '<pad>', '<unk>']
        
        # 选择要可视化的样本
        # 优先选择预测概率接近阈值的样本，因为这些样本更有趣
        pred_diff = np.abs(predictions['preds'] - threshold)
        vis_indices = np.argsort(pred_diff)[:args.visualize_samples]
        
        # 可视化选定的样本
        for i, idx in enumerate(vis_indices):
            sample_id = predictions['sample_ids'][idx]
            sample = dataset[idx]
            logger.info(f"可视化样本 {i+1}/{len(vis_indices)}: {sample_id}")
            
            # 可视化样本
            vis_results = visualize_sample(
                model, 
                sample, 
                aa_vocab, 
                vis_dir, 
                sample_id, 
                args.interactive
            )
            
            logger.info(f"样本 {sample_id} 可视化完成")
    
    logger.info("预测和可视化完成!")


if __name__ == "__main__":
    main() 