#!/usr/bin/env python
"""
模型功能测试脚本

该脚本用于测试模型的基本功能，包括数据加载、模型初始化和前向传播。
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Tuple, Union, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import MyDataSet, DataLoader
from src.models.tcr_pep_model import TCRPepModel
from src.models.hla_pep_model import HLAPepModel
from src.models.trimer_model import TrimerModel
from src.utils.config import load_config
from src.utils.logger import setup_logger


def test_binary_models(config: Dict[str, Any], device: torch.device, logger):
    """
    测试二元模型功能
    
    Args:
        config: 配置字典
        device: 设备
        logger: 日志记录器
    """
    logger.info("=== 测试二元模型功能 ===")
    
    # 创建TCR-Pep模型
    logger.info("创建TCR-Pep模型...")
    tcr_pep_model = TCRPepModel(
        vocab_size=22,  # 20种氨基酸 + 填充 + 未知
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        max_tcr_len=config['data']['max_tcr_len'],
        max_pep_len=config['data']['max_pep_len'],
        use_biochem=config['model']['use_biochem_features'],
        dropout=config['model']['dropout'],
        attention_type="fused",  # 使用融合注意力
        sigma=config['attention']['physical_sliding']['sigma'],
        num_iterations=config['attention']['physical_sliding']['num_iterations'],
        fusion_method=config['attention']['fusion_method'],
        fusion_weights=config['attention']['fusion_weights']
    ).to(device)
    
    # 创建HLA-Pep模型
    logger.info("创建HLA-Pep模型...")
    hla_pep_model = HLAPepModel(
        vocab_size=22,  # 20种氨基酸 + 填充 + 未知
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        max_hla_len=config['data']['max_hla_len'],
        max_pep_len=config['data']['max_pep_len'],
        use_biochem=config['model']['use_biochem_features'],
        dropout=config['model']['dropout'],
        attention_type="fused",  # 使用融合注意力
        sigma=config['attention']['physical_sliding']['sigma'],
        num_iterations=config['attention']['physical_sliding']['num_iterations'],
        fusion_method=config['attention']['fusion_method'],
        fusion_weights=config['attention']['fusion_weights']
    ).to(device)
    
    # 创建测试数据
    logger.info("创建测试数据...")
    batch_size = 2
    
    # TCR-Pep测试数据
    tcr_idx = torch.randint(0, 22, (batch_size, config['data']['max_tcr_len'])).to(device)
    pep_idx = torch.randint(0, 22, (batch_size, config['data']['max_pep_len'])).to(device)
    tcr_biochem = torch.rand(batch_size, config['data']['max_tcr_len'], 5).to(device)
    pep_biochem = torch.rand(batch_size, config['data']['max_pep_len'], 5).to(device)
    tcr_mask = torch.ones(batch_size, config['data']['max_tcr_len'], dtype=torch.bool).to(device)
    pep_mask = torch.ones(batch_size, config['data']['max_pep_len'], dtype=torch.bool).to(device)
    
    # HLA测试数据
    hla_idx = torch.randint(0, 22, (batch_size, config['data']['max_hla_len'])).to(device)
    hla_biochem = torch.rand(batch_size, config['data']['max_hla_len'], 5).to(device)
    hla_mask = torch.ones(batch_size, config['data']['max_hla_len'], dtype=torch.bool).to(device)
    
    # 测试TCR-Pep模型
    logger.info("测试TCR-Pep模型前向传播...")
    start_time = time.time()
    with torch.no_grad():
        tcr_pep_outputs = tcr_pep_model(
            tcr_idx, pep_idx, tcr_biochem, pep_biochem, tcr_mask, pep_mask
        )
    tcr_pep_time = time.time() - start_time
    logger.info(f"TCR-Pep模型前向传播耗时: {tcr_pep_time:.4f}秒")
    logger.info(f"TCR-Pep预测结果形状: {tcr_pep_outputs['pred'].shape}")
    logger.info(f"TCR-Pep注意力权重形状: {next(iter(tcr_pep_outputs['attn_weights'].values())).shape}")
    
    # 测试HLA-Pep模型
    logger.info("测试HLA-Pep模型前向传播...")
    start_time = time.time()
    with torch.no_grad():
        hla_pep_outputs = hla_pep_model(
            hla_idx, pep_idx, hla_biochem, pep_biochem, hla_mask, pep_mask
        )
    hla_pep_time = time.time() - start_time
    logger.info(f"HLA-Pep模型前向传播耗时: {hla_pep_time:.4f}秒")
    logger.info(f"HLA-Pep预测结果形状: {hla_pep_outputs['pred'].shape}")
    logger.info(f"HLA-Pep注意力权重形状: {next(iter(hla_pep_outputs['attn_weights'].values())).shape}")
    
    return tcr_pep_model, hla_pep_model


def test_trimer_model(tcr_pep_model: TCRPepModel, 
                     hla_pep_model: HLAPepModel, 
                     config: Dict[str, Any], 
                     device: torch.device, 
                     logger):
    """
    测试三元模型功能
    
    Args:
        tcr_pep_model: TCR-Pep模型
        hla_pep_model: HLA-Pep模型
        config: 配置字典
        device: 设备
        logger: 日志记录器
    """
    logger.info("\n=== 测试三元模型功能 ===")
    
    # 创建三元模型
    logger.info("创建三元模型...")
    trimer_model = TrimerModel(
        vocab_size=22,  # 20种氨基酸 + 填充 + 未知
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        max_tcr_len=config['data']['max_tcr_len'],
        max_hla_len=config['data']['max_hla_len'],
        max_pep_len=config['data']['max_pep_len'],
        use_biochem=config['model']['use_biochem_features'],
        dropout=config['model']['dropout'],
        attention_type="fused",  # 使用融合注意力
        sigma=config['attention']['physical_sliding']['sigma'],
        num_iterations=config['attention']['physical_sliding']['num_iterations'],
        fusion_method=config['attention']['fusion_method'],
        fusion_weights=config['attention']['fusion_weights'],
        tcr_pep_model=tcr_pep_model,
        hla_pep_model=hla_pep_model,
        joint_training=True
    ).to(device)
    
    # 创建测试数据
    logger.info("创建测试数据...")
    batch_size = 2
    
    # 测试数据
    tcr_idx = torch.randint(0, 22, (batch_size, config['data']['max_tcr_len'])).to(device)
    pep_idx = torch.randint(0, 22, (batch_size, config['data']['max_pep_len'])).to(device)
    hla_idx = torch.randint(0, 22, (batch_size, config['data']['max_hla_len'])).to(device)
    tcr_biochem = torch.rand(batch_size, config['data']['max_tcr_len'], 5).to(device)
    pep_biochem = torch.rand(batch_size, config['data']['max_pep_len'], 5).to(device)
    hla_biochem = torch.rand(batch_size, config['data']['max_hla_len'], 5).to(device)
    tcr_mask = torch.ones(batch_size, config['data']['max_tcr_len'], dtype=torch.bool).to(device)
    pep_mask = torch.ones(batch_size, config['data']['max_pep_len'], dtype=torch.bool).to(device)
    hla_mask = torch.ones(batch_size, config['data']['max_hla_len'], dtype=torch.bool).to(device)
    
    # 测试三元模型
    logger.info("测试三元模型前向传播...")
    start_time = time.time()
    with torch.no_grad():
        trimer_outputs = trimer_model(
            tcr_idx, pep_idx, hla_idx,
            tcr_biochem, pep_biochem, hla_biochem,
            tcr_mask, pep_mask, hla_mask
        )
    trimer_time = time.time() - start_time
    logger.info(f"三元模型前向传播耗时: {trimer_time:.4f}秒")
    logger.info(f"三元模型预测结果形状: {trimer_outputs['pred'].shape}")
    logger.info(f"三元模型TCR-Pep预测结果形状: {trimer_outputs['tcr_pep_pred'].shape}")
    logger.info(f"三元模型HLA-Pep预测结果形状: {trimer_outputs['hla_pep_pred'].shape}")
    
    # 检查注意力权重
    logger.info("检查注意力权重...")
    for key, value in trimer_outputs['attn_weights'].items():
        logger.info(f"注意力权重 {key}: 形状 {value.shape}")
    
    return trimer_model


def test_real_data(config: Dict[str, Any], device: torch.device, logger):
    """
    测试真实数据加载和模型推理
    
    Args:
        config: 配置字典
        device: 设备
        logger: 日志记录器
    """
    logger.info("\n=== 测试真实数据加载和模型推理 ===")
    
    # 尝试加载测试数据
    test_file = "data/processed/trimer_test.csv"
    if not os.path.exists(test_file):
        test_file = "data/raw/test_data.csv"
        if not os.path.exists(test_file):
            logger.warning(f"找不到测试数据文件: {test_file}")
            return
    
    logger.info(f"加载测试数据: {test_file}")
    test_df = pd.read_csv(test_file)
    logger.info(f"测试数据大小: {test_df.shape}")
    
    # 只使用少量数据进行测试
    test_df = test_df.head(10)
    
    # 创建数据集
    logger.info("创建数据集...")
    try:
        dataset = MyDataSet(
            test_df, 
            data_type='trimer',
            max_tcr_len=config['data']['max_tcr_len'],
            max_pep_len=config['data']['max_pep_len'],
            max_hla_len=config['data']['max_hla_len']
        )
        
        # 创建数据加载器
        logger.info("创建数据加载器...")
        dataloader = DataLoader(
            dataset, 
            batch_size=5,
            shuffle=False,
            num_workers=0
        )
        
        # 创建模型
        logger.info("创建模型...")
        tcr_pep_model = TCRPepModel(
            vocab_size=22,
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            max_tcr_len=config['data']['max_tcr_len'],
            max_pep_len=config['data']['max_pep_len'],
            use_biochem=config['model']['use_biochem_features'],
            dropout=config['model']['dropout'],
            attention_type="fused"  # 使用融合注意力
        ).to(device)
        
        hla_pep_model = HLAPepModel(
            vocab_size=22,
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            max_hla_len=config['data']['max_hla_len'],
            max_pep_len=config['data']['max_pep_len'],
            use_biochem=config['model']['use_biochem_features'],
            dropout=config['model']['dropout'],
            attention_type="fused"  # 使用融合注意力
        ).to(device)
        
        trimer_model = TrimerModel(
            vocab_size=22,
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            max_tcr_len=config['data']['max_tcr_len'],
            max_hla_len=config['data']['max_hla_len'],
            max_pep_len=config['data']['max_pep_len'],
            use_biochem=config['model']['use_biochem_features'],
            dropout=config['model']['dropout'],
            tcr_pep_model=tcr_pep_model,
            hla_pep_model=hla_pep_model,
            attention_type="fused"  # 使用融合注意力
        ).to(device)
        
        # 测试数据加载和模型推理
        logger.info("测试数据加载和模型推理...")
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
            
            # 前向传播
            with torch.no_grad():
                outputs = trimer_model(
                    tcr_idx, pep_idx, hla_idx,
                    tcr_biochem, pep_biochem, hla_biochem,
                    tcr_mask, pep_mask, hla_mask
                )
            
            logger.info(f"批次大小: {tcr_idx.shape[0]}")
            logger.info(f"预测结果: {outputs['pred']}")
            break
        
        logger.info("真实数据测试完成")
    
    except Exception as e:
        logger.error(f"真实数据测试失败: {str(e)}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="测试TCR-HLA-Pep模型功能")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="配置文件路径")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU设备ID，-1表示使用CPU")
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("test_model", os.path.join(args.log_dir, "test_model.log"))
    
    # 加载配置
    logger.info(f"从{args.config}加载配置...")
    config = load_config(args.config)
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"使用GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    
    # 测试二元模型
    tcr_pep_model, hla_pep_model = test_binary_models(config, device, logger)
    
    # 测试三元模型
    trimer_model = test_trimer_model(tcr_pep_model, hla_pep_model, config, device, logger)
    
    # 测试真实数据
    test_real_data(config, device, logger)
    
    logger.info("测试完成")


if __name__ == "__main__":
    main() 