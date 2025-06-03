#!/usr/bin/env python
"""
TCR-Pep模型训练脚本

该脚本用于训练TCR-Pep二元互作预测模型。
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Union, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import MyDataSet, DataLoader
from src.models.tcr_pep_model import TCRPepModel
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.early_stopping import EarlyStopping
from src.utils.threshold_finder import find_optimal_threshold
from src.utils.metrics import calculate_metrics, print_metrics_report


def train_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               criterion: nn.Module, 
               optimizer: optim.Optimizer, 
               device: torch.device) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        
    Returns:
        包含训练指标的字典
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        # 获取数据
        tcr_idx = batch['tcr_idx'].to(device)
        pep_idx = batch['pep_idx'].to(device)
        tcr_biochem = batch['tcr_biochem'].to(device) if 'tcr_biochem' in batch else None
        pep_biochem = batch['pep_biochem'].to(device) if 'pep_biochem' in batch else None
        tcr_mask = batch['tcr_mask'].to(device) if 'tcr_mask' in batch else None
        pep_mask = batch['pep_mask'].to(device) if 'pep_mask' in batch else None
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(tcr_idx, pep_idx, tcr_biochem, pep_biochem, tcr_mask, pep_mask)
        preds = outputs['pred']
        
        # 计算损失
        loss = criterion(preds, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        
        # 收集预测结果和标签
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 合并预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc']
    }


def validate(model: nn.Module, 
            dataloader: DataLoader, 
            criterion: nn.Module, 
            device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        包含验证指标的字典，预测概率数组和标签数组
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取数据
            tcr_idx = batch['tcr_idx'].to(device)
            pep_idx = batch['pep_idx'].to(device)
            tcr_biochem = batch['tcr_biochem'].to(device) if 'tcr_biochem' in batch else None
            pep_biochem = batch['pep_biochem'].to(device) if 'pep_biochem' in batch else None
            tcr_mask = batch['tcr_mask'].to(device) if 'tcr_mask' in batch else None
            pep_mask = batch['pep_mask'].to(device) if 'pep_mask' in batch else None
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(tcr_idx, pep_idx, tcr_biochem, pep_biochem, tcr_mask, pep_mask)
            preds = outputs['pred']
            
            # 计算损失
            loss = criterion(preds, labels)
            
            # 累计损失
            total_loss += loss.item()
            
            # 收集预测结果和标签
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 合并预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc']
    }, all_preds, all_labels


def test(model: nn.Module, 
        dataloader: DataLoader, 
        threshold: float, 
        device: torch.device) -> Dict[str, float]:
    """
    测试模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        threshold: 分类阈值
        device: 设备
        
    Returns:
        包含测试指标的字典
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # 获取数据
            tcr_idx = batch['tcr_idx'].to(device)
            pep_idx = batch['pep_idx'].to(device)
            tcr_biochem = batch['tcr_biochem'].to(device) if 'tcr_biochem' in batch else None
            pep_biochem = batch['pep_biochem'].to(device) if 'pep_biochem' in batch else None
            tcr_mask = batch['tcr_mask'].to(device) if 'tcr_mask' in batch else None
            pep_mask = batch['pep_mask'].to(device) if 'pep_mask' in batch else None
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(tcr_idx, pep_idx, tcr_biochem, pep_biochem, tcr_mask, pep_mask)
            preds = outputs['pred']
            
            # 收集预测结果和标签
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    # 合并预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds, threshold=threshold)
    
    # 打印评估报告
    print_metrics_report(metrics)
    
    return metrics


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练TCR-Pep互作预测模型")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="配置文件路径")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="数据目录")
    parser.add_argument("--model_dir", type=str, default="models", help="模型保存目录")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("train_tcr_pep", os.path.join(args.log_dir, "train_tcr_pep.log"))
    
    # 加载配置
    logger.info(f"从{args.config}加载配置...")
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载训练数据...")
    train_df = pd.read_csv(os.path.join(args.data_dir, "tcr_pep_train.csv"))
    logger.info(f"训练数据大小: {train_df.shape}")
    
    logger.info("加载验证数据...")
    val_df = pd.read_csv(os.path.join(args.data_dir, "tcr_pep_val.csv"))
    logger.info(f"验证数据大小: {val_df.shape}")
    
    logger.info("加载测试数据...")
    test_df = pd.read_csv(os.path.join(args.data_dir, "tcr_pep_test.csv"))
    logger.info(f"测试数据大小: {test_df.shape}")
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = MyDataSet(
        train_df, 
        data_type='tcr_pep',
        max_tcr_len=config['data']['max_tcr_len'],
        max_pep_len=config['data']['max_pep_len']
    )
    
    val_dataset = MyDataSet(
        val_df, 
        data_type='tcr_pep',
        max_tcr_len=config['data']['max_tcr_len'],
        max_pep_len=config['data']['max_pep_len']
    )
    
    test_dataset = MyDataSet(
        test_df, 
        data_type='tcr_pep',
        max_tcr_len=config['data']['max_tcr_len'],
        max_pep_len=config['data']['max_pep_len']
    )
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = TCRPepModel(
        vocab_size=22,  # 20种氨基酸 + 填充 + 未知
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        max_tcr_len=config['data']['max_tcr_len'],
        max_pep_len=config['data']['max_pep_len'],
        use_biochem=config['model']['use_biochem_features'],
        dropout=config['model']['dropout'],
        attention_type=config['attention']['fusion_method'],
        sigma=config['attention']['physical_sliding']['sigma'],
        num_iterations=config['attention']['physical_sliding']['num_iterations'],
        fusion_method=config['attention']['fusion_method'],
        fusion_weights=config['attention']['fusion_weights']
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    if config['optimizer']['lr_scheduler']['type'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['optimizer']['lr_scheduler']['factor'],
            patience=config['optimizer']['lr_scheduler']['patience'],
            min_lr=config['optimizer']['lr_scheduler']['min_lr']
        )
    else:
        scheduler = None
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        mode='max',
        delta=0.001,
        save_path=os.path.join(args.model_dir, "tcr_pep_best.pt")
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))
    
    # 训练循环
    logger.info("开始训练...")
    best_val_f1 = 0.0
    best_threshold = 0.5
    
    for epoch in range(config['training']['max_epochs']):
        # 训练
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_metrics, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # 查找最佳阈值
        threshold, _ = find_optimal_threshold(val_labels, val_preds, metric='f1')
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step(val_metrics['f1'])
        
        # 记录指标
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)
        writer.add_scalar("val/threshold", threshold, epoch)
        
        # 打印进度
        logger.info(f"Epoch {epoch+1}/{config['training']['max_epochs']} - "
                   f"Train Loss: {train_metrics['loss']:.4f}, "
                   f"Train F1: {train_metrics['f1']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val F1: {val_metrics['f1']:.4f}, "
                   f"Threshold: {threshold:.4f}")
        
        # 更新最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_threshold = threshold
            logger.info(f"发现更好的模型，F1: {best_val_f1:.4f}, 阈值: {best_threshold:.4f}")
        
        # 早停检查
        if early_stopping(epoch, val_metrics['f1'], model):
            logger.info(f"早停触发，最佳轮次: {early_stopping.best_epoch}")
            break
    
    # 加载最佳模型
    logger.info("加载最佳模型...")
    early_stopping.load_best_model(model)
    
    # 在测试集上评估
    logger.info(f"在测试集上评估，使用阈值: {best_threshold:.4f}...")
    test_metrics = test(model, test_loader, best_threshold, device)
    
    # 记录测试指标
    for key, value in test_metrics.items():
        writer.add_scalar(f"test/{key}", value, 0)
    
    # 保存最终模型和阈值
    final_model_path = os.path.join(args.model_dir, "tcr_pep_final.pt")
    logger.info(f"保存最终模型到{final_model_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'threshold': best_threshold,
        'metrics': test_metrics
    }, final_model_path)
    
    logger.info("训练完成!")
    writer.close()


if __name__ == "__main__":
    main() 