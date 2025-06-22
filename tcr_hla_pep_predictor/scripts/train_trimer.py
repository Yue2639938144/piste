#!/usr/bin/env python
"""
TCR-HLA-Pep三元模型训练脚本

该脚本用于训练TCR-HLA-Pep三元互作预测模型，支持联合优化策略。
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
from src.models.trimer_model import TrimerModel
from src.models.tcr_pep_model import TCRPepModel
from src.models.hla_pep_model import HLAPepModel
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.early_stopping import EarlyStopping
from src.utils.threshold_finder import find_optimal_threshold
from src.utils.metrics import calculate_metrics, print_metrics_report


class JointLoss(nn.Module):
    """
    联合损失函数
    
    结合三元模型损失和二元模型损失的加权损失函数。
    """
    
    def __init__(self, 
                 trimer_weight: float = 1.0, 
                 tcr_pep_weight: float = 0.3, 
                 hla_pep_weight: float = 0.3):
        """
        初始化联合损失函数
        
        Args:
            trimer_weight: 三元模型损失权重
            tcr_pep_weight: TCR-Pep模型损失权重
            hla_pep_weight: HLA-Pep模型损失权重
        """
        super().__init__()
        self.trimer_weight = trimer_weight
        self.tcr_pep_weight = tcr_pep_weight
        self.hla_pep_weight = hla_pep_weight
        self.criterion = nn.BCELoss(reduction='none')
    
    def forward(self, 
               outputs: Dict[str, torch.Tensor], 
               labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播
        
        Args:
            outputs: 模型输出字典，包含三元和二元预测结果
            labels: 标签
            
        Returns:
            加权总损失和各部分损失的字典
        """
        # 计算各部分损失
        trimer_loss = self.criterion(outputs['pred'], labels).mean()
        tcr_pep_loss = self.criterion(outputs['tcr_pep_pred'], labels).mean()
        hla_pep_loss = self.criterion(outputs['hla_pep_pred'], labels).mean()
        
        # 计算加权总损失
        total_loss = (self.trimer_weight * trimer_loss + 
                      self.tcr_pep_weight * tcr_pep_loss + 
                      self.hla_pep_weight * hla_pep_loss)
        
        # 返回总损失和各部分损失
        loss_dict = {
            'total': total_loss,
            'trimer': trimer_loss,
            'tcr_pep': tcr_pep_loss,
            'hla_pep': hla_pep_loss
        }
        
        return total_loss, loss_dict


def train_epoch(model: nn.Module, 
               dataloader: DataLoader, 
               joint_loss: JointLoss, 
               optimizer: optim.Optimizer, 
               device: torch.device) -> Dict[str, float]:
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        joint_loss: 联合损失函数
        optimizer: 优化器
        device: 设备
        
    Returns:
        包含训练指标的字典
    """
    model.train()
    total_loss = 0.0
    total_trimer_loss = 0.0
    total_tcr_pep_loss = 0.0
    total_hla_pep_loss = 0.0
    all_preds = []
    all_labels = []
    
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
        labels = batch['label'].to(device)
        
        # 前向传播
        outputs = model(
            tcr_idx, pep_idx, hla_idx,
            tcr_biochem, pep_biochem, hla_biochem,
            tcr_mask, pep_mask, hla_mask
        )
        
        # 计算损失
        loss, loss_dict = joint_loss(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        total_trimer_loss += loss_dict['trimer'].item()
        total_tcr_pep_loss += loss_dict['tcr_pep'].item()
        total_hla_pep_loss += loss_dict['hla_pep'].item()
        
        # 收集预测结果和标签
        all_preds.append(outputs['pred'].detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_trimer_loss = total_trimer_loss / len(dataloader)
    avg_tcr_pep_loss = total_tcr_pep_loss / len(dataloader)
    avg_hla_pep_loss = total_hla_pep_loss / len(dataloader)
    
    # 合并预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 返回训练指标
    return {
        'loss': avg_loss,
        'trimer_loss': avg_trimer_loss,
        'tcr_pep_loss': avg_tcr_pep_loss,
        'hla_pep_loss': avg_hla_pep_loss,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc']
    }


def validate(model: nn.Module, 
            dataloader: DataLoader, 
            joint_loss: JointLoss, 
            device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        joint_loss: 联合损失函数
        device: 设备
        
    Returns:
        包含验证指标的字典，预测概率数组和标签数组
    """
    model.eval()
    total_loss = 0.0
    total_trimer_loss = 0.0
    total_tcr_pep_loss = 0.0
    total_hla_pep_loss = 0.0
    all_preds = []
    all_labels = []
    
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
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(
                tcr_idx, pep_idx, hla_idx,
                tcr_biochem, pep_biochem, hla_biochem,
                tcr_mask, pep_mask, hla_mask
            )
            
            # 计算损失
            loss, loss_dict = joint_loss(outputs, labels)
            
            # 累计损失
            total_loss += loss.item()
            total_trimer_loss += loss_dict['trimer'].item()
            total_tcr_pep_loss += loss_dict['tcr_pep'].item()
            total_hla_pep_loss += loss_dict['hla_pep'].item()
            
            # 收集预测结果和标签
            all_preds.append(outputs['pred'].detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_trimer_loss = total_trimer_loss / len(dataloader)
    avg_tcr_pep_loss = total_tcr_pep_loss / len(dataloader)
    avg_hla_pep_loss = total_hla_pep_loss / len(dataloader)
    
    # 合并预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 返回验证指标、预测概率和标签
    return {
        'loss': avg_loss,
        'trimer_loss': avg_trimer_loss,
        'tcr_pep_loss': avg_tcr_pep_loss,
        'hla_pep_loss': avg_hla_pep_loss,
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
            hla_idx = batch['hla_idx'].to(device)
            tcr_biochem = batch['tcr_biochem'].to(device) if 'tcr_biochem' in batch else None
            pep_biochem = batch['pep_biochem'].to(device) if 'pep_biochem' in batch else None
            hla_biochem = batch['hla_biochem'].to(device) if 'hla_biochem' in batch else None
            tcr_mask = batch['tcr_mask'].to(device) if 'tcr_mask' in batch else None
            pep_mask = batch['pep_mask'].to(device) if 'pep_mask' in batch else None
            hla_mask = batch['hla_mask'].to(device) if 'hla_mask' in batch else None
            labels = batch['label'].to(device)
            
            # 前向传播
            outputs = model(
                tcr_idx, pep_idx, hla_idx,
                tcr_biochem, pep_biochem, hla_biochem,
                tcr_mask, pep_mask, hla_mask
            )
            
            # 收集预测结果和标签
            all_preds.append(outputs['pred'].detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    # 合并预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds, threshold=threshold)
    
    # 打印评估报告
    print_metrics_report(metrics)
    
    return metrics


def load_binary_models(config: Dict, device: torch.device) -> Tuple[TCRPepModel, HLAPepModel]:
    """
    加载预训练的二元模型
    
    Args:
        config: 配置字典
        device: 设备
        
    Returns:
        TCR-Pep模型和HLA-Pep模型的元组
    """
    # 加载TCR-Pep模型
    tcr_pep_path = config['model']['tcr_pep_model_path']
    if os.path.exists(tcr_pep_path):
        tcr_pep_checkpoint = torch.load(tcr_pep_path, map_location=device)
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
            attention_type=config['attention']['type'],
            sigma=config['attention']['physical_sliding']['sigma'],
            num_iterations=config['attention']['physical_sliding']['num_iterations'],
            fusion_method=config['attention']['fusion_method'],
            fusion_weights=config['attention']['fusion_weights']
        ).to(device)
        tcr_pep_model.load_state_dict(tcr_pep_checkpoint['model_state_dict'])
    else:
        # 如果没有预训练模型，创建新模型
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
            attention_type=config['attention']['type'],
            sigma=config['attention']['physical_sliding']['sigma'],
            num_iterations=config['attention']['physical_sliding']['num_iterations'],
            fusion_method=config['attention']['fusion_method'],
            fusion_weights=config['attention']['fusion_weights']
        ).to(device)
    
    # 加载HLA-Pep模型
    hla_pep_path = config['model']['hla_pep_model_path']
    if os.path.exists(hla_pep_path):
        hla_pep_checkpoint = torch.load(hla_pep_path, map_location=device)
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
            attention_type=config['attention']['type'],
            sigma=config['attention']['physical_sliding']['sigma'],
            num_iterations=config['attention']['physical_sliding']['num_iterations'],
            fusion_method=config['attention']['fusion_method'],
            fusion_weights=config['attention']['fusion_weights']
        ).to(device)
        hla_pep_model.load_state_dict(hla_pep_checkpoint['model_state_dict'])
    else:
        # 如果没有预训练模型，创建新模型
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
            attention_type=config['attention']['type'],
            sigma=config['attention']['physical_sliding']['sigma'],
            num_iterations=config['attention']['physical_sliding']['num_iterations'],
            fusion_method=config['attention']['fusion_method'],
            fusion_weights=config['attention']['fusion_weights']
        ).to(device)
    
    return tcr_pep_model, hla_pep_model


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练TCR-HLA-Pep三元互作预测模型")
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
    logger = setup_logger("train_trimer", os.path.join(args.log_dir, "train_trimer.log"))
    
    # 加载配置
    logger.info(f"从{args.config}加载配置...")
    config = load_config(args.config)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载训练数据...")
    train_df = pd.read_csv(os.path.join(args.data_dir, "trimer_train.csv"))
    logger.info(f"训练数据大小: {train_df.shape}")
    
    logger.info("加载验证数据...")
    val_df = pd.read_csv(os.path.join(args.data_dir, "trimer_val.csv"))
    logger.info(f"验证数据大小: {val_df.shape}")
    
    logger.info("加载测试数据...")
    test_df = pd.read_csv(os.path.join(args.data_dir, "trimer_test.csv"))
    logger.info(f"测试数据大小: {test_df.shape}")
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = MyDataSet(
        train_df, 
        data_type='trimer',
        max_tcr_len=config['data']['max_tcr_len'],
        max_hla_len=config['data']['max_hla_len'],
        max_pep_len=config['data']['max_pep_len']
    )
    
    val_dataset = MyDataSet(
        val_df, 
        data_type='trimer',
        max_tcr_len=config['data']['max_tcr_len'],
        max_hla_len=config['data']['max_hla_len'],
        max_pep_len=config['data']['max_pep_len']
    )
    
    test_dataset = MyDataSet(
        test_df, 
        data_type='trimer',
        max_tcr_len=config['data']['max_tcr_len'],
        max_hla_len=config['data']['max_hla_len'],
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
    
    # 加载预训练的二元模型
    logger.info("加载预训练的二元模型...")
    tcr_pep_model, hla_pep_model = load_binary_models(config, device)
    
    # 创建三元模型
    logger.info("创建三元模型...")
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
        attention_type=config['attention']['type'],
        sigma=config['attention']['physical_sliding']['sigma'],
        num_iterations=config['attention']['physical_sliding']['num_iterations'],
        fusion_method=config['attention']['fusion_method'],
        fusion_weights=config['attention']['fusion_weights'],
        tcr_pep_model=tcr_pep_model,
        hla_pep_model=hla_pep_model,
        joint_training=config['training']['joint_training']
    ).to(device)
    
    # 定义联合损失函数和优化器
    joint_loss = JointLoss(
        trimer_weight=config['training']['loss_weights']['trimer'],
        tcr_pep_weight=config['training']['loss_weights']['tcr_pep'],
        hla_pep_weight=config['training']['loss_weights']['hla_pep']
    )
    
    # 循环联合优化策略
    training_phases = config['training']['phases']
    current_phase = 0
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=config['training']['patience'],
        mode='max',
        delta=0.001,
        save_path=os.path.join(args.model_dir, "trimer_best.pt")
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tensorboard"))
    
    # 训练循环
    logger.info("开始训练...")
    best_val_f1 = 0.0
    best_threshold = 0.5
    global_epoch = 0
    
    # 循环训练各个阶段
    for phase_idx, phase in enumerate(training_phases):
        logger.info(f"开始训练阶段 {phase_idx+1}/{len(training_phases)}: {phase['name']}")
        
        # 设置参数冻结状态
        if phase['freeze_binary_models']:
            logger.info("冻结二元模型参数...")
            model.freeze_binary_models()
        else:
            logger.info("解冻二元模型参数...")
            model.unfreeze_binary_models()
        
        # 设置优化器
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=phase['lr'],
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
        
        # 阶段训练循环
        for epoch in range(phase['epochs']):
            # 训练
            train_metrics = train_epoch(model, train_loader, joint_loss, optimizer, device)
            
            # 验证
            val_metrics, val_preds, val_labels = validate(model, val_loader, joint_loss, device)
            
            # 查找最佳阈值
            threshold, _ = find_optimal_threshold(val_labels, val_preds, metric='f1')
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step(val_metrics['f1'])
            
            # 记录指标
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, global_epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, global_epoch)
            writer.add_scalar("val/threshold", threshold, global_epoch)
            
            # 打印进度
            logger.info(f"Phase {phase_idx+1} - Epoch {epoch+1}/{phase['epochs']} "
                       f"(Global {global_epoch+1}) - "
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
            if early_stopping(global_epoch, val_metrics['f1'], model):
                logger.info(f"早停触发，最佳轮次: {early_stopping.best_epoch}")
                break
            
            global_epoch += 1
    
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
    final_model_path = os.path.join(args.model_dir, "trimer_final.pt")
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