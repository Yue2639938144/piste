#!/usr/bin/env python
"""
TCR-HLA-Pep模型评估脚本

该脚本用于对模型进行全面评估，包括计算各种评估指标和生成评估报告。
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from typing import Dict, List, Tuple, Union, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataloader import MyDataSet, DataLoader
from src.models.trimer_model import TrimerModel
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.metrics import calculate_metrics, print_metrics_report
from src.utils.threshold_finder import find_optimal_threshold


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


def evaluate(model: TrimerModel, 
            dataloader: DataLoader, 
            device: torch.device) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    评估模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        包含评估指标的字典，预测概率数组和标签数组
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_tcr_pep_preds = []
    all_hla_pep_preds = []
    
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
            all_tcr_pep_preds.append(outputs['tcr_pep_pred'].detach().cpu().numpy())
            all_hla_pep_preds.append(outputs['hla_pep_pred'].detach().cpu().numpy())
    
    # 合并预测结果和标签
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_tcr_pep_preds = np.concatenate(all_tcr_pep_preds)
    all_hla_pep_preds = np.concatenate(all_hla_pep_preds)
    
    # 计算评估指标
    metrics = calculate_metrics(all_labels, all_preds)
    tcr_pep_metrics = calculate_metrics(all_labels, all_tcr_pep_preds)
    hla_pep_metrics = calculate_metrics(all_labels, all_hla_pep_preds)
    
    # 合并指标
    metrics.update({
        'tcr_pep_accuracy': tcr_pep_metrics['accuracy'],
        'tcr_pep_precision': tcr_pep_metrics['precision'],
        'tcr_pep_recall': tcr_pep_metrics['recall'],
        'tcr_pep_f1': tcr_pep_metrics['f1'],
        'tcr_pep_roc_auc': tcr_pep_metrics['roc_auc'],
        'tcr_pep_pr_auc': tcr_pep_metrics['pr_auc'],
        'hla_pep_accuracy': hla_pep_metrics['accuracy'],
        'hla_pep_precision': hla_pep_metrics['precision'],
        'hla_pep_recall': hla_pep_metrics['recall'],
        'hla_pep_f1': hla_pep_metrics['f1'],
        'hla_pep_roc_auc': hla_pep_metrics['roc_auc'],
        'hla_pep_pr_auc': hla_pep_metrics['pr_auc']
    })
    
    return metrics, all_preds, all_labels, all_tcr_pep_preds, all_hla_pep_preds


def plot_roc_curve(labels: np.ndarray, 
                  preds: np.ndarray, 
                  tcr_pep_preds: np.ndarray, 
                  hla_pep_preds: np.ndarray, 
                  output_path: str):
    """
    绘制ROC曲线
    
    Args:
        labels: 标签
        preds: 预测概率
        tcr_pep_preds: TCR-Pep预测概率
        hla_pep_preds: HLA-Pep预测概率
        output_path: 输出路径
    """
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(labels, preds)
    tcr_pep_fpr, tcr_pep_tpr, _ = roc_curve(labels, tcr_pep_preds)
    hla_pep_fpr, hla_pep_tpr, _ = roc_curve(labels, hla_pep_preds)
    
    # 计算AUC
    roc_auc = auc(fpr, tpr)
    tcr_pep_roc_auc = auc(tcr_pep_fpr, tcr_pep_tpr)
    hla_pep_roc_auc = auc(hla_pep_fpr, hla_pep_tpr)
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, label=f'三元模型 (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot(tcr_pep_fpr, tcr_pep_tpr, label=f'TCR-Pep模型 (AUC = {tcr_pep_roc_auc:.3f})', linewidth=2, linestyle='--')
    plt.plot(hla_pep_fpr, hla_pep_tpr, label=f'HLA-Pep模型 (AUC = {hla_pep_roc_auc:.3f})', linewidth=2, linestyle='-.')
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('接收者操作特征曲线 (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curve(labels: np.ndarray, 
                 preds: np.ndarray, 
                 tcr_pep_preds: np.ndarray, 
                 hla_pep_preds: np.ndarray, 
                 output_path: str):
    """
    绘制PR曲线
    
    Args:
        labels: 标签
        preds: 预测概率
        tcr_pep_preds: TCR-Pep预测概率
        hla_pep_preds: HLA-Pep预测概率
        output_path: 输出路径
    """
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(labels, preds)
    tcr_pep_precision, tcr_pep_recall, _ = precision_recall_curve(labels, tcr_pep_preds)
    hla_pep_precision, hla_pep_recall, _ = precision_recall_curve(labels, hla_pep_preds)
    
    # 计算AUC
    pr_auc = auc(recall, precision)
    tcr_pep_pr_auc = auc(tcr_pep_recall, tcr_pep_precision)
    hla_pep_pr_auc = auc(hla_pep_recall, hla_pep_precision)
    
    # 创建图表
    plt.figure(figsize=(10, 8))
    
    # 绘制PR曲线
    plt.plot(recall, precision, label=f'三元模型 (AUC = {pr_auc:.3f})', linewidth=2)
    plt.plot(tcr_pep_recall, tcr_pep_precision, label=f'TCR-Pep模型 (AUC = {tcr_pep_pr_auc:.3f})', linewidth=2, linestyle='--')
    plt.plot(hla_pep_recall, hla_pep_precision, label=f'HLA-Pep模型 (AUC = {hla_pep_pr_auc:.3f})', linewidth=2, linestyle='-.')
    
    # 计算随机猜测的基准线（正例比例）
    baseline = np.sum(labels) / len(labels)
    plt.plot([0, 1], [baseline, baseline], 'k--', label=f'随机猜测 (AUC = {baseline:.3f})')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线 (PR)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels: np.ndarray, 
                         binary_preds: np.ndarray, 
                         output_path: str):
    """
    绘制混淆矩阵
    
    Args:
        labels: 标签
        binary_preds: 二分类预测结果
        output_path: 输出路径
    """
    # 计算混淆矩阵
    cm = confusion_matrix(labels, binary_preds)
    
    # 创建图表
    plt.figure(figsize=(8, 6))
    
    # 绘制混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    
    # 设置图表属性
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.xticks([0.5, 1.5], ['阴性', '阳性'])
    plt.yticks([0.5, 1.5], ['阴性', '阳性'])
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_metrics(labels: np.ndarray, 
                          preds: np.ndarray, 
                          output_path: str):
    """
    绘制不同阈值下的指标变化
    
    Args:
        labels: 标签
        preds: 预测概率
        output_path: 输出路径
    """
    # 设置阈值范围
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    # 计算不同阈值下的指标
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        binary_preds = (preds >= threshold).astype(int)
        metrics = calculate_metrics(labels, preds, threshold=threshold)
        accuracies.append(metrics['accuracy'])
        precisions.append(metrics['precision'])
        recalls.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制指标曲线
    plt.plot(thresholds, accuracies, label='准确率', linewidth=2)
    plt.plot(thresholds, precisions, label='精确率', linewidth=2)
    plt.plot(thresholds, recalls, label='召回率', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1分数', linewidth=2)
    
    # 找到F1分数最大的阈值
    best_threshold, best_f1 = find_optimal_threshold(labels, preds, metric='f1')
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'最佳阈值 = {best_threshold:.2f}')
    
    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('阈值')
    plt.ylabel('指标值')
    plt.title('不同阈值下的评估指标变化')
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    # 保存图表
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_report(metrics: Dict[str, float], 
                          best_threshold: float, 
                          output_path: str):
    """
    保存评估报告
    
    Args:
        metrics: 评估指标字典
        best_threshold: 最佳阈值
        output_path: 输出路径
    """
    # 创建报告内容
    report = f"""# TCR-HLA-Pep模型评估报告

## 三元模型性能指标
- 准确率: {metrics['accuracy']:.4f}
- 精确率: {metrics['precision']:.4f}
- 召回率: {metrics['recall']:.4f}
- F1分数: {metrics['f1']:.4f}
- ROC-AUC: {metrics['roc_auc']:.4f}
- PR-AUC: {metrics['pr_auc']:.4f}
- MCC: {metrics['mcc']:.4f}
- 最佳阈值: {best_threshold:.4f}

## TCR-Pep模型性能指标
- 准确率: {metrics['tcr_pep_accuracy']:.4f}
- 精确率: {metrics['tcr_pep_precision']:.4f}
- 召回率: {metrics['tcr_pep_recall']:.4f}
- F1分数: {metrics['tcr_pep_f1']:.4f}
- ROC-AUC: {metrics['tcr_pep_roc_auc']:.4f}
- PR-AUC: {metrics['tcr_pep_pr_auc']:.4f}

## HLA-Pep模型性能指标
- 准确率: {metrics['hla_pep_accuracy']:.4f}
- 精确率: {metrics['hla_pep_precision']:.4f}
- 召回率: {metrics['hla_pep_recall']:.4f}
- F1分数: {metrics['hla_pep_f1']:.4f}
- ROC-AUC: {metrics['hla_pep_roc_auc']:.4f}
- PR-AUC: {metrics['hla_pep_pr_auc']:.4f}

## 模型比较
- 三元模型相对于TCR-Pep模型的F1提升: {(metrics['f1'] - metrics['tcr_pep_f1']) / metrics['tcr_pep_f1'] * 100:.2f}%
- 三元模型相对于HLA-Pep模型的F1提升: {(metrics['f1'] - metrics['hla_pep_f1']) / metrics['hla_pep_f1'] * 100:.2f}%
- 三元模型相对于TCR-Pep模型的AUC提升: {(metrics['roc_auc'] - metrics['tcr_pep_roc_auc']) / metrics['tcr_pep_roc_auc'] * 100:.2f}%
- 三元模型相对于HLA-Pep模型的AUC提升: {(metrics['roc_auc'] - metrics['hla_pep_roc_auc']) / metrics['hla_pep_roc_auc'] * 100:.2f}%
"""
    
    # 保存报告
    with open(output_path, 'w') as f:
        f.write(report)


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="TCR-HLA-Pep模型评估")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("evaluate", os.path.join(args.output_dir, "evaluate.log"))
    
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
    
    # 加载测试数据
    logger.info(f"从{args.test_data}加载测试数据...")
    test_df = pd.read_csv(args.test_data)
    logger.info(f"测试数据大小: {test_df.shape}")
    
    # 创建数据集
    logger.info("创建数据集...")
    test_dataset = MyDataSet(
        test_df, 
        data_type='trimer',
        max_tcr_len=config['data']['max_tcr_len'],
        max_hla_len=config['data']['max_hla_len'],
        max_pep_len=config['data']['max_pep_len']
    )
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 评估模型
    logger.info("评估模型...")
    metrics, preds, labels, tcr_pep_preds, hla_pep_preds = evaluate(model, test_loader, device)
    
    # 打印评估指标
    logger.info("三元模型评估指标:")
    print_metrics_report(metrics)
    
    # 查找最佳阈值
    logger.info("查找最佳阈值...")
    best_threshold, best_f1 = find_optimal_threshold(labels, preds, metric='f1')
    logger.info(f"最佳阈值: {best_threshold:.4f}, F1: {best_f1:.4f}")
    
    # 使用最佳阈值计算二分类预测
    binary_preds = (preds >= best_threshold).astype(int)
    
    # 绘制ROC曲线
    logger.info("绘制ROC曲线...")
    roc_path = os.path.join(args.output_dir, "roc_curve.png")
    plot_roc_curve(labels, preds, tcr_pep_preds, hla_pep_preds, roc_path)
    
    # 绘制PR曲线
    logger.info("绘制PR曲线...")
    pr_path = os.path.join(args.output_dir, "pr_curve.png")
    plot_pr_curve(labels, preds, tcr_pep_preds, hla_pep_preds, pr_path)
    
    # 绘制混淆矩阵
    logger.info("绘制混淆矩阵...")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(labels, binary_preds, cm_path)
    
    # 绘制阈值-指标曲线
    logger.info("绘制阈值-指标曲线...")
    threshold_path = os.path.join(args.output_dir, "threshold_metrics.png")
    plot_threshold_metrics(labels, preds, threshold_path)
    
    # 保存评估报告
    logger.info("保存评估报告...")
    report_path = os.path.join(args.output_dir, "evaluation_report.md")
    save_evaluation_report(metrics, best_threshold, report_path)
    
    logger.info("评估完成!")


if __name__ == "__main__":
    main() 