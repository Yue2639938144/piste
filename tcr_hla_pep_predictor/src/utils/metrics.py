"""
评估指标模块

该模块提供模型性能评估和可视化功能，包括ROC曲线、PR曲线和各种分类指标的计算。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix
)


def calculate_metrics(y_true: np.ndarray, 
                     y_prob: np.ndarray, 
                     threshold: float = 0.5,
                     pos_label: int = 1) -> Dict[str, float]:
    """
    计算各种分类性能指标
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        threshold: 分类阈值
        pos_label: 正类标签值
        
    Returns:
        包含各种性能指标的字典
    """
    # 将概率转换为二分类预测
    y_pred = (y_prob >= threshold).astype(int)
    
    # 计算ROC曲线和AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    # 计算PR曲线和AUC
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    pr_auc = auc(recall, precision)
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 计算各种指标
    accuracy = accuracy_score(y_true, y_pred)
    precision_val = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall_val = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # 计算特异性
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 计算阳性预测值和阴性预测值
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    # 返回所有指标
    return {
        'accuracy': accuracy,
        'precision': precision_val,
        'recall': recall_val,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'ppv': ppv,
        'npv': npv,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'threshold': threshold
    }


def plot_roc_curve(y_true: np.ndarray, 
                  y_prob: np.ndarray, 
                  pos_label: int = 1,
                  ax: Optional[plt.Axes] = None,
                  title: str = 'ROC Curve',
                  label: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制ROC曲线
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        pos_label: 正类标签值
        ax: 可选的Matplotlib轴对象
        title: 图表标题
        label: 曲线标签
        
    Returns:
        Figure和Axes对象元组
    """
    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    # 创建图表
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # 绘制ROC曲线
    if label is None:
        label = f'ROC curve (AUC = {roc_auc:.3f})'
    ax.plot(fpr, tpr, lw=2, label=label)
    
    # 绘制对角线（随机猜测）
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    
    return fig, ax


def plot_pr_curve(y_true: np.ndarray, 
                 y_prob: np.ndarray, 
                 pos_label: int = 1,
                 ax: Optional[plt.Axes] = None,
                 title: str = 'Precision-Recall Curve',
                 label: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制精确率-召回率曲线
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        pos_label: 正类标签值
        ax: 可选的Matplotlib轴对象
        title: 图表标题
        label: 曲线标签
        
    Returns:
        Figure和Axes对象元组
    """
    # 计算PR曲线
    precision, recall, _ = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    pr_auc = auc(recall, precision)
    
    # 计算随机猜测的基准线（正例比例）
    baseline = np.sum(y_true == pos_label) / len(y_true)
    
    # 创建图表
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # 绘制PR曲线
    if label is None:
        label = f'PR curve (AUC = {pr_auc:.3f})'
    ax.plot(recall, precision, lw=2, label=label)
    
    # 绘制基准线（随机猜测）
    ax.plot([0, 1], [baseline, baseline], 'k--', lw=2, label=f'Baseline ({baseline:.3f})')
    
    # 设置图表属性
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    
    return fig, ax


def print_metrics_report(metrics: Dict[str, float]) -> None:
    """
    打印性能指标报告
    
    Args:
        metrics: 包含性能指标的字典
    """
    print("\n" + "="*50)
    print("性能评估报告")
    print("="*50)
    print(f"阈值: {metrics['threshold']:.4f}")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"特异性 (Specificity): {metrics['specificity']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"PR AUC: {metrics['pr_auc']:.4f}")
    print("-"*50)
    print("混淆矩阵:")
    print(f"真正例 (TP): {metrics['tp']}")
    print(f"假正例 (FP): {metrics['fp']}")
    print(f"真负例 (TN): {metrics['tn']}")
    print(f"假负例 (FN): {metrics['fn']}")
    print("="*50) 