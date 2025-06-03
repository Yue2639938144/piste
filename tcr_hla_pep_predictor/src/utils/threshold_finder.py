"""
最佳阈值筛选模块

该模块提供基于验证集性能确定最优分类阈值的功能。
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, matthews_corrcoef


def find_optimal_threshold(y_true: np.ndarray, 
                           y_prob: np.ndarray, 
                           metric: str = 'f1',
                           pos_label: int = 1) -> Tuple[float, float]:
    """
    基于指定指标找到最优分类阈值
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        metric: 优化指标，可选值为'f1'、'mcc'、'youden'、'precision'或'recall'
        pos_label: 正类标签值
        
    Returns:
        最优阈值和对应的指标值
    """
    if metric.lower() == 'f1':
        return find_optimal_f1_threshold(y_true, y_prob, pos_label)
    elif metric.lower() == 'mcc':
        return find_optimal_mcc_threshold(y_true, y_prob, pos_label)
    elif metric.lower() == 'youden':
        return find_optimal_youden_threshold(y_true, y_prob, pos_label)
    elif metric.lower() == 'precision':
        return find_optimal_precision_threshold(y_true, y_prob, pos_label)
    elif metric.lower() == 'recall':
        return find_optimal_recall_threshold(y_true, y_prob, pos_label)
    else:
        raise ValueError(f"不支持的指标: {metric}，支持的指标有'f1'、'mcc'、'youden'、'precision'和'recall'")


def find_optimal_f1_threshold(y_true: np.ndarray, 
                             y_prob: np.ndarray, 
                             pos_label: int = 1) -> Tuple[float, float]:
    """
    找到使F1分数最大的阈值
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        pos_label: 正类标签值
        
    Returns:
        最优阈值和对应的F1分数
    """
    # 获取不同阈值下的精确率和召回率
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    
    # 计算每个阈值对应的F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 找到F1分数最大的索引
    optimal_idx = np.argmax(f1_scores)
    
    # 获取最优阈值和对应的F1分数
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1


def find_optimal_mcc_threshold(y_true: np.ndarray, 
                              y_prob: np.ndarray, 
                              pos_label: int = 1) -> Tuple[float, float]:
    """
    找到使Matthews相关系数最大的阈值
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        pos_label: 正类标签值
        
    Returns:
        最优阈值和对应的MCC值
    """
    # 获取ROC曲线的阈值
    _, _, thresholds = roc_curve(y_true, y_prob, pos_label=pos_label)
    
    # 添加0和1作为极端阈值
    thresholds = np.append(thresholds, [0.0, 1.0])
    thresholds = np.unique(thresholds)
    
    # 计算每个阈值对应的MCC值
    mcc_scores = []
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        mcc_scores.append(mcc)
    
    # 找到MCC最大的索引
    optimal_idx = np.argmax(mcc_scores)
    
    # 获取最优阈值和对应的MCC值
    optimal_threshold = thresholds[optimal_idx]
    optimal_mcc = mcc_scores[optimal_idx]
    
    return optimal_threshold, optimal_mcc


def find_optimal_youden_threshold(y_true: np.ndarray, 
                                 y_prob: np.ndarray, 
                                 pos_label: int = 1) -> Tuple[float, float]:
    """
    找到使Youden指数最大的阈值（敏感性+特异性-1）
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        pos_label: 正类标签值
        
    Returns:
        最优阈值和对应的Youden指数
    """
    # 获取ROC曲线的假正率、真正率和阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=pos_label)
    
    # 计算Youden指数（敏感性+特异性-1）
    youden_index = tpr - fpr
    
    # 找到Youden指数最大的索引
    optimal_idx = np.argmax(youden_index)
    
    # 获取最优阈值和对应的Youden指数
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden_index[optimal_idx]
    
    return optimal_threshold, optimal_youden


def find_optimal_precision_threshold(y_true: np.ndarray, 
                                    y_prob: np.ndarray, 
                                    pos_label: int = 1,
                                    min_recall: float = 0.5) -> Tuple[float, float]:
    """
    找到使精确率最大且召回率不低于指定值的阈值
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        pos_label: 正类标签值
        min_recall: 最小召回率要求
        
    Returns:
        最优阈值和对应的精确率
    """
    # 获取不同阈值下的精确率和召回率
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    
    # 找到满足最小召回率要求的索引
    valid_indices = np.where(recall >= min_recall)[0]
    
    if len(valid_indices) == 0:
        # 如果没有满足最小召回率要求的阈值，则选择召回率最高的阈值
        optimal_idx = np.argmax(recall)
    else:
        # 在满足最小召回率要求的阈值中，选择精确率最高的
        valid_precision = precision[valid_indices]
        optimal_sub_idx = np.argmax(valid_precision)
        optimal_idx = valid_indices[optimal_sub_idx]
    
    # 获取最优阈值和对应的精确率
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_precision = precision[optimal_idx]
    
    return optimal_threshold, optimal_precision


def find_optimal_recall_threshold(y_true: np.ndarray, 
                                 y_prob: np.ndarray, 
                                 pos_label: int = 1,
                                 min_precision: float = 0.5) -> Tuple[float, float]:
    """
    找到使召回率最大且精确率不低于指定值的阈值
    
    Args:
        y_true: 真实标签数组
        y_prob: 预测概率数组
        pos_label: 正类标签值
        min_precision: 最小精确率要求
        
    Returns:
        最优阈值和对应的召回率
    """
    # 获取不同阈值下的精确率和召回率
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=pos_label)
    
    # 找到满足最小精确率要求的索引
    valid_indices = np.where(precision >= min_precision)[0]
    
    if len(valid_indices) == 0:
        # 如果没有满足最小精确率要求的阈值，则选择精确率最高的阈值
        optimal_idx = np.argmax(precision)
    else:
        # 在满足最小精确率要求的阈值中，选择召回率最高的
        valid_recall = recall[valid_indices]
        optimal_sub_idx = np.argmax(valid_recall)
        optimal_idx = valid_indices[optimal_sub_idx]
    
    # 获取最优阈值和对应的召回率
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    optimal_recall = recall[optimal_idx]
    
    return optimal_threshold, optimal_recall 