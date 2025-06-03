"""
数据处理脚本

该脚本用于从原始数据中提取TCR-Pep、HLA-Pep和TCR-HLA-Pep数据，并进行预处理。
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from sklearn.model_selection import train_test_split
import logging
import argparse
import sys
import yaml

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.clustering import sequence_similarity_clustering, split_by_clusters
from src.utils.config import load_config, save_config, get_default_config
from src.utils.logger import setup_logger


def load_raw_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    加载原始数据
    
    Args:
        data_dir: 原始数据目录
        
    Returns:
        包含不同数据集的字典
    """
    logger = logging.getLogger("data_processor")
    
    # 定义数据文件路径
    train_data_path = os.path.join(data_dir, "train_data.csv")
    val_data_path = os.path.join(data_dir, "val_data.csv")
    test_data_path = os.path.join(data_dir, "test_data.csv")
    hla_seq_path = os.path.join(data_dir, "common_hla_sequence.csv")
    
    # 加载数据
    logger.info("加载训练数据...")
    train_df = pd.read_csv(train_data_path)
    logger.info(f"训练数据大小: {train_df.shape}")
    
    logger.info("加载验证数据...")
    val_df = pd.read_csv(val_data_path)
    logger.info(f"验证数据大小: {val_df.shape}")
    
    logger.info("加载测试数据...")
    test_df = pd.read_csv(test_data_path)
    logger.info(f"测试数据大小: {test_df.shape}")
    
    logger.info("加载HLA序列数据...")
    hla_seq_df = pd.read_csv(hla_seq_path)
    logger.info(f"HLA序列数据大小: {hla_seq_df.shape}")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "hla_seq": hla_seq_df
    }


def process_data(data: Dict[str, pd.DataFrame], output_dir: str, config: Dict) -> None:
    """
    处理数据，提取TCR-Pep、HLA-Pep和TCR-HLA-Pep数据
    
    Args:
        data: 包含不同数据集的字典
        output_dir: 输出目录
        config: 配置字典
    """
    logger = logging.getLogger("data_processor")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并HLA序列信息
    logger.info("合并HLA序列信息...")
    hla_seq_dict = dict(zip(data["hla_seq"]["HLA"], data["hla_seq"]["pseudo_sequence"]))
    
    # 处理训练数据
    logger.info("处理训练数据...")
    train_df = process_dataset(data["train"], hla_seq_dict, "train", config)
    
    # 处理验证数据
    logger.info("处理验证数据...")
    val_df = process_dataset(data["val"], hla_seq_dict, "val", config)
    
    # 处理测试数据
    logger.info("处理测试数据...")
    test_df = process_dataset(data["test"], hla_seq_dict, "test", config)
    
    # 提取不同类型的数据
    logger.info("提取TCR-Pep数据...")
    tcr_pep_train = train_df[["CDR3", "MT_pep", "Label"]].copy()
    tcr_pep_val = val_df[["CDR3", "MT_pep", "Label"]].copy()
    tcr_pep_test = test_df[["CDR3", "MT_pep", "Label"]].copy()
    
    logger.info("提取HLA-Pep数据...")
    hla_pep_train = train_df[["HLA_sequence", "MT_pep", "Label"]].copy()
    hla_pep_val = val_df[["HLA_sequence", "MT_pep", "Label"]].copy()
    hla_pep_test = test_df[["HLA_sequence", "MT_pep", "Label"]].copy()
    
    logger.info("提取TCR-HLA-Pep数据...")
    trimer_train = train_df[["CDR3", "MT_pep", "HLA_sequence", "Label"]].copy()
    trimer_val = val_df[["CDR3", "MT_pep", "HLA_sequence", "Label"]].copy()
    trimer_test = test_df[["CDR3", "MT_pep", "HLA_sequence", "Label"]].copy()
    
    # 保存处理后的数据
    logger.info("保存处理后的数据...")
    tcr_pep_train.to_csv(os.path.join(output_dir, "tcr_pep_train.csv"), index=False)
    tcr_pep_val.to_csv(os.path.join(output_dir, "tcr_pep_val.csv"), index=False)
    tcr_pep_test.to_csv(os.path.join(output_dir, "tcr_pep_test.csv"), index=False)
    
    hla_pep_train.to_csv(os.path.join(output_dir, "hla_pep_train.csv"), index=False)
    hla_pep_val.to_csv(os.path.join(output_dir, "hla_pep_val.csv"), index=False)
    hla_pep_test.to_csv(os.path.join(output_dir, "hla_pep_test.csv"), index=False)
    
    trimer_train.to_csv(os.path.join(output_dir, "trimer_train.csv"), index=False)
    trimer_val.to_csv(os.path.join(output_dir, "trimer_val.csv"), index=False)
    trimer_test.to_csv(os.path.join(output_dir, "trimer_test.csv"), index=False)
    
    # 打印数据统计信息
    logger.info("\n数据统计信息:")
    logger.info(f"TCR-Pep训练集: {tcr_pep_train.shape[0]}样本，正例比例: {tcr_pep_train['Label'].mean():.4f}")
    logger.info(f"TCR-Pep验证集: {tcr_pep_val.shape[0]}样本，正例比例: {tcr_pep_val['Label'].mean():.4f}")
    logger.info(f"TCR-Pep测试集: {tcr_pep_test.shape[0]}样本，正例比例: {tcr_pep_test['Label'].mean():.4f}")
    
    logger.info(f"HLA-Pep训练集: {hla_pep_train.shape[0]}样本，正例比例: {hla_pep_train['Label'].mean():.4f}")
    logger.info(f"HLA-Pep验证集: {hla_pep_val.shape[0]}样本，正例比例: {hla_pep_val['Label'].mean():.4f}")
    logger.info(f"HLA-Pep测试集: {hla_pep_test.shape[0]}样本，正例比例: {hla_pep_test['Label'].mean():.4f}")
    
    logger.info(f"TCR-HLA-Pep训练集: {trimer_train.shape[0]}样本，正例比例: {trimer_train['Label'].mean():.4f}")
    logger.info(f"TCR-HLA-Pep验证集: {trimer_val.shape[0]}样本，正例比例: {trimer_val['Label'].mean():.4f}")
    logger.info(f"TCR-HLA-Pep测试集: {trimer_test.shape[0]}样本，正例比例: {trimer_test['Label'].mean():.4f}")


def process_dataset(df: pd.DataFrame, hla_seq_dict: Dict[str, str], dataset_name: str, config: Dict) -> pd.DataFrame:
    """
    处理单个数据集
    
    Args:
        df: 数据集DataFrame
        hla_seq_dict: HLA名称到序列的映射字典
        dataset_name: 数据集名称
        config: 配置字典
        
    Returns:
        处理后的DataFrame
    """
    logger = logging.getLogger("data_processor")
    
    # 创建副本
    df = df.copy()
    
    # 添加HLA序列
    logger.info(f"为{dataset_name}数据集添加HLA序列...")
    df["HLA_sequence"] = df["HLA"].map(hla_seq_dict)
    
    # 删除缺失值
    n_before = df.shape[0]
    df = df.dropna(subset=["CDR3", "MT_pep", "HLA_sequence", "Label"])
    n_after = df.shape[0]
    logger.info(f"删除缺失值后，{dataset_name}数据集从{n_before}行减少到{n_after}行")
    
    # 过滤序列长度
    max_tcr_len = config["data"]["max_tcr_len"]
    max_pep_len = config["data"]["max_pep_len"]
    max_hla_len = config["data"]["max_hla_len"]
    
    n_before = df.shape[0]
    df = df[
        (df["CDR3"].str.len() <= max_tcr_len) & 
        (df["MT_pep"].str.len() <= max_pep_len) & 
        (df["HLA_sequence"].str.len() <= max_hla_len)
    ]
    n_after = df.shape[0]
    logger.info(f"过滤序列长度后，{dataset_name}数据集从{n_before}行减少到{n_after}行")
    
    return df


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理TCR-HLA-Pep数据")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw", help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("data_processor", os.path.join(args.log_dir, "data_processor.log"))
    
    # 加载配置
    if args.config is not None and os.path.exists(args.config):
        logger.info(f"从{args.config}加载配置...")
        config = load_config(args.config)
    else:
        logger.info("使用默认配置...")
        config = get_default_config()
    
    # 加载原始数据
    logger.info(f"从{args.raw_data_dir}加载原始数据...")
    data = load_raw_data(args.raw_data_dir)
    
    # 处理数据
    logger.info(f"处理数据并保存到{args.output_dir}...")
    process_data(data, args.output_dir, config)
    
    logger.info("数据处理完成！")


if __name__ == "__main__":
    main() 