#!/usr/bin/env python
"""
数据预处理脚本

该脚本用于处理原始数据，包括数据清洗、序列编码、特征提取和数据集划分。
支持基于序列相似性的聚类划分，以避免数据泄露。
"""

import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from sklearn.model_selection import train_test_split
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_processor import process_data, load_raw_data
from src.data.clustering import sequence_similarity_clustering, split_by_clusters
from src.utils.config import load_config, save_config, get_default_config
from src.utils.logger import setup_logger


def preprocess_data(raw_data_dir: str, 
                   output_dir: str, 
                   config: Dict, 
                   mode: str, 
                   split: bool = False, 
                   clustering: bool = False) -> None:
    """
    预处理数据
    
    Args:
        raw_data_dir: 原始数据目录
        output_dir: 输出目录
        config: 配置字典
        mode: 预处理模式，'tcr_pep'、'hla_pep'或'trimer'
        split: 是否拆分数据集
        clustering: 是否使用聚类进行数据拆分
    """
    logger = logging.getLogger("preprocess")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据
    logger.info(f"从{raw_data_dir}加载原始数据...")
    data = load_raw_data(raw_data_dir)
    
    # 合并数据集（如果需要拆分）
    if split:
        logger.info("合并数据集以进行拆分...")
        all_data = pd.concat([data["train"], data["val"], data["test"]], ignore_index=True)
        
        # 拆分数据集
        if clustering:
            logger.info("使用序列相似性聚类进行数据拆分...")
            
            if mode == 'tcr_pep':
                # 基于TCR序列相似性聚类
                logger.info("基于TCR序列相似性聚类...")
                clusters = sequence_similarity_clustering(
                    all_data["CDR3"].tolist(), 
                    n_clusters=config["data"]["n_clusters"],
                    method=config["data"]["clustering_method"],
                    threshold=config["data"]["similarity_threshold"]
                )
                
                # 基于聚类拆分数据集
                train_data, val_data, test_data = split_by_clusters(
                    all_data, clusters,
                    train_ratio=config["data"]["train_ratio"],
                    val_ratio=config["data"]["val_ratio"],
                    test_ratio=config["data"]["test_ratio"]
                )
                
            elif mode == 'hla_pep':
                # 基于肽段序列相似性聚类
                logger.info("基于肽段序列相似性聚类...")
                clusters = sequence_similarity_clustering(
                    all_data["MT_pep"].tolist(), 
                    n_clusters=config["data"]["n_clusters"],
                    method=config["data"]["clustering_method"],
                    threshold=config["data"]["similarity_threshold"]
                )
                
                # 基于聚类拆分数据集
                train_data, val_data, test_data = split_by_clusters(
                    all_data, clusters,
                    train_ratio=config["data"]["train_ratio"],
                    val_ratio=config["data"]["val_ratio"],
                    test_ratio=config["data"]["test_ratio"]
                )
                
            elif mode == 'trimer':
                # 基于TCR和肽段序列相似性聚类
                logger.info("基于TCR和肽段序列相似性聚类...")
                
                # 首先基于TCR序列聚类
                tcr_clusters = sequence_similarity_clustering(
                    all_data["CDR3"].tolist(), 
                    n_clusters=config["data"]["n_clusters"] // 2,
                    method=config["data"]["clustering_method"],
                    threshold=config["data"]["similarity_threshold"]
                )
                
                # 然后基于肽段序列聚类
                pep_clusters = sequence_similarity_clustering(
                    all_data["MT_pep"].tolist(), 
                    n_clusters=config["data"]["n_clusters"] // 2,
                    method=config["data"]["clustering_method"],
                    threshold=config["data"]["similarity_threshold"]
                )
                
                # 合并聚类结果
                combined_clusters = []
                for i in range(len(all_data)):
                    combined_clusters.append(f"{tcr_clusters[i]}_{pep_clusters[i]}")
                
                # 基于聚类拆分数据集
                train_data, val_data, test_data = split_by_clusters(
                    all_data, combined_clusters,
                    train_ratio=config["data"]["train_ratio"],
                    val_ratio=config["data"]["val_ratio"],
                    test_ratio=config["data"]["test_ratio"]
                )
            
            else:
                raise ValueError(f"不支持的预处理模式: {mode}")
                
        else:
            # 使用随机拆分
            logger.info("使用随机拆分数据集...")
            
            # 先拆分出测试集
            train_val_data, test_data = train_test_split(
                all_data, 
                test_size=config["data"]["test_ratio"],
                random_state=42,
                stratify=all_data["Label"]
            )
            
            # 再从剩余数据中拆分出验证集
            val_ratio_adjusted = config["data"]["val_ratio"] / (config["data"]["train_ratio"] + config["data"]["val_ratio"])
            train_data, val_data = train_test_split(
                train_val_data, 
                test_size=val_ratio_adjusted,
                random_state=42,
                stratify=train_val_data["Label"]
            )
        
        # 更新数据字典
        data = {
            "train": train_data,
            "val": val_data,
            "test": test_data,
            "hla_seq": data["hla_seq"]
        }
    
    # 处理数据
    logger.info(f"处理{mode}数据并保存到{output_dir}...")
    process_data(data, output_dir, config)
    
    logger.info("数据预处理完成！")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="预处理TCR-HLA-Pep数据")
    parser.add_argument("--raw_data_dir", type=str, default="data/raw", help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--mode", type=str, required=True, choices=['tcr_pep', 'hla_pep', 'trimer'], 
                       help="预处理模式: tcr_pep, hla_pep, trimer")
    parser.add_argument("--split", action="store_true", help="是否拆分数据集")
    parser.add_argument("--clustering", action="store_true", help="是否使用聚类进行数据拆分")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="测试集比例")
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置日志
    logger = setup_logger("preprocess", os.path.join(args.log_dir, "preprocess.log"))
    
    # 加载配置
    if args.config is not None and os.path.exists(args.config):
        logger.info(f"从{args.config}加载配置...")
        config = load_config(args.config)
    else:
        logger.info("使用默认配置...")
        config = get_default_config()
    
    # 更新配置
    config["data"]["train_ratio"] = args.train_ratio
    config["data"]["val_ratio"] = args.val_ratio
    config["data"]["test_ratio"] = args.test_ratio
    
    # 预处理数据
    preprocess_data(
        args.raw_data_dir, 
        args.output_dir, 
        config, 
        args.mode, 
        args.split, 
        args.clustering
    )


if __name__ == "__main__":
    main() 