#!/usr/bin/env python
"""
数据预处理脚本

该脚本用于处理原始数据，支持以下功能：
1. 二元模型：输入包含阳性集合与阴性集合的文件夹（支持调整阴性集合倍数）
2. 三元模型：输入包含3对阳性阴性集合的文件夹，分别用于训练模型
3. 自动检测数据格式和缺失值（pep：长度9-12的氨基酸序列，tcr：30氨基酸以内，hla：形如HLA-A01:02）
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

from src.data.data_processor import preprocess_data
from src.utils.config import load_config, save_config, get_default_config
from src.utils.logger import setup_logger


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="预处理TCR-HLA-Pep数据")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="输出目录")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    parser.add_argument("--log_dir", type=str, default="logs", help="日志目录")
    parser.add_argument("--mode", type=str, required=True, choices=['tcr_pep', 'hla_pep', 'trimer'], 
                       help="预处理模式: tcr_pep, hla_pep, trimer")
    parser.add_argument("--negative_ratio", type=float, default=1.0, 
                       help="阴性样本相对于阳性样本的比例（仅用于二元模型）")
    parser.add_argument("--split", action="store_true", help="是否拆分数据集")
    parser.add_argument("--clustering", action="store_true", help="是否使用聚类进行数据拆分")
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
    
    # 更新配置中的数据集拆分比例
    config["data"]["train_ratio"] = 0.7
    config["data"]["val_ratio"] = 0.15
    config["data"]["test_ratio"] = 0.15
    
    # 预处理数据
    logger.info(f"开始处理{args.mode}数据...")
    preprocess_data(
        args.data_dir, 
        args.output_dir, 
        config, 
        args.mode,
        args.negative_ratio,
        args.split, 
        args.clustering
    )


if __name__ == "__main__":
    main() 