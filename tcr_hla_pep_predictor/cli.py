#!/usr/bin/env python
"""
TCR-HLA-Pep预测器命令行接口

该脚本提供了TCR-HLA-Pep预测器的命令行接口，支持训练、评估、预测和可视化功能。
"""

import os
import sys
import argparse
import logging
import yaml
import torch
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import setup_logger
from src.utils.config import load_config


def setup_common_args(parser: argparse.ArgumentParser):
    """
    设置通用命令行参数
    
    Args:
        parser: 命令行参数解析器
    """
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", 
                       help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="results", 
                       help="输出目录")
    parser.add_argument("--gpu", type=int, default=0, 
                       help="GPU设备ID，-1表示使用CPU")
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子")


def train_command(args: argparse.Namespace):
    """
    训练命令处理函数
    
    Args:
        args: 命令行参数
    """
    # 导入训练相关模块
    if args.mode == 'tcr_pep':
        from scripts.train_tcr_pep import main as train_main
    elif args.mode == 'hla_pep':
        from scripts.train_hla_pep import main as train_main
    elif args.mode == 'trimer':
        from scripts.train_trimer import main as train_main
    else:
        raise ValueError(f"不支持的训练模式: {args.mode}")
    
    # 准备训练参数
    train_args = [
        "--config", args.config,
        "--train_data", args.train_data,
        "--val_data", args.val_data,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
    ]
    
    # 添加可选参数
    if args.test_data:
        train_args.extend(["--test_data", args.test_data])
    
    if args.pretrained_model:
        train_args.extend(["--pretrained_model", args.pretrained_model])
    
    if args.early_stopping:
        train_args.append("--early_stopping")
    
    if args.joint_optimization and args.mode == 'trimer':
        train_args.append("--joint_optimization")
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 设置随机种子
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    # 执行训练
    sys.argv = [sys.argv[0]] + train_args
    train_main()


def evaluate_command(args: argparse.Namespace):
    """
    评估命令处理函数
    
    Args:
        args: 命令行参数
    """
    # 导入评估模块
    from scripts.evaluate_model import main as evaluate_main
    
    # 准备评估参数
    evaluate_args = [
        "--config", args.config,
        "--model", args.model,
        "--test_data", args.test_data,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
    ]
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 设置随机种子
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    # 执行评估
    sys.argv = [sys.argv[0]] + evaluate_args
    evaluate_main()


def predict_command(args: argparse.Namespace):
    """
    预测命令处理函数
    
    Args:
        args: 命令行参数
    """
    # 导入预测模块
    from scripts.predict_and_visualize import main as predict_main
    
    # 准备预测参数
    predict_args = [
        "--config", args.config,
        "--model", args.model,
        "--input", args.input,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
    ]
    
    # 添加可视化参数
    if args.visualize:
        predict_args.append("--visualize")
        predict_args.extend(["--visualize_samples", str(args.visualize_samples)])
        
        if args.interactive:
            predict_args.append("--interactive")
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 设置随机种子
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    # 执行预测
    sys.argv = [sys.argv[0]] + predict_args
    predict_main()


def preprocess_command(args: argparse.Namespace):
    """
    数据预处理命令处理函数
    
    Args:
        args: 命令行参数
    """
    # 导入预处理模块
    from scripts.preprocess_data import main as preprocess_main
    
    # 准备预处理参数
    preprocess_args = [
        "--config", args.config,
        "--input", args.input,
        "--output", args.output,
        "--mode", args.mode,
    ]
    
    # 添加可选参数
    if args.split:
        preprocess_args.append("--split")
        preprocess_args.extend(["--train_ratio", str(args.train_ratio)])
        preprocess_args.extend(["--val_ratio", str(args.val_ratio)])
        preprocess_args.extend(["--test_ratio", str(args.test_ratio)])
    
    if args.clustering:
        preprocess_args.append("--clustering")
    
    # 执行预处理
    sys.argv = [sys.argv[0]] + preprocess_args
    preprocess_main()


def main():
    """
    主函数，处理命令行参数并执行相应的命令
    """
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description="TCR-HLA-Pep预测器命令行工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    setup_common_args(train_parser)
    train_parser.add_argument("--mode", type=str, required=True, choices=['tcr_pep', 'hla_pep', 'trimer'],
                             help="训练模式: tcr_pep, hla_pep, trimer")
    train_parser.add_argument("--train_data", type=str, required=True, 
                             help="训练数据路径")
    train_parser.add_argument("--val_data", type=str, required=True, 
                             help="验证数据路径")
    train_parser.add_argument("--test_data", type=str, 
                             help="测试数据路径")
    train_parser.add_argument("--pretrained_model", type=str, 
                             help="预训练模型路径")
    train_parser.add_argument("--batch_size", type=int, default=32, 
                             help="批大小")
    train_parser.add_argument("--epochs", type=int, default=50, 
                             help="训练轮数")
    train_parser.add_argument("--learning_rate", type=float, default=0.001, 
                             help="学习率")
    train_parser.add_argument("--early_stopping", action="store_true", 
                             help="是否使用早停")
    train_parser.add_argument("--joint_optimization", action="store_true", 
                             help="是否使用联合优化（仅限trimer模式）")
    
    # 评估命令
    evaluate_parser = subparsers.add_parser("evaluate", help="评估模型")
    setup_common_args(evaluate_parser)
    evaluate_parser.add_argument("--model", type=str, required=True, 
                                help="模型路径")
    evaluate_parser.add_argument("--test_data", type=str, required=True, 
                                help="测试数据路径")
    evaluate_parser.add_argument("--batch_size", type=int, default=32, 
                                help="批大小")
    
    # 预测命令
    predict_parser = subparsers.add_parser("predict", help="预测和可视化")
    setup_common_args(predict_parser)
    predict_parser.add_argument("--model", type=str, required=True, 
                               help="模型路径")
    predict_parser.add_argument("--input", type=str, required=True, 
                               help="输入数据路径")
    predict_parser.add_argument("--batch_size", type=int, default=32, 
                               help="批大小")
    predict_parser.add_argument("--visualize", action="store_true", 
                               help="是否生成可视化")
    predict_parser.add_argument("--visualize_samples", type=int, default=5, 
                               help="可视化样本数量")
    predict_parser.add_argument("--interactive", action="store_true", 
                               help="是否生成交互式可视化")
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser("preprocess", help="预处理数据")
    setup_common_args(preprocess_parser)
    preprocess_parser.add_argument("--input", type=str, required=True, 
                                  help="输入数据路径")
    preprocess_parser.add_argument("--output", type=str, required=True, 
                                  help="输出数据路径")
    preprocess_parser.add_argument("--mode", type=str, required=True, 
                                  choices=['tcr_pep', 'hla_pep', 'trimer'],
                                  help="预处理模式: tcr_pep, hla_pep, trimer")
    preprocess_parser.add_argument("--split", action="store_true", 
                                  help="是否拆分数据集")
    preprocess_parser.add_argument("--train_ratio", type=float, default=0.7, 
                                  help="训练集比例")
    preprocess_parser.add_argument("--val_ratio", type=float, default=0.15, 
                                  help="验证集比例")
    preprocess_parser.add_argument("--test_ratio", type=float, default=0.15, 
                                  help="测试集比例")
    preprocess_parser.add_argument("--clustering", action="store_true", 
                                  help="是否使用聚类进行数据拆分")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助信息
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # 创建输出目录
    if hasattr(args, 'output_dir'):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据命令执行相应的函数
    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "preprocess":
        preprocess_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 