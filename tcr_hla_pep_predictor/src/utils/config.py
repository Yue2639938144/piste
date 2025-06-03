"""
配置工具模块

该模块提供配置加载和保存功能，支持YAML和JSON格式。
"""

import os
import json
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，支持YAML和JSON格式
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif file_ext == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {file_ext}")
    except Exception as e:
        raise RuntimeError(f"加载配置文件失败: {str(e)}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str, overwrite: bool = False) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 保存路径，支持YAML和JSON格式
        overwrite: 是否覆盖已存在的文件
    """
    if os.path.exists(config_path) and not overwrite:
        raise FileExistsError(f"配置文件已存在，设置overwrite=True可覆盖: {config_path}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    file_ext = os.path.splitext(config_path)[1].lower()
    
    try:
        if file_ext in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif file_ext == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"不支持的配置文件格式: {file_ext}")
    except Exception as e:
        raise RuntimeError(f"保存配置文件失败: {str(e)}")


def get_default_config() -> Dict[str, Any]:
    """
    获取默认配置
    
    Returns:
        默认配置字典
    """
    return {
        # 数据配置
        "data": {
            "max_tcr_len": 30,
            "max_pep_len": 15,
            "max_hla_len": 34,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "n_clusters": 10,
            "clustering_method": "hierarchical",
            "similarity_threshold": 0.7
        },
        
        # 模型配置
        "model": {
            "embedding_dim": 128,
            "hidden_dim": 256,
            "num_heads": 8,
            "num_layers": 4,
            "dropout": 0.1,
            "use_biochem_features": True
        },
        
        # 注意力机制配置
        "attention": {
            "physical_sliding": {
                "enabled": True,
                "sigma": 1.0,
                "num_iterations": 3
            },
            "data_driven": {
                "enabled": True
            },
            "fusion_method": "weighted_sum",  # weighted_sum, concat, or gated
            "fusion_weights": [0.5, 0.5]  # 物理滑动和数据驱动的权重
        },
        
        # 训练配置
        "training": {
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "max_epochs": 200,
            "patience": 20,
            "loss_weights": {
                "tcr_pep": 1.0,
                "hla_pep": 1.0,
                "trimer": 1.0
            }
        },
        
        # 优化器配置
        "optimizer": {
            "type": "adam",
            "lr_scheduler": {
                "type": "reduce_on_plateau",
                "factor": 0.5,
                "patience": 10,
                "min_lr": 1e-6
            }
        },
        
        # 路径配置
        "paths": {
            "data_dir": "data",
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "model_dir": "models",
            "log_dir": "logs",
            "results_dir": "results"
        }
    } 