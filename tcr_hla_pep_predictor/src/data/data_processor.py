"""
数据处理脚本

该脚本用于处理TCR-Pep、HLA-Pep和TCR-HLA-Pep数据，支持以下功能：
1. 二元模型：输入包含阳性集合与阴性集合的文件夹（支持调整阴性集合倍数）
2. 三元模型：输入包含3对阳性阴性集合的文件夹，分别用于训练模型
3. 自动检测数据格式和缺失值（pep：长度9-12的氨基酸序列，tcr：30氨基酸以内，hla：形如HLA-A01:02）
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Set
from sklearn.model_selection import train_test_split
import logging
import argparse
import sys
import yaml
import re
import glob

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.clustering import sequence_similarity_clustering, split_by_clusters
from src.utils.config import load_config, save_config, get_default_config
from src.utils.logger import setup_logger


def validate_pep_sequence(seq: str) -> bool:
    """
    验证肽段序列是否有效（长度9-12的氨基酸序列）
    
    Args:
        seq: 肽段序列
        
    Returns:
        True如果有效，否则False
    """
    if not isinstance(seq, str):
        return False
    
    # 氨基酸标准字符集
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    # 检查长度和字符
    if 9 <= len(seq) <= 12 and all(aa in amino_acids for aa in seq.upper()):
        return True
    return False


def validate_tcr_sequence(seq: str) -> bool:
    """
    验证TCR序列是否有效（30氨基酸以内）
    
    Args:
        seq: TCR序列
        
    Returns:
        True如果有效，否则False
    """
    if not isinstance(seq, str):
        return False
    
    # 氨基酸标准字符集
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    # 检查长度和字符
    if 0 < len(seq) <= 30 and all(aa in amino_acids for aa in seq):
        return True
    return False


def validate_hla_format(hla: str) -> bool:
    """
    验证HLA格式是否有效（形如HLA-A01:02）
    
    Args:
        hla: HLA标识符
        
    Returns:
        True如果有效，否则False
    """
    if not isinstance(hla, str):
        return False
    
    # 匹配HLA-X*XX:XX或HLA-XXX:XX格式（X是字母或数字）
    pattern = r'^HLA-[A-Z]\d{2}:\d{2}$|^HLA-[A-Z]\*\d{2}:\d{2}$'
    return bool(re.match(pattern, hla))


def validate_hla_sequence(seq: str) -> bool:
    """
    验证HLA序列是否有效
    
    Args:
        seq: HLA序列
        
    Returns:
        True如果有效，否则False
    """
    if not isinstance(seq, str):
        return False
    
    # 氨基酸标准字符集
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    # 检查长度和字符（HLA序列通常较长）
    if 0 < len(seq) <= 400 and all(aa in amino_acids for aa in seq):
        return True
    return False


def check_data_format(df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[int]]:
    """
    检查数据格式并返回错误行的索引
    
    Args:
        df: 数据DataFrame
        data_type: 数据类型，'tcr_pep', 'hla_pep'或'trimer'
        
    Returns:
        处理后的DataFrame和错误行索引列表
    """
    logger = logging.getLogger("data_processor")
    
    # 创建一个包含所有错误行索引的集合
    error_indices = set()
    
    # 根据数据类型检查必要的列
    if data_type == 'tcr_pep':
        required_columns = ['CDR3', 'MT_pep', 'Label']
    elif data_type == 'hla_pep':
        required_columns = ['HLA', 'MT_pep', 'Label']
    elif data_type == 'trimer':
        required_columns = ['CDR3', 'HLA', 'MT_pep', 'Label']
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")
    
    # 检查必要的列是否存在
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"缺少必要的列: {col}")
            raise ValueError(f"缺少必要的列: {col}")
    
    # 检查缺失值
    for col in required_columns:
        missing_indices = df[df[col].isnull()].index.tolist()
        if missing_indices:
            logger.warning(f"列'{col}'中有{len(missing_indices)}个缺失值")
            error_indices.update(missing_indices)
    
    # 检查数据格式
    # 1. 检查肽段序列
    for idx, pep in enumerate(df['MT_pep']):
        if pd.notna(pep) and not validate_pep_sequence(pep):
            logger.warning(f"行{idx+1}的肽段序列格式无效: {pep}")
            error_indices.add(idx)
    
    # 2. 根据数据类型检查其他序列
    if data_type in ['tcr_pep', 'trimer']:
        for idx, tcr in enumerate(df['CDR3']):
            if pd.notna(tcr) and not validate_tcr_sequence(tcr):
                logger.warning(f"行{idx+1}的TCR序列格式无效: {tcr}")
                error_indices.add(idx)
    
    if data_type in ['hla_pep', 'trimer']:
        for idx, hla in enumerate(df['HLA']):
            if pd.notna(hla) and not validate_hla_format(hla):
                logger.warning(f"行{idx+1}的HLA格式无效: {hla}")
                error_indices.add(idx)
    
    # 转换错误索引集合为列表并排序
    error_indices = sorted(list(error_indices))
    
    # 返回不包含错误行的DataFrame和错误行索引列表
    clean_df = df.drop(index=error_indices).reset_index(drop=True)
    
    return clean_df, error_indices


def load_binary_data(data_dir: str, binary_type: str, negative_ratio: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    加载二元模型数据（从包含阳性和阴性集合的文件夹）
    
    Args:
        data_dir: 数据目录，应包含pos和neg子文件夹
        binary_type: 二元类型，'tcr_pep'或'hla_pep'
        negative_ratio: 阴性样本相对于阳性样本的比例
        
    Returns:
        包含训练、验证和测试集的字典
    """
    logger = logging.getLogger("data_processor")
    
    # 验证二元类型
    if binary_type not in ['tcr_pep', 'hla_pep']:
        raise ValueError(f"不支持的二元类型: {binary_type}")
    
    # 构建阳性和阴性数据路径
    pos_path = os.path.join(data_dir, "pos")
    neg_path = os.path.join(data_dir, "neg")
    
    # 检查目录是否存在
    if not os.path.exists(pos_path) or not os.path.isdir(pos_path):
        raise FileNotFoundError(f"找不到阳性数据目录: {pos_path}")
    
    if not os.path.exists(neg_path) or not os.path.isdir(neg_path):
        raise FileNotFoundError(f"找不到阴性数据目录: {neg_path}")
    
    # 获取所有CSV文件
    pos_files = glob.glob(os.path.join(pos_path, "*.csv"))
    neg_files = glob.glob(os.path.join(neg_path, "*.csv"))
    
    if not pos_files:
        raise FileNotFoundError(f"在{pos_path}中找不到CSV文件")
    
    if not neg_files:
        raise FileNotFoundError(f"在{neg_path}中找不到CSV文件")
    
    # 加载并合并所有阳性数据
    logger.info(f"加载阳性数据: {len(pos_files)}个文件")
    pos_dfs = []
    for file in pos_files:
        try:
            df = pd.read_csv(file)
            # 添加标签列
            df['Label'] = 1
            pos_dfs.append(df)
        except Exception as e:
            logger.error(f"加载文件{file}失败: {str(e)}")
    
    if not pos_dfs:
        raise ValueError("无法加载任何阳性数据")
    
    pos_data = pd.concat(pos_dfs, ignore_index=True)
    logger.info(f"阳性数据大小: {pos_data.shape}")
    
    # 加载并合并所有阴性数据
    logger.info(f"加载阴性数据: {len(neg_files)}个文件")
    neg_dfs = []
    for file in neg_files:
        try:
            df = pd.read_csv(file)
            # 添加标签列
            df['Label'] = 0
            neg_dfs.append(df)
        except Exception as e:
            logger.error(f"加载文件{file}失败: {str(e)}")
    
    if not neg_dfs:
        raise ValueError("无法加载任何阴性数据")
    
    neg_data = pd.concat(neg_dfs, ignore_index=True)
    logger.info(f"阴性数据大小: {neg_data.shape}")
    
    # 检查数据格式
    logger.info("检查阳性数据格式...")
    pos_data, pos_error_indices = check_data_format(pos_data, binary_type)
    logger.info(f"阳性数据中发现{len(pos_error_indices)}行格式错误")
    
    logger.info("检查阴性数据格式...")
    neg_data, neg_error_indices = check_data_format(neg_data, binary_type)
    logger.info(f"阴性数据中发现{len(neg_error_indices)}行格式错误")
    
    # 根据negative_ratio调整阴性样本数量
    pos_count = len(pos_data)
    target_neg_count = int(pos_count * negative_ratio)
    
    if len(neg_data) > target_neg_count:
        logger.info(f"调整阴性样本数量: 从{len(neg_data)}减少到{target_neg_count} (比例: {negative_ratio})")
        neg_data = neg_data.sample(n=target_neg_count, random_state=42)
    else:
        logger.warning(f"阴性样本数量({len(neg_data)})少于目标数量({target_neg_count})")
    
    # 合并阳性和阴性数据
    all_data = pd.concat([pos_data, neg_data], ignore_index=True)
    logger.info(f"合并后的数据大小: {all_data.shape}")
    
    # 随机打乱数据
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 拆分为训练集、验证集和测试集
    train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15
    
    # 先拆分出测试集
    train_val_data, test_data = train_test_split(
        all_data, 
        test_size=test_ratio,
        random_state=42,
        stratify=all_data["Label"]
    )
    
    # 再从剩余数据中拆分出验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(
        train_val_data, 
        test_size=val_ratio_adjusted,
        random_state=42,
        stratify=train_val_data["Label"]
    )
    
    # 如果是HLA-Pep类型，则还需加载HLA序列数据
    hla_seq_df = None
    if binary_type == 'hla_pep':
        hla_seq_path = os.path.join(data_dir, "hla_sequences.csv")
        if os.path.exists(hla_seq_path):
            logger.info(f"加载HLA序列数据: {hla_seq_path}")
            try:
                hla_seq_df = pd.read_csv(hla_seq_path)
            except Exception as e:
                logger.error(f"加载HLA序列数据失败: {str(e)}")
                raise
        else:
            logger.warning(f"找不到HLA序列数据文件: {hla_seq_path}")
            logger.warning("将尝试使用其他HLA序列信息")
    
    return {
        "train": train_data,
        "val": val_data,
        "test": test_data,
        "hla_seq": hla_seq_df
    }


def load_trimer_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    加载三元模型数据（从包含3对阳性阴性集合的文件夹）
    
    Args:
        data_dir: 数据目录，应包含tcr_pep、hla_pep和trimer子文件夹，每个子文件夹中又有pos和neg
        
    Returns:
        包含不同数据集的字典
    """
    logger = logging.getLogger("data_processor")
    
    # 构建数据路径
    tcr_pep_dir = os.path.join(data_dir, "tcr_pep")
    hla_pep_dir = os.path.join(data_dir, "hla_pep")
    trimer_dir = os.path.join(data_dir, "trimer")
    
    # 检查目录是否存在
    for dir_path, dir_name in [(tcr_pep_dir, "TCR-Pep"), (hla_pep_dir, "HLA-Pep"), (trimer_dir, "Trimer")]:
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            raise FileNotFoundError(f"找不到{dir_name}数据目录: {dir_path}")
    
    # 加载TCR-Pep数据
    logger.info("加载TCR-Pep数据...")
    tcr_pep_data = load_binary_data(tcr_pep_dir, "tcr_pep")
    
    # 加载HLA-Pep数据
    logger.info("加载HLA-Pep数据...")
    hla_pep_data = load_binary_data(hla_pep_dir, "hla_pep")
    
    # 加载Trimer数据
    logger.info("加载Trimer数据...")
    trimer_data = load_binary_data(trimer_dir, "trimer")
    
    # 合并结果
    return {
        "tcr_pep": {
            "train": tcr_pep_data["train"],
            "val": tcr_pep_data["val"],
            "test": tcr_pep_data["test"]
        },
        "hla_pep": {
            "train": hla_pep_data["train"],
            "val": hla_pep_data["val"],
            "test": hla_pep_data["test"]
        },
        "trimer": {
            "train": trimer_data["train"],
            "val": trimer_data["val"],
            "test": trimer_data["test"]
        },
        "hla_seq": hla_pep_data["hla_seq"]
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
    
    # 针对二元模型数据处理
    if "train" in data and "val" in data and "test" in data:
        # 原始格式数据处理
        # 合并HLA序列信息（如果有）
        hla_seq_dict = {}
        if "hla_seq" in data and data["hla_seq"] is not None:
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
        
        # 根据数据类型提取和保存不同的数据集
        if "CDR3" in train_df.columns and "MT_pep" in train_df.columns:
            # 提取TCR-Pep数据
            logger.info("提取TCR-Pep数据...")
            tcr_pep_train = train_df[["CDR3", "MT_pep", "Label"]].copy()
            tcr_pep_val = val_df[["CDR3", "MT_pep", "Label"]].copy()
            tcr_pep_test = test_df[["CDR3", "MT_pep", "Label"]].copy()
            
            # 保存TCR-Pep数据
            logger.info("保存TCR-Pep数据...")
            tcr_pep_train.to_csv(os.path.join(output_dir, "tcr_pep_train.csv"), index=False)
            tcr_pep_val.to_csv(os.path.join(output_dir, "tcr_pep_val.csv"), index=False)
            tcr_pep_test.to_csv(os.path.join(output_dir, "tcr_pep_test.csv"), index=False)
            
            # 打印TCR-Pep数据统计信息
            logger.info("\nTCR-Pep数据统计信息:")
            logger.info(f"训练集: {tcr_pep_train.shape[0]}样本，正例比例: {tcr_pep_train['Label'].mean():.4f}")
            logger.info(f"验证集: {tcr_pep_val.shape[0]}样本，正例比例: {tcr_pep_val['Label'].mean():.4f}")
            logger.info(f"测试集: {tcr_pep_test.shape[0]}样本，正例比例: {tcr_pep_test['Label'].mean():.4f}")
        
        if "HLA_sequence" in train_df.columns and "MT_pep" in train_df.columns:
            # 提取HLA-Pep数据
            logger.info("提取HLA-Pep数据...")
            hla_pep_train = train_df[["HLA_sequence", "MT_pep", "Label"]].copy()
            hla_pep_val = val_df[["HLA_sequence", "MT_pep", "Label"]].copy()
            hla_pep_test = test_df[["HLA_sequence", "MT_pep", "Label"]].copy()
            
            # 保存HLA-Pep数据
            logger.info("保存HLA-Pep数据...")
            hla_pep_train.to_csv(os.path.join(output_dir, "hla_pep_train.csv"), index=False)
            hla_pep_val.to_csv(os.path.join(output_dir, "hla_pep_val.csv"), index=False)
            hla_pep_test.to_csv(os.path.join(output_dir, "hla_pep_test.csv"), index=False)
            
            # 打印HLA-Pep数据统计信息
            logger.info("\nHLA-Pep数据统计信息:")
            logger.info(f"训练集: {hla_pep_train.shape[0]}样本，正例比例: {hla_pep_train['Label'].mean():.4f}")
            logger.info(f"验证集: {hla_pep_val.shape[0]}样本，正例比例: {hla_pep_val['Label'].mean():.4f}")
            logger.info(f"测试集: {hla_pep_test.shape[0]}样本，正例比例: {hla_pep_test['Label'].mean():.4f}")
        
        if "CDR3" in train_df.columns and "MT_pep" in train_df.columns and "HLA_sequence" in train_df.columns:
            # 提取TCR-HLA-Pep数据
            logger.info("提取TCR-HLA-Pep数据...")
            trimer_train = train_df[["CDR3", "MT_pep", "HLA_sequence", "Label"]].copy()
            trimer_val = val_df[["CDR3", "MT_pep", "HLA_sequence", "Label"]].copy()
            trimer_test = test_df[["CDR3", "MT_pep", "HLA_sequence", "Label"]].copy()
            
            # 保存TCR-HLA-Pep数据
            logger.info("保存TCR-HLA-Pep数据...")
            trimer_train.to_csv(os.path.join(output_dir, "trimer_train.csv"), index=False)
            trimer_val.to_csv(os.path.join(output_dir, "trimer_val.csv"), index=False)
            trimer_test.to_csv(os.path.join(output_dir, "trimer_test.csv"), index=False)
            
            # 打印TCR-HLA-Pep数据统计信息
            logger.info("\nTCR-HLA-Pep数据统计信息:")
            logger.info(f"训练集: {trimer_train.shape[0]}样本，正例比例: {trimer_train['Label'].mean():.4f}")
            logger.info(f"验证集: {trimer_val.shape[0]}样本，正例比例: {trimer_val['Label'].mean():.4f}")
            logger.info(f"测试集: {trimer_test.shape[0]}样本，正例比例: {trimer_test['Label'].mean():.4f}")
    
    # 针对三元模型数据处理
    elif "tcr_pep" in data and "hla_pep" in data and "trimer" in data:
        # 新格式三元数据处理
        # 创建子目录
        tcr_pep_dir = os.path.join(output_dir, "tcr_pep")
        hla_pep_dir = os.path.join(output_dir, "hla_pep")
        trimer_dir = os.path.join(output_dir, "trimer")
        
        os.makedirs(tcr_pep_dir, exist_ok=True)
        os.makedirs(hla_pep_dir, exist_ok=True)
        os.makedirs(trimer_dir, exist_ok=True)
        
        # 处理TCR-Pep数据
        logger.info("处理TCR-Pep数据...")
        tcr_pep_data = data["tcr_pep"]
        tcr_pep_data["train"].to_csv(os.path.join(tcr_pep_dir, "train.csv"), index=False)
        tcr_pep_data["val"].to_csv(os.path.join(tcr_pep_dir, "val.csv"), index=False)
        tcr_pep_data["test"].to_csv(os.path.join(tcr_pep_dir, "test.csv"), index=False)
        
        # 处理HLA-Pep数据
        logger.info("处理HLA-Pep数据...")
        hla_pep_data = data["hla_pep"]
        
        # 如果有HLA序列数据，添加HLA序列
        if data["hla_seq"] is not None:
            logger.info("合并HLA序列信息...")
            hla_seq_dict = dict(zip(data["hla_seq"]["HLA"], data["hla_seq"]["pseudo_sequence"]))
            
            for split, df in hla_pep_data.items():
                if "HLA" in df.columns and "HLA_sequence" not in df.columns:
                    df["HLA_sequence"] = df["HLA"].map(hla_seq_dict)
        
        hla_pep_data["train"].to_csv(os.path.join(hla_pep_dir, "train.csv"), index=False)
        hla_pep_data["val"].to_csv(os.path.join(hla_pep_dir, "val.csv"), index=False)
        hla_pep_data["test"].to_csv(os.path.join(hla_pep_dir, "test.csv"), index=False)
        
        # 处理Trimer数据
        logger.info("处理Trimer数据...")
        trimer_data = data["trimer"]
        
        # 如果有HLA序列数据，添加HLA序列
        if data["hla_seq"] is not None:
            logger.info("合并HLA序列信息...")
            hla_seq_dict = dict(zip(data["hla_seq"]["HLA"], data["hla_seq"]["pseudo_sequence"]))
            
            for split, df in trimer_data.items():
                if "HLA" in df.columns and "HLA_sequence" not in df.columns:
                    df["HLA_sequence"] = df["HLA"].map(hla_seq_dict)
        
        trimer_data["train"].to_csv(os.path.join(trimer_dir, "train.csv"), index=False)
        trimer_data["val"].to_csv(os.path.join(trimer_dir, "val.csv"), index=False)
        trimer_data["test"].to_csv(os.path.join(trimer_dir, "test.csv"), index=False)
        
        # 保存HLA序列数据
        if data["hla_seq"] is not None:
            logger.info("保存HLA序列数据...")
            data["hla_seq"].to_csv(os.path.join(output_dir, "hla_sequences.csv"), index=False)
        
        # 打印数据统计信息
        logger.info("\n数据统计信息:")
        logger.info(f"TCR-Pep训练集: {tcr_pep_data['train'].shape[0]}样本，正例比例: {tcr_pep_data['train']['Label'].mean():.4f}")
        logger.info(f"TCR-Pep验证集: {tcr_pep_data['val'].shape[0]}样本，正例比例: {tcr_pep_data['val']['Label'].mean():.4f}")
        logger.info(f"TCR-Pep测试集: {tcr_pep_data['test'].shape[0]}样本，正例比例: {tcr_pep_data['test']['Label'].mean():.4f}")
        
        logger.info(f"HLA-Pep训练集: {hla_pep_data['train'].shape[0]}样本，正例比例: {hla_pep_data['train']['Label'].mean():.4f}")
        logger.info(f"HLA-Pep验证集: {hla_pep_data['val'].shape[0]}样本，正例比例: {hla_pep_data['val']['Label'].mean():.4f}")
        logger.info(f"HLA-Pep测试集: {hla_pep_data['test'].shape[0]}样本，正例比例: {hla_pep_data['test']['Label'].mean():.4f}")
        
        logger.info(f"TCR-HLA-Pep训练集: {trimer_data['train'].shape[0]}样本，正例比例: {trimer_data['train']['Label'].mean():.4f}")
        logger.info(f"TCR-HLA-Pep验证集: {trimer_data['val'].shape[0]}样本，正例比例: {trimer_data['val']['Label'].mean():.4f}")
        logger.info(f"TCR-HLA-Pep测试集: {trimer_data['test'].shape[0]}样本，正例比例: {trimer_data['test']['Label'].mean():.4f}")
    
    else:
        raise ValueError("不支持的数据格式")


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
    
    # 添加HLA序列（如果有HLA列和HLA序列字典）
    if "HLA" in df.columns and hla_seq_dict and len(hla_seq_dict) > 0:
        logger.info(f"为{dataset_name}数据集添加HLA序列...")
        df["HLA_sequence"] = df["HLA"].map(hla_seq_dict)
    
    # 删除缺失值
    required_columns = [col for col in ["CDR3", "MT_pep", "HLA_sequence", "Label"] if col in df.columns]
    n_before = df.shape[0]
    df = df.dropna(subset=required_columns)
    n_after = df.shape[0]
    logger.info(f"删除缺失值后，{dataset_name}数据集从{n_before}行减少到{n_after}行")
    
    # 过滤序列长度
    max_tcr_len = config["data"]["max_tcr_len"]
    max_pep_len = config["data"]["max_pep_len"]
    max_hla_len = config["data"]["max_hla_len"]
    
    # 创建过滤条件
    filter_conditions = []
    
    if "CDR3" in df.columns:
        filter_conditions.append(df["CDR3"].str.len() <= max_tcr_len)
    
    if "MT_pep" in df.columns:
        filter_conditions.append(df["MT_pep"].str.len() <= max_pep_len)
    
    if "HLA_sequence" in df.columns:
        filter_conditions.append(df["HLA_sequence"].str.len() <= max_hla_len)
    
    # 应用过滤条件
    n_before = df.shape[0]
    if filter_conditions:
        df = df[pd.concat(filter_conditions, axis=1).all(axis=1)]
    n_after = df.shape[0]
    logger.info(f"过滤序列长度后，{dataset_name}数据集从{n_before}行减少到{n_after}行")
    
    return df


def preprocess_data(data_dir: str, 
                   output_dir: str, 
                   config: Dict, 
                   mode: str, 
                   negative_ratio: float = 1.0,
                   split: bool = False, 
                   clustering: bool = False) -> None:
    """
    预处理数据
    
    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        config: 配置字典
        mode: 预处理模式，'tcr_pep'、'hla_pep'或'trimer'
        negative_ratio: 阴性样本相对于阳性样本的比例（仅用于二元模型）
        split: 是否拆分数据集
        clustering: 是否使用聚类进行数据拆分
    """
    logger = logging.getLogger("preprocess")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 根据模式加载数据
    if mode in ['tcr_pep', 'hla_pep']:
        # 二元模型数据加载
        logger.info(f"加载{mode}数据（二元模型）...")
        try:
            data = load_binary_data(data_dir, mode, negative_ratio)
            
            # 处理数据
            logger.info(f"处理{mode}数据并保存到{output_dir}...")
            process_data(data, output_dir, config)
            
        except Exception as e:
            logger.error(f"处理{mode}数据失败: {str(e)}")
            raise
    
    elif mode == 'trimer':
        # 三元模型数据加载
        logger.info("加载三元模型数据...")
        try:
            data = load_trimer_data(data_dir)
            
            # 处理数据
            logger.info(f"处理三元模型数据并保存到{output_dir}...")
            process_data(data, output_dir, config)
            
        except Exception as e:
            logger.error(f"处理三元模型数据失败: {str(e)}")
            raise
    
    else:
        raise ValueError(f"不支持的预处理模式: {mode}")
    
    logger.info("数据预处理完成！")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理TCR-HLA-Pep数据")
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
    logger = setup_logger("data_processor", os.path.join(args.log_dir, "data_processor.log"))
    
    # 加载配置
    if args.config is not None and os.path.exists(args.config):
        logger.info(f"从{args.config}加载配置...")
        config = load_config(args.config)
    else:
        logger.info("使用默认配置...")
        config = get_default_config()
    
    # 预处理数据
    logger.info(f"开始预处理{args.mode}数据...")
    preprocess_data(
        args.data_dir, 
        args.output_dir, 
        config, 
        args.mode,
        args.negative_ratio,
        args.split, 
        args.clustering
    )
    
    logger.info("数据处理完成！")


if __name__ == "__main__":
    main() 