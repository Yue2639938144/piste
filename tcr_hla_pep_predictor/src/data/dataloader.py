"""
数据加载器模块

该模块提供数据集和数据加载器的实现，支持TCR-Pep、HLA-Pep和TCR-HLA-Pep三种数据类型。
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Dict, List, Tuple, Union, Optional, Any
from .preprocessing import preprocess_sequences, encode_amino_acids, generate_masks


class MyDataSet(Dataset):
    """
    自定义数据集类，支持TCR-Pep、HLA-Pep和TCR-HLA-Pep三种数据类型
    """
    
    def __init__(self, 
                 data_df: pd.DataFrame, 
                 data_type: str = 'trimer',
                 max_tcr_len: int = 30,
                 max_pep_len: int = 15,
                 max_hla_len: int = 34):
        """
        初始化数据集
        
        Args:
            data_df: 包含序列和标签的DataFrame
            data_type: 数据类型，可选值为'tcr_pep'、'hla_pep'或'trimer'
            max_tcr_len: TCR序列最大长度
            max_pep_len: 肽段序列最大长度
            max_hla_len: HLA序列最大长度
        """
        self.data_df = data_df
        self.data_type = data_type.lower()
        self.max_tcr_len = max_tcr_len
        self.max_pep_len = max_pep_len
        self.max_hla_len = max_hla_len
        
        # 验证数据类型
        valid_types = ['tcr_pep', 'hla_pep', 'trimer']
        if self.data_type not in valid_types:
            raise ValueError(f"数据类型必须是{valid_types}中的一种")
        
        # 验证必要的列是否存在
        required_columns = {
            'tcr_pep': ['CDR3', 'MT_pep', 'Label'],
            'hla_pep': ['HLA_sequence', 'MT_pep', 'Label'],
            'trimer': ['CDR3', 'MT_pep', 'HLA_sequence', 'Label']
        }
        
        missing_cols = [col for col in required_columns[self.data_type] if col not in data_df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必要的列: {missing_cols}")
        
        # 预处理序列
        if 'CDR3' in data_df.columns:
            self.tcr_seqs = preprocess_sequences(data_df['CDR3'].tolist())
        if 'MT_pep' in data_df.columns:
            self.pep_seqs = preprocess_sequences(data_df['MT_pep'].tolist())
        if 'HLA_sequence' in data_df.columns:
            self.hla_seqs = preprocess_sequences(data_df['HLA_sequence'].tolist())
        
        # 标签
        self.labels = torch.tensor(data_df['Label'].values, dtype=torch.float32)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取指定索引的数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            包含编码序列和标签的字典
        """
        item = {}
        
        # 根据数据类型处理不同的序列组合
        if self.data_type in ['tcr_pep', 'trimer']:
            # TCR序列处理
            tcr_seq = self.tcr_seqs[idx]
            tcr_idx, tcr_biochem = encode_amino_acids([tcr_seq], self.max_tcr_len)
            tcr_mask = generate_masks([tcr_seq], self.max_tcr_len)
            
            item['tcr_idx'] = tcr_idx[0]
            item['tcr_biochem'] = tcr_biochem[0]
            item['tcr_mask'] = tcr_mask[0]
        
        if self.data_type in ['tcr_pep', 'hla_pep', 'trimer']:
            # 肽段序列处理
            pep_seq = self.pep_seqs[idx]
            pep_idx, pep_biochem = encode_amino_acids([pep_seq], self.max_pep_len)
            pep_mask = generate_masks([pep_seq], self.max_pep_len)
            
            item['pep_idx'] = pep_idx[0]
            item['pep_biochem'] = pep_biochem[0]
            item['pep_mask'] = pep_mask[0]
        
        if self.data_type in ['hla_pep', 'trimer']:
            # HLA序列处理
            hla_seq = self.hla_seqs[idx]
            hla_idx, hla_biochem = encode_amino_acids([hla_seq], self.max_hla_len)
            hla_mask = generate_masks([hla_seq], self.max_hla_len)
            
            item['hla_idx'] = hla_idx[0]
            item['hla_biochem'] = hla_biochem[0]
            item['hla_mask'] = hla_mask[0]
        
        # 标签
        item['label'] = self.labels[idx]
        
        return item


class DataLoader:
    """
    数据加载器，封装PyTorch的DataLoader，提供批量数据加载功能
    """
    
    def __init__(self, 
                 dataset: MyDataSet, 
                 batch_size: int = 64,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        """
        初始化数据加载器
        
        Args:
            dataset: 自定义数据集实例
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 数据加载的线程数
            pin_memory: 是否将数据加载到固定内存中
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        自定义批次整理函数，将单个样本组合成批次
        
        Args:
            batch: 样本列表
            
        Returns:
            批次数据字典
        """
        # 初始化批次字典
        batch_dict = {}
        
        # 获取第一个样本的键
        keys = batch[0].keys()
        
        # 对每个键，将所有样本的对应值堆叠起来
        for key in keys:
            if key == 'label':
                batch_dict[key] = torch.stack([sample[key] for sample in batch])
            else:
                batch_dict[key] = torch.stack([sample[key] for sample in batch])
        
        return batch_dict
    
    def __iter__(self):
        """返回数据加载器的迭代器"""
        return iter(self.data_loader)
    
    def __len__(self) -> int:
        """返回批次数量"""
        return len(self.data_loader) 