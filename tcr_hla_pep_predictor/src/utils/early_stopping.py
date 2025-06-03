"""
早停机制模块

该模块提供训练过程中的早停功能，防止过拟合。
"""

import numpy as np
import torch
from typing import Optional, Dict, Any


class EarlyStopping:
    """
    早停机制类，监控验证指标，在指标不再改善时停止训练
    """
    
    def __init__(self, 
                 patience: int = 20, 
                 verbose: bool = True, 
                 delta: float = 0.0,
                 mode: str = 'max',
                 save_path: Optional[str] = None):
        """
        初始化早停机制
        
        Args:
            patience: 容忍验证指标不改善的轮数
            verbose: 是否打印早停信息
            delta: 判断指标改善的最小变化量
            mode: 监控模式，'min'表示指标越小越好，'max'表示指标越大越好
            save_path: 最佳模型保存路径，如果为None则不保存模型
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.mode = mode
        self.save_path = save_path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_state_dict = None
        
        # 根据模式设置比较函数
        if mode == 'min':
            self.is_better = lambda score, best: score < best - delta
        elif mode == 'max':
            self.is_better = lambda score, best: score > best + delta
        else:
            raise ValueError(f"模式必须是'min'或'max'，获得: {mode}")
    
    def __call__(self, epoch: int, score: float, model: torch.nn.Module) -> bool:
        """
        检查是否应该早停
        
        Args:
            epoch: 当前轮次
            score: 当前验证指标
            model: 当前模型
            
        Returns:
            是否应该早停
        """
        # 首次调用，初始化最佳分数
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
        # 检查当前分数是否比最佳分数更好
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
        # 分数没有改善
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # 达到容忍轮数，触发早停
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """
        保存最佳模型状态
        
        Args:
            model: 当前模型
        """
        # 保存模型状态字典的深拷贝
        self.best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        # 如果指定了保存路径，则保存模型
        if self.save_path is not None:
            if self.verbose:
                print(f'保存最佳模型到 {self.save_path}')
            torch.save({
                'epoch': self.best_epoch,
                'model_state_dict': self.best_state_dict,
                'score': self.best_score
            }, self.save_path)
    
    def load_best_model(self, model: torch.nn.Module) -> None:
        """
        加载最佳模型状态
        
        Args:
            model: 要加载状态的模型
        """
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
        elif self.save_path is not None and torch.cuda.is_available():
            checkpoint = torch.load(self.save_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        elif self.save_path is not None:
            checkpoint = torch.load(self.save_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise RuntimeError("没有可用的最佳模型状态")
    
    def get_best_score(self) -> float:
        """
        获取最佳分数
        
        Returns:
            最佳验证指标分数
        """
        return self.best_score
    
    def get_best_epoch(self) -> int:
        """
        获取最佳轮次
        
        Returns:
            最佳指标对应的轮次
        """
        return self.best_epoch 