"""
模型模块

该模块包含TCR-HLA-Pep预测模型的所有组件，包括编码器、注意力机制和预测模块。
"""

from .tcr_pep_model import TCRPepModel
from .hla_pep_model import HLAPepModel
from .trimer_model import TrimerModel
from .attention import PhysicalSlidingAttention, DataDrivenAttention, FusedAttention

__all__ = [
    'TCRPepModel',
    'HLAPepModel',
    'TrimerModel',
    'PhysicalSlidingAttention',
    'DataDrivenAttention',
    'FusedAttention'
] 