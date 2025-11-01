"""
条件扩散模型用于时间序列表征学习
Conditional Diffusion Models for Time Series Representation Learning
"""

__version__ = '1.0.0'
__author__ = 'Your Name'

from .models import (
    ConditionalDiffusionTS,
    DiffusionProcess,
    TimeSeriesEncoder,
    ConvolutionalEncoder,
    MLPEncoder,
    ConditionalUNet1D
)

from .data import (
    TimeSeriesDataset,
    SyntheticTimeSeriesDataset,
    ContrastiveTimeSeriesDataset,
    create_dataloader
)

from .utils import (
    Trainer,
    ContrastiveTrainer
)

__all__ = [
    # Models
    'ConditionalDiffusionTS',
    'DiffusionProcess',
    'TimeSeriesEncoder',
    'ConvolutionalEncoder',
    'MLPEncoder',
    'ConditionalUNet1D',
    # Data
    'TimeSeriesDataset',
    'SyntheticTimeSeriesDataset',
    'ContrastiveTimeSeriesDataset',
    'create_dataloader',
    # Utils
    'Trainer',
    'ContrastiveTrainer',
]
