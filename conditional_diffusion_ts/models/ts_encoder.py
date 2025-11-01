"""
时间序列编码器
使用Transformer架构提取时间序列的表征
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    """
    时间序列编码器
    使用Transformer编码时间序列特征
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        """
        Args:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_len: 最大序列长度
        """
        super().__init__()
        
        self.d_model = d_model
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影层
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
            mask: 注意力mask（可选）
            
        Returns:
            编码后的表征 [batch_size, seq_len, d_model]
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x


class ConvolutionalEncoder(nn.Module):
    """
    基于卷积的时间序列编码器
    适用于长序列的高效编码
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 各层隐藏维度列表
            kernel_size: 卷积核大小
            dropout: Dropout比率
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
            
        Returns:
            编码后的表征 [batch_size, seq_len, output_dim]
        """
        # 转换为卷积格式 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 卷积编码
        x = self.encoder(x)
        
        # 转换回 [batch_size, seq_len, output_dim]
        x = x.transpose(1, 2)
        
        return x


class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.block(x))


class MLPEncoder(nn.Module):
    """
    基于MLP的简单编码器
    适用于短序列或作为基线模型
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
        Returns:
            编码后的表征 [batch_size, seq_len, output_dim]
        """
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_proj(x)
        
        return x
