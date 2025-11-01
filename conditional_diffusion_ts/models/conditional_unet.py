"""
条件UNet去噪模型
用于扩散模型的反向过程，支持条件注入
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """时间步的正弦位置编码"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: [batch_size]
        Returns:
            [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock1D(nn.Module):
    """1D残差块，支持时间步和条件嵌入"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        cond_emb_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # 时间步嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        # 条件嵌入投影（如果提供）
        if cond_emb_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_emb_dim, out_channels)
            )
        else:
            self.cond_mlp = None
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        
        # 残差连接
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, seq_len]
            time_emb: [batch_size, time_emb_dim]
            cond_emb: [batch_size, cond_emb_dim] (可选)
        """
        residue = x
        
        # 第一层卷积
        x = self.conv1(x)
        x = self.norm1(x)
        
        # 添加时间步嵌入
        time_emb = self.time_mlp(time_emb)
        x = x + time_emb[:, :, None]
        
        # 添加条件嵌入（如果提供）
        if cond_emb is not None and self.cond_mlp is not None:
            cond_emb = self.cond_mlp(cond_emb)
            x = x + cond_emb[:, :, None]
        
        x = self.act(x)
        x = self.dropout(x)
        
        # 第二层卷积
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        
        # 残差连接
        return x + self.residual_conv(residue)


class AttentionBlock1D(nn.Module):
    """1D自注意力块"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, channels, seq_len]
        """
        batch_size, channels, seq_len = x.shape
        residue = x
        
        x = self.norm(x)
        qkv = self.qkv(x)
        
        # 重塑为多头注意力格式
        qkv = qkv.reshape(batch_size, 3, self.num_heads, channels // self.num_heads, seq_len)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # 计算注意力
        scale = (channels // self.num_heads) ** -0.5
        attn = torch.einsum('bhds,bhdt->bhst', q, k) * scale
        attn = torch.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.einsum('bhst,bhdt->bhds', attn, v)
        out = out.reshape(batch_size, channels, seq_len)
        
        out = self.proj(out)
        
        return out + residue


class ConditionalUNet1D(nn.Module):
    """
    条件UNet模型用于时间序列去噪
    支持时间步条件和额外的上下文条件
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [64, 128, 256, 512],
        time_emb_dim: int = 128,
        cond_dim: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 各层隐藏维度列表
            time_emb_dim: 时间步嵌入维度
            cond_dim: 条件嵌入维度（可选）
            num_heads: 注意力头数
            dropout: Dropout比率
            use_attention: 是否使用注意力机制
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim
        self.cond_dim = cond_dim
        
        # 时间步嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # 条件嵌入（如果提供）
        if cond_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, time_emb_dim),
                nn.SiLU(),
                nn.Linear(time_emb_dim, time_emb_dim)
            )
        else:
            self.cond_mlp = None
        
        # 初始卷积
        self.init_conv = nn.Conv1d(input_dim, hidden_dims[0], kernel_size=3, padding=1)
        
        # 下采样路径（编码器）
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_channels = hidden_dims[0]
        for i, out_channels in enumerate(hidden_dims):
            self.down_blocks.append(
                nn.ModuleList([
                    ResidualBlock1D(in_channels, out_channels, time_emb_dim, cond_dim, dropout),
                    ResidualBlock1D(out_channels, out_channels, time_emb_dim, cond_dim, dropout),
                    AttentionBlock1D(out_channels, num_heads) if use_attention else nn.Identity()
                ])
            )
            
            if i < len(hidden_dims) - 1:
                self.down_samples.append(nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
            else:
                self.down_samples.append(nn.Identity())
            
            in_channels = out_channels
        
        # 瓶颈层
        mid_channels = hidden_dims[-1]
        self.mid_block1 = ResidualBlock1D(mid_channels, mid_channels, time_emb_dim, cond_dim, dropout)
        self.mid_attn = AttentionBlock1D(mid_channels, num_heads) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock1D(mid_channels, mid_channels, time_emb_dim, cond_dim, dropout)
        
        # 上采样路径（解码器）
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        reversed_hidden_dims = list(reversed(hidden_dims))
        for i, out_channels in enumerate(reversed_hidden_dims):
            in_channels = reversed_hidden_dims[i]
            
            if i > 0:
                self.up_samples.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            else:
                self.up_samples.append(nn.Identity())
            
            # 跳跃连接使通道数翻倍
            self.up_blocks.append(
                nn.ModuleList([
                    ResidualBlock1D(in_channels + out_channels if i > 0 else in_channels, out_channels, 
                                   time_emb_dim, cond_dim, dropout),
                    ResidualBlock1D(out_channels, out_channels, time_emb_dim, cond_dim, dropout),
                    AttentionBlock1D(out_channels, num_heads) if use_attention else nn.Identity()
                ])
            )
        
        # 输出卷积
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[0]),
            nn.SiLU(),
            nn.Conv1d(hidden_dims[0], input_dim, kernel_size=3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
            time: 时间步 [batch_size]
            condition: 条件信息 [batch_size, cond_dim] (可选)
            
        Returns:
            预测的噪声 [batch_size, seq_len, input_dim]
        """
        # 转换为卷积格式
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        # 时间步嵌入
        time_emb = self.time_mlp(time)
        
        # 条件嵌入
        cond_emb = None
        if condition is not None and self.cond_mlp is not None:
            if len(condition.shape) == 3:  # [batch_size, seq_len, cond_dim]
                # 对序列条件进行池化
                cond_emb = condition.mean(dim=1)
            else:
                cond_emb = condition
            cond_emb = self.cond_mlp(cond_emb)
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 下采样路径
        skip_connections = []
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            block1, block2, attn = blocks
            x = block1(x, time_emb, cond_emb)
            x = block2(x, time_emb, cond_emb)
            x = attn(x)
            skip_connections.append(x)
            x = downsample(x)
        
        # 瓶颈层
        x = self.mid_block1(x, time_emb, cond_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb, cond_emb)
        
        # 上采样路径
        for blocks, upsample, skip in zip(self.up_blocks, self.up_samples, reversed(skip_connections)):
            x = upsample(x)
            # 跳跃连接
            if not isinstance(upsample, nn.Identity):
                x = torch.cat([x, skip], dim=1)
            block1, block2, attn = blocks
            x = block1(x, time_emb, cond_emb)
            x = block2(x, time_emb, cond_emb)
            x = attn(x)
        
        # 输出
        x = self.out_conv(x)
        
        # 转换回原始格式
        x = x.transpose(1, 2)  # [batch_size, seq_len, input_dim]
        
        return x
