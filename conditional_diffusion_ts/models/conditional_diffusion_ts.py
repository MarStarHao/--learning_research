"""
条件扩散模型用于时间序列表征学习的完整模型
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .diffusion_process import DiffusionProcess
from .conditional_unet import ConditionalUNet1D
from .ts_encoder import TimeSeriesEncoder, ConvolutionalEncoder, MLPEncoder


class ConditionalDiffusionTS(nn.Module):
    """
    条件扩散模型用于时间序列表征学习
    
    该模型结合了：
    1. 时间序列编码器：提取时间序列的表征
    2. 扩散过程：学习数据的生成分布
    3. 条件UNet：基于条件的去噪模型
    """
    
    def __init__(
        self,
        # 数据参数
        input_dim: int,
        seq_len: int,
        
        # 编码器参数
        encoder_type: str = "transformer",  # "transformer", "conv", "mlp"
        encoder_dim: int = 256,
        encoder_layers: int = 4,
        
        # 扩散参数
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear",
        
        # UNet参数
        unet_hidden_dims: list = None,
        time_emb_dim: int = 128,
        
        # 其他参数
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        """
        初始化条件扩散时间序列模型
        
        Args:
            input_dim: 输入特征维度
            seq_len: 序列长度
            encoder_type: 编码器类型
            encoder_dim: 编码器输出维度
            encoder_layers: 编码器层数
            num_timesteps: 扩散步数
            beta_start: 初始噪声方差
            beta_end: 最终噪声方差
            schedule_type: 噪声调度类型
            unet_hidden_dims: UNet隐藏层维度
            time_emb_dim: 时间步嵌入维度
            dropout: Dropout比率
            use_attention: 是否使用注意力机制
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.encoder_dim = encoder_dim
        
        # 时间序列编码器（用于提取条件）
        if encoder_type == "transformer":
            self.encoder = TimeSeriesEncoder(
                input_dim=input_dim,
                d_model=encoder_dim,
                num_layers=encoder_layers,
                dropout=dropout
            )
        elif encoder_type == "conv":
            hidden_dims = [64, 128, encoder_dim]
            self.encoder = ConvolutionalEncoder(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout=dropout
            )
        elif encoder_type == "mlp":
            self.encoder = MLPEncoder(
                input_dim=input_dim,
                hidden_dim=encoder_dim,
                output_dim=encoder_dim,
                num_layers=encoder_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # 扩散过程
        self.diffusion = DiffusionProcess(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type
        )
        
        # 条件UNet去噪模型
        if unet_hidden_dims is None:
            unet_hidden_dims = [64, 128, 256, 512]
        
        self.denoiser = ConditionalUNet1D(
            input_dim=input_dim,
            hidden_dims=unet_hidden_dims,
            time_emb_dim=time_emb_dim,
            cond_dim=encoder_dim,
            dropout=dropout,
            use_attention=use_attention
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码时间序列，提取表征
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
            
        Returns:
            时间序列表征 [batch_size, seq_len, encoder_dim]
        """
        return self.encoder(x)
    
    def forward(
        self,
        x: torch.Tensor,
        return_loss: bool = True,
        condition: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
            return_loss: 是否返回训练损失
            condition: 外部条件（可选）
            
        Returns:
            包含loss和representation的字典
        """
        # 提取表征作为条件
        if condition is None:
            condition = self.encode(x)
        
        outputs = {"representation": condition}
        
        # 计算扩散损失
        if return_loss:
            loss = self.diffusion.compute_loss(
                model=self.denoiser,
                x_start=x,
                condition=condition
            )
            outputs["loss"] = loss
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        condition: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        生成时间序列
        
        Args:
            batch_size: 批次大小
            condition: 条件信息（可选）
            device: 设备
            
        Returns:
            生成的时间序列 [batch_size, seq_len, input_dim]
        """
        shape = (batch_size, self.seq_len, self.input_dim)
        
        # 使用扩散过程生成
        generated = self.diffusion.p_sample_loop(
            model=self.denoiser,
            shape=shape,
            condition=condition,
            device=device
        )
        
        return generated
    
    @torch.no_grad()
    def reconstruct(
        self,
        x: torch.Tensor,
        num_steps: int = None
    ) -> torch.Tensor:
        """
        重构时间序列（通过加噪再去噪）
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
            num_steps: 扩散步数（默认使用一半）
            
        Returns:
            重构的时间序列
        """
        if num_steps is None:
            num_steps = self.diffusion.num_timesteps // 2
        
        device = x.device
        batch_size = x.shape[0]
        
        # 提取条件
        condition = self.encode(x)
        
        # 前向扩散（加噪）
        t = torch.full((batch_size,), num_steps - 1, device=device, dtype=torch.long)
        x_noisy, _ = self.diffusion.q_sample(x, t)
        
        # 反向扩散（去噪）
        for i in reversed(range(num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_noisy = self.diffusion.p_sample(
                model=self.denoiser,
                x_t=x_noisy,
                t=t,
                condition=condition
            )
        
        return x_noisy
    
    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取时间序列的表征（用于下游任务）
        
        Args:
            x: 输入时间序列 [batch_size, seq_len, input_dim]
            
        Returns:
            表征向量 [batch_size, encoder_dim]
        """
        # 获取序列表征
        seq_repr = self.encode(x)
        
        # 全局平均池化
        global_repr = seq_repr.mean(dim=1)
        
        return global_repr
    
    def save_pretrained(self, save_path: str):
        """保存预训练模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'seq_len': self.seq_len,
                'encoder_dim': self.encoder_dim,
            }
        }, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_pretrained(self, load_path: str):
        """加载预训练模型"""
        checkpoint = torch.load(load_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {load_path} 加载")
        return checkpoint.get('config', {})
