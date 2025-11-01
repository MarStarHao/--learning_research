"""
条件扩散模型核心模块 - 扩散过程实现
实现了前向扩散（加噪）和反向扩散（去噪）过程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class DiffusionProcess(nn.Module):
    """
    扩散过程核心类
    实现DDPM (Denoising Diffusion Probabilistic Models)的扩散过程
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        """
        初始化扩散过程
        
        Args:
            num_timesteps: 扩散步数
            beta_start: 初始噪声方差
            beta_end: 最终噪声方差
            schedule_type: 噪声调度类型 ('linear', 'cosine', 'quadratic')
        """
        super().__init__()
        
        self.num_timesteps = num_timesteps
        
        # 根据不同调度类型生成beta序列
        if schedule_type == "linear":
            betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        elif schedule_type == "quadratic":
            betas = np.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=np.float32) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # 计算扩散过程中需要的各种系数
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        
        # 注册为buffer，方便GPU加速但不作为模型参数
        self.register_buffer("betas", torch.from_numpy(betas))
        self.register_buffer("alphas", torch.from_numpy(alphas))
        self.register_buffer("alphas_cumprod", torch.from_numpy(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", torch.from_numpy(alphas_cumprod_prev))
        
        # 前向扩散过程需要的系数
        self.register_buffer("sqrt_alphas_cumprod", torch.from_numpy(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.from_numpy(np.sqrt(1.0 - alphas_cumprod)))
        
        # 反向扩散过程需要的系数
        self.register_buffer("sqrt_recip_alphas", torch.from_numpy(np.sqrt(1.0 / alphas)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", torch.from_numpy(posterior_variance))
        
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> np.ndarray:
        """
        余弦噪声调度
        参考: Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021)
        """
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps, dtype=np.float32)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向扩散过程：q(x_t | x_0)
        在时间步t对原始数据x_0添加噪声
        
        Args:
            x_start: 原始数据 [batch_size, seq_len, feature_dim]
            t: 时间步 [batch_size]
            noise: 可选的噪声张量，如果为None则随机生成
            
        Returns:
            x_t: 加噪后的数据
            noise: 使用的噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 提取对应时间步的系数
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        反向扩散过程单步：p(x_{t-1} | x_t)
        从x_t预测x_{t-1}
        
        Args:
            model: 去噪模型
            x_t: 当前时间步的数据 [batch_size, seq_len, feature_dim]
            t: 当前时间步 [batch_size]
            condition: 条件信息（可选）
            
        Returns:
            x_{t-1}: 去噪后的数据
        """
        # 使用模型预测噪声
        if condition is not None:
            predicted_noise = model(x_t, t, condition)
        else:
            predicted_noise = model(x_t, t)
        
        # 提取系数
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        
        # 计算均值: mu = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * epsilon)
        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # 添加噪声（除了最后一步）
        if t[0] > 0:
            posterior_variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            x_prev = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x_prev = model_mean
        
        return x_prev
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        完整的反向扩散过程：从纯噪声生成数据
        
        Args:
            model: 去噪模型
            shape: 生成数据的形状
            condition: 条件信息（可选）
            device: 设备
            
        Returns:
            生成的数据
        """
        batch_size = shape[0]
        
        # 从纯噪声开始
        x = torch.randn(shape, device=device)
        
        # 逐步去噪
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, condition)
        
        return x
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
        """
        从系数数组a中提取时间步t对应的值，并reshape以匹配x的形状
        
        Args:
            a: 系数数组 [num_timesteps]
            t: 时间步索引 [batch_size]
            x_shape: 目标形状
            
        Returns:
            提取并reshape后的系数
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def compute_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        loss_type: str = "mse"
    ) -> torch.Tensor:
        """
        计算扩散模型的训练损失
        
        Args:
            model: 去噪模型
            x_start: 原始数据 [batch_size, seq_len, feature_dim]
            condition: 条件信息（可选）
            loss_type: 损失类型 ('mse', 'mae', 'huber')
            
        Returns:
            损失值
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # 随机采样时间步
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # 生成噪声
        noise = torch.randn_like(x_start)
        
        # 前向扩散
        x_t, _ = self.q_sample(x_start, t, noise)
        
        # 预测噪声
        if condition is not None:
            predicted_noise = model(x_t, t, condition)
        else:
            predicted_noise = model(x_t, t)
        
        # 计算损失
        if loss_type == "mse":
            loss = F.mse_loss(predicted_noise, noise)
        elif loss_type == "mae":
            loss = F.l1_loss(predicted_noise, noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(predicted_noise, noise)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return loss
