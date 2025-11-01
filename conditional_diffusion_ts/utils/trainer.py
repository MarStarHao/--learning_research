"""
训练器类
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Dict, Any, Callable
import os
from tqdm import tqdm
import numpy as np


class Trainer:
    """
    条件扩散模型训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
    ):
        """
        Args:
            model: 条件扩散模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            checkpoint_dir: 检查点保存目录
            log_interval: 日志记录间隔
            eval_interval: 评估间隔
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        
        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
        
        # 学习率调度器
        self.scheduler = scheduler
        
        # 创建检查点目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 训练统计
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # 获取数据
            sequences = batch["sequence"].to(self.device)
            
            # 前向传播
            outputs = self.model(sequences, return_loss=True)
            loss = outputs["loss"]
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item()})
            
            # 记录日志
            if self.global_step % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                self.train_losses.append((self.global_step, avg_loss))
                print(f"Step {self.global_step}, Loss: {avg_loss:.6f}")
            
            # 验证
            if self.val_loader is not None and self.global_step % self.eval_interval == 0:
                val_loss = self.validate()
                self.val_losses.append((self.global_step, val_loss))
                print(f"Validation Loss: {val_loss:.6f}")
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                
                self.model.train()
        
        # 学习率调度
        if self.scheduler is not None:
            self.scheduler.step()
        
        return epoch_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """验证"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            sequences = batch["sequence"].to(self.device)
            
            outputs = self.model(sequences, return_loss=True)
            loss = outputs["loss"]
            
            val_loss += loss.item()
            num_batches += 1
        
        return val_loss / num_batches
    
    def train(self, num_epochs: int):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
        """
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            print(f"验证样本数: {len(self.val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            print(f"\nEpoch {epoch} 完成, 平均训练损失: {train_loss:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()
        
        print("训练完成！")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")
        
        # 保存epoch检查点
        epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{self.epoch}.pt')
        torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"从 {checkpoint_path} 加载检查点")
        print(f"Epoch: {self.epoch}, Global Step: {self.global_step}")


class ContrastiveTrainer(Trainer):
    """
    对比学习训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
        temperature: float = 0.07,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
            eval_interval=eval_interval,
        )
        self.temperature = temperature
    
    def contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失 (NT-Xent Loss)
        
        Args:
            z1: 第一个视图的表征 [batch_size, dim]
            z2: 第二个视图的表征 [batch_size, dim]
            
        Returns:
            对比损失
        """
        batch_size = z1.shape[0]
        
        # 归一化
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)
        
        # 计算相似度矩阵
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.mm(representations, representations.T)
        
        # 创建正样本mask
        mask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        mask = mask.repeat(2, 2)
        
        # 创建标签
        labels = torch.arange(batch_size, device=self.device)
        labels = torch.cat([labels + batch_size, labels])
        
        # 去除对角线（自己和自己的相似度）
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        # 计算损失
        similarity_matrix = similarity_matrix / self.temperature
        loss = nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def train_epoch(self) -> float:
        """训练一个epoch（对比学习版本）"""
        self.model.train()
        epoch_loss = 0.0
        epoch_diffusion_loss = 0.0
        epoch_contrastive_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in pbar:
            # 获取数据
            x1 = batch["augmented_1"].to(self.device)
            x2 = batch["augmented_2"].to(self.device)
            
            # 计算扩散损失
            outputs1 = self.model(x1, return_loss=True)
            outputs2 = self.model(x2, return_loss=True)
            diffusion_loss = (outputs1["loss"] + outputs2["loss"]) / 2
            
            # 计算对比损失
            z1 = outputs1["representation"].mean(dim=1)  # [batch_size, encoder_dim]
            z2 = outputs2["representation"].mean(dim=1)
            contrastive_loss = self.contrastive_loss(z1, z2)
            
            # 总损失
            loss = diffusion_loss + 0.1 * contrastive_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            epoch_loss += loss.item()
            epoch_diffusion_loss += diffusion_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'diff': diffusion_loss.item(),
                'contr': contrastive_loss.item()
            })
            
            # 记录和验证
            if self.global_step % self.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                print(f"\nStep {self.global_step}")
                print(f"  Total Loss: {avg_loss:.6f}")
                print(f"  Diffusion Loss: {epoch_diffusion_loss / num_batches:.6f}")
                print(f"  Contrastive Loss: {epoch_contrastive_loss / num_batches:.6f}")
            
            if self.val_loader is not None and self.global_step % self.eval_interval == 0:
                val_loss = self.validate()
                self.val_losses.append((self.global_step, val_loss))
                print(f"Validation Loss: {val_loss:.6f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(is_best=True)
                
                self.model.train()
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return epoch_loss / num_batches
