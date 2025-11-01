"""
训练脚本
"""

import torch
import argparse
import yaml
import os
from pathlib import Path

from models import ConditionalDiffusionTS
from data import SyntheticTimeSeriesDataset, create_dataloader, ContrastiveTimeSeriesDataset
from utils import Trainer, ContrastiveTrainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练条件扩散时间序列模型')
    
    # 数据参数
    parser.add_argument('--data_type', type=str, default='synthetic', 
                       choices=['synthetic', 'custom'],
                       help='数据类型')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='合成数据样本数')
    parser.add_argument('--seq_len', type=int, default=100,
                       help='序列长度')
    parser.add_argument('--input_dim', type=int, default=1,
                       help='输入特征维度')
    parser.add_argument('--pattern_type', type=str, default='mixed',
                       choices=['sine', 'random_walk', 'periodic', 'mixed'],
                       help='合成数据模式类型')
    
    # 模型参数
    parser.add_argument('--encoder_type', type=str, default='transformer',
                       choices=['transformer', 'conv', 'mlp'],
                       help='编码器类型')
    parser.add_argument('--encoder_dim', type=int, default=256,
                       help='编码器维度')
    parser.add_argument('--encoder_layers', type=int, default=4,
                       help='编码器层数')
    parser.add_argument('--num_timesteps', type=int, default=1000,
                       help='扩散步数')
    parser.add_argument('--schedule_type', type=str, default='linear',
                       choices=['linear', 'cosine', 'quadratic'],
                       help='噪声调度类型')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--use_contrastive', action='store_true',
                       help='是否使用对比学习')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='检查点保存目录')
    parser.add_argument('--log_interval', type=int, default=100,
                       help='日志记录间隔')
    parser.add_argument('--eval_interval', type=int, default=1000,
                       help='评估间隔')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，加载配置
    if args.config is not None and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # 更新参数
        for key, value in config.items():
            setattr(args, key, value)
    
    return args


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    print("\n创建数据集...")
    if args.data_type == 'synthetic':
        # 合成数据集
        train_dataset = SyntheticTimeSeriesDataset(
            num_samples=args.num_samples,
            seq_len=args.seq_len,
            feature_dim=args.input_dim,
            pattern_type=args.pattern_type
        )
        
        val_dataset = SyntheticTimeSeriesDataset(
            num_samples=args.num_samples // 10,
            seq_len=args.seq_len,
            feature_dim=args.input_dim,
            pattern_type=args.pattern_type
        )
    else:
        raise NotImplementedError("自定义数据集加载尚未实现")
    
    # 如果使用对比学习，包装数据集
    if args.use_contrastive:
        print("使用对比学习模式")
        train_dataset = ContrastiveTimeSeriesDataset(train_dataset)
        val_dataset = ContrastiveTimeSeriesDataset(val_dataset)
    
    # 创建数据加载器
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 创建模型
    print("\n创建模型...")
    model = ConditionalDiffusionTS(
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        encoder_type=args.encoder_type,
        encoder_dim=args.encoder_dim,
        encoder_layers=args.encoder_layers,
        num_timesteps=args.num_timesteps,
        schedule_type=args.schedule_type
    )
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # 创建训练器
    print("\n创建训练器...")
    if args.use_contrastive:
        trainer = ContrastiveTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            checkpoint_dir=args.checkpoint_dir,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=str(device),
            checkpoint_dir=args.checkpoint_dir,
            log_interval=args.log_interval,
            eval_interval=args.eval_interval
        )
    
    # 开始训练
    print("\n" + "="*50)
    print("开始训练")
    print("="*50 + "\n")
    
    trainer.train(num_epochs=args.num_epochs)
    
    # 保存最终模型
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    model.save_pretrained(final_model_path)
    print(f"\n最终模型已保存到: {final_model_path}")


if __name__ == "__main__":
    main()
