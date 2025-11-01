"""
基本使用示例
"""

import torch
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from models import ConditionalDiffusionTS
from data import SyntheticTimeSeriesDataset, create_dataloader
from utils import Trainer


def main():
    """基本使用示例"""
    
    print("=" * 60)
    print("条件扩散模型时间序列表征学习 - 基本使用示例")
    print("=" * 60)
    
    # 1. 创建数据集
    print("\n1. 创建数据集...")
    train_dataset = SyntheticTimeSeriesDataset(
        num_samples=1000,
        seq_len=100,
        feature_dim=1,
        pattern_type='mixed'
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )
    
    print(f"   训练样本数: {len(train_dataset)}")
    
    # 2. 创建模型
    print("\n2. 创建模型...")
    model = ConditionalDiffusionTS(
        input_dim=1,
        seq_len=100,
        encoder_type='transformer',
        encoder_dim=128,
        encoder_layers=2,
        num_timesteps=100  # 为了快速演示，使用较少的时间步
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   模型参数量: {num_params:,}")
    
    # 3. 测试前向传播
    print("\n3. 测试前向传播...")
    sample_batch = next(iter(train_loader))
    x = sample_batch['sequence']
    
    model.eval()
    with torch.no_grad():
        outputs = model(x, return_loss=True)
        print(f"   损失: {outputs['loss'].item():.6f}")
        print(f"   表征形状: {outputs['representation'].shape}")
    
    # 4. 提取表征
    print("\n4. 提取表征...")
    with torch.no_grad():
        representation = model.get_representation(x)
        print(f"   全局表征形状: {representation.shape}")
    
    # 5. 生成样本
    print("\n5. 生成样本...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    with torch.no_grad():
        generated = model.generate(batch_size=5, device=device)
        print(f"   生成样本形状: {generated.shape}")
    
    # 可视化生成的样本
    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        plt.plot(generated[i, :, 0].cpu().numpy())
        plt.title(f'生成样本 {i+1}')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=150)
    print("   生成样本已保存到: generated_samples.png")
    
    # 6. 重构样本
    print("\n6. 重构样本...")
    x_sample = x[:3].to(device)
    with torch.no_grad():
        reconstructed = model.reconstruct(x_sample, num_steps=50)
    
    # 可视化重构
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(x_sample[i, :, 0].cpu().numpy(), label='原始', alpha=0.7)
        plt.plot(reconstructed[i, :, 0].cpu().numpy(), label='重构', alpha=0.7, linestyle='--')
        plt.title(f'样本 {i+1}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reconstruction.png', dpi=150)
    print("   重构结果已保存到: reconstruction.png")
    
    # 7. 简短训练演示
    print("\n7. 简短训练演示（5个epoch）...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir='./demo_checkpoints',
        log_interval=10
    )
    
    trainer.train(num_epochs=5)
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
