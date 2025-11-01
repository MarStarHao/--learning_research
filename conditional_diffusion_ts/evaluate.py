"""
评估和可视化脚本
"""

import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from models import ConditionalDiffusionTS
from data import SyntheticTimeSeriesDataset, create_dataloader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='评估条件扩散时间序列模型')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='测试样本数')
    parser.add_argument('--seq_len', type=int, default=100,
                       help='序列长度')
    parser.add_argument('--input_dim', type=int, default=1,
                       help='输入特征维度')
    parser.add_argument('--pattern_type', type=str, default='mixed',
                       help='数据模式类型')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--num_generate', type=int, default=10,
                       help='生成的样本数')
    parser.add_argument('--visualize_tsne', action='store_true',
                       help='是否可视化t-SNE')
    
    return parser.parse_args()


@torch.no_grad()
def evaluate_reconstruction(model, dataloader, device):
    """评估重构性能"""
    model.eval()
    
    all_originals = []
    all_reconstructed = []
    reconstruction_errors = []
    
    print("评估重构性能...")
    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        
        # 重构
        reconstructed = model.reconstruct(sequences)
        
        # 计算重构误差
        mse = torch.mean((sequences - reconstructed) ** 2, dim=(1, 2))
        reconstruction_errors.extend(mse.cpu().numpy())
        
        all_originals.append(sequences.cpu())
        all_reconstructed.append(reconstructed.cpu())
    
    all_originals = torch.cat(all_originals, dim=0)
    all_reconstructed = torch.cat(all_reconstructed, dim=0)
    
    avg_mse = np.mean(reconstruction_errors)
    std_mse = np.std(reconstruction_errors)
    
    print(f"重构MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    
    return all_originals, all_reconstructed, reconstruction_errors


@torch.no_grad()
def extract_representations(model, dataloader, device):
    """提取时间序列表征"""
    model.eval()
    
    all_representations = []
    all_labels = []
    
    print("提取表征...")
    for batch in dataloader:
        sequences = batch["sequence"].to(device)
        
        # 获取表征
        representations = model.get_representation(sequences)
        all_representations.append(representations.cpu())
        
        if "label" in batch:
            all_labels.append(batch["label"])
    
    all_representations = torch.cat(all_representations, dim=0).numpy()
    
    if len(all_labels) > 0:
        all_labels = torch.cat(all_labels, dim=0).numpy()
    else:
        all_labels = None
    
    return all_representations, all_labels


def visualize_reconstruction(originals, reconstructed, num_samples=5, save_path=None):
    """可视化重构结果"""
    num_samples = min(num_samples, len(originals))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # 只显示第一个特征维度
        original = originals[i, :, 0].numpy()
        recon = reconstructed[i, :, 0].numpy()
        
        ax.plot(original, label='原始', linewidth=2, alpha=0.7)
        ax.plot(recon, label='重构', linewidth=2, alpha=0.7, linestyle='--')
        ax.set_xlabel('时间步')
        ax.set_ylabel('值')
        ax.set_title(f'样本 {i+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"重构可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


@torch.no_grad()
def generate_samples(model, num_samples, device):
    """生成新样本"""
    model.eval()
    
    print(f"生成 {num_samples} 个样本...")
    generated = model.generate(batch_size=num_samples, device=device)
    
    return generated


def visualize_generated(generated, num_samples=10, save_path=None):
    """可视化生成的样本"""
    num_samples = min(num_samples, len(generated))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 2 * num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        ax = axes[i]
        
        # 只显示第一个特征维度
        sample = generated[i, :, 0].cpu().numpy()
        
        ax.plot(sample, linewidth=2)
        ax.set_xlabel('时间步')
        ax.set_ylabel('值')
        ax.set_title(f'生成样本 {i+1}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"生成样本可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_representations(representations, labels=None, method='tsne', save_path=None):
    """可视化表征空间"""
    print(f"使用{method.upper()}降维可视化表征空间...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced = reducer.fit_transform(representations)
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, 
                            cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='标签')
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6, s=50)
    
    plt.xlabel(f'{method.upper()}维度 1')
    plt.ylabel(f'{method.upper()}维度 2')
    plt.title(f'时间序列表征空间 ({method.upper()})')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"表征可视化已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建测试数据集
    print("\n创建测试数据集...")
    test_dataset = SyntheticTimeSeriesDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        feature_dim=args.input_dim,
        pattern_type=args.pattern_type
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 加载模型
    print("\n加载模型...")
    model = ConditionalDiffusionTS(
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        encoder_type='transformer',
        encoder_dim=256,
        encoder_layers=4,
        num_timesteps=1000
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    print("模型加载成功")
    
    # 1. 评估重构
    print("\n" + "="*50)
    print("1. 评估重构性能")
    print("="*50)
    originals, reconstructed, errors = evaluate_reconstruction(model, test_loader, device)
    
    # 可视化重构
    recon_path = os.path.join(args.output_dir, 'reconstruction.png')
    visualize_reconstruction(originals, reconstructed, num_samples=5, save_path=recon_path)
    
    # 2. 生成新样本
    print("\n" + "="*50)
    print("2. 生成新样本")
    print("="*50)
    generated = generate_samples(model, args.num_generate, device)
    
    # 可视化生成
    gen_path = os.path.join(args.output_dir, 'generated_samples.png')
    visualize_generated(generated, num_samples=args.num_generate, save_path=gen_path)
    
    # 3. 提取和可视化表征
    print("\n" + "="*50)
    print("3. 提取和可视化表征")
    print("="*50)
    representations, labels = extract_representations(model, test_loader, device)
    
    # t-SNE可视化
    if args.visualize_tsne:
        tsne_path = os.path.join(args.output_dir, 'representations_tsne.png')
        visualize_representations(representations, labels, method='tsne', save_path=tsne_path)
    
    # PCA可视化
    pca_path = os.path.join(args.output_dir, 'representations_pca.png')
    visualize_representations(representations, labels, method='pca', save_path=pca_path)
    
    print("\n" + "="*50)
    print("评估完成！")
    print("="*50)
    print(f"所有结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
