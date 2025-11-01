"""
时间序列数据集加载和预处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """
    时间序列数据集
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int,
        stride: int = 1,
        normalize: bool = True,
        labels: Optional[np.ndarray] = None
    ):
        """
        Args:
            data: 时间序列数据 [num_samples, feature_dim] 或 [num_samples, seq_len, feature_dim]
            seq_len: 序列长度
            stride: 滑动窗口步长
            normalize: 是否标准化
            labels: 标签（可选）
        """
        self.seq_len = seq_len
        self.stride = stride
        self.labels = labels
        
        # 处理数据形状
        if len(data.shape) == 2:
            # 如果是2D数据，使用滑动窗口切分
            self.sequences = self._create_sequences(data)
        else:
            # 如果已经是3D数据，直接使用
            self.sequences = data
        
        # 标准化
        if normalize:
            self.scaler = StandardScaler()
            original_shape = self.sequences.shape
            # 重塑为2D进行标准化
            self.sequences = self.sequences.reshape(-1, original_shape[-1])
            self.sequences = self.scaler.fit_transform(self.sequences)
            # 重塑回3D
            self.sequences = self.sequences.reshape(original_shape)
        else:
            self.scaler = None
        
        self.sequences = torch.FloatTensor(self.sequences)
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        使用滑动窗口创建序列
        
        Args:
            data: [num_samples, feature_dim]
            
        Returns:
            sequences: [num_sequences, seq_len, feature_dim]
        """
        sequences = []
        for i in range(0, len(data) - self.seq_len + 1, self.stride):
            sequences.append(data[i:i + self.seq_len])
        return np.array(sequences)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        sample = {"sequence": self.sequences[idx]}
        
        if self.labels is not None:
            sample["label"] = self.labels[idx]
        
        return sample


class SyntheticTimeSeriesDataset(Dataset):
    """
    合成时间序列数据集（用于测试和演示）
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 100,
        feature_dim: int = 1,
        pattern_type: str = "sine"
    ):
        """
        Args:
            num_samples: 样本数量
            seq_len: 序列长度
            feature_dim: 特征维度
            pattern_type: 模式类型 ('sine', 'random_walk', 'periodic', 'mixed')
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.pattern_type = pattern_type
        
        self.data = self._generate_data()
        
    def _generate_data(self) -> torch.Tensor:
        """生成合成数据"""
        data = []
        
        for _ in range(self.num_samples):
            if self.pattern_type == "sine":
                # 正弦波
                t = np.linspace(0, 4 * np.pi, self.seq_len)
                freq = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                amplitude = np.random.uniform(0.5, 2.0)
                sample = amplitude * np.sin(freq * t + phase)
                
            elif self.pattern_type == "random_walk":
                # 随机游走
                sample = np.cumsum(np.random.randn(self.seq_len))
                
            elif self.pattern_type == "periodic":
                # 周期性模式
                period = np.random.randint(10, 30)
                sample = np.tile(np.random.randn(period), self.seq_len // period + 1)[:self.seq_len]
                
            elif self.pattern_type == "mixed":
                # 混合模式
                t = np.linspace(0, 4 * np.pi, self.seq_len)
                sine_component = np.sin(t)
                trend_component = np.linspace(0, 1, self.seq_len)
                noise_component = 0.1 * np.random.randn(self.seq_len)
                sample = sine_component + trend_component + noise_component
                
            else:
                raise ValueError(f"Unknown pattern type: {self.pattern_type}")
            
            # 扩展到多维
            if self.feature_dim > 1:
                sample = np.tile(sample.reshape(-1, 1), (1, self.feature_dim))
                # 添加一些随机扰动
                sample += 0.1 * np.random.randn(self.seq_len, self.feature_dim)
            else:
                sample = sample.reshape(-1, 1)
            
            data.append(sample)
        
        return torch.FloatTensor(np.array(data))
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        return {"sequence": self.data[idx]}


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作线程数
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )


def load_ucr_dataset(
    dataset_name: str,
    data_path: str = "./data/UCR"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载UCR时间序列数据集
    
    Args:
        dataset_name: 数据集名称
        data_path: 数据路径
        
    Returns:
        train_data, train_labels, test_data, test_labels
    """
    import os
    
    train_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TRAIN.tsv")
    test_file = os.path.join(data_path, dataset_name, f"{dataset_name}_TEST.tsv")
    
    # 加载训练数据
    train_data = np.loadtxt(train_file, delimiter="\t")
    train_labels = train_data[:, 0]
    train_data = train_data[:, 1:]
    
    # 加载测试数据
    test_data = np.loadtxt(test_file, delimiter="\t")
    test_labels = test_data[:, 0]
    test_data = test_data[:, 1:]
    
    return train_data, train_labels, test_data, test_labels


class ContrastiveTimeSeriesDataset(Dataset):
    """
    对比学习数据集
    为每个样本生成正样本对（通过数据增强）
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        augmentation_prob: float = 0.8
    ):
        """
        Args:
            base_dataset: 基础数据集
            augmentation_prob: 增强概率
        """
        self.base_dataset = base_dataset
        self.augmentation_prob = augmentation_prob
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        数据增强
        
        Args:
            x: [seq_len, feature_dim]
            
        Returns:
            增强后的序列
        """
        # 添加高斯噪声
        if np.random.rand() < self.augmentation_prob:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
        
        # 时间扭曲（缩放）
        if np.random.rand() < self.augmentation_prob:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
        
        # 随机裁剪和填充
        if np.random.rand() < self.augmentation_prob:
            seq_len = x.shape[0]
            crop_len = int(seq_len * np.random.uniform(0.8, 1.0))
            start = np.random.randint(0, seq_len - crop_len + 1)
            x_crop = x[start:start + crop_len]
            
            # 插值到原始长度
            x = torch.nn.functional.interpolate(
                x_crop.T.unsqueeze(0),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).squeeze(0).T
        
        return x
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> dict:
        sample = self.base_dataset[idx]
        x = sample["sequence"]
        
        # 生成两个增强视图
        x1 = self._augment(x.clone())
        x2 = self._augment(x.clone())
        
        result = {
            "sequence": x,
            "augmented_1": x1,
            "augmented_2": x2
        }
        
        if "label" in sample:
            result["label"] = sample["label"]
        
        return result
