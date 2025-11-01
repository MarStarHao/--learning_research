# 快速入门指南

本指南将帮助您快速上手条件扩散模型用于时间序列表征学习。

## 📦 安装

```bash
cd conditional_diffusion_ts
pip install -r requirements.txt
```

## 🎯 5分钟快速体验

### 1. 运行基本示例

```bash
cd examples
python basic_usage.py
```

这将：
- ✅ 创建合成时间序列数据
- ✅ 初始化条件扩散模型
- ✅ 提取时间序列表征
- ✅ 生成新的时间序列样本
- ✅ 重构输入时间序列
- ✅ 进行简短的训练演示

### 2. 完整训练（默认配置）

```bash
python train.py \
    --num_samples 10000 \
    --seq_len 100 \
    --batch_size 64 \
    --num_epochs 100 \
    --encoder_type transformer
```

训练将自动：
- 创建合成数据集
- 初始化模型和优化器
- 定期保存检查点
- 记录训练和验证损失

### 3. 评估模型

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_samples 1000 \
    --output_dir ./outputs \
    --visualize_tsne
```

评估将生成：
- 重构质量分析
- 生成样本可视化
- 表征空间可视化（PCA和t-SNE）

## 🎨 使用不同配置

### Transformer编码器（默认）

```bash
python train.py --config configs/default_config.yaml
```

**优点**：最佳性能，捕捉长程依赖  
**缺点**：计算量大

### 卷积编码器

```bash
python train.py --config configs/conv_config.yaml
```

**优点**：计算高效，适合长序列  
**缺点**：感受野有限

### 对比学习模式

```bash
python train.py --config configs/contrastive_config.yaml
```

**优点**：学习更好的表征  
**应用**：下游分类、聚类任务

## 💻 Python API 使用

### 基本使用

```python
import torch
from models import ConditionalDiffusionTS

# 创建模型
model = ConditionalDiffusionTS(
    input_dim=1,
    seq_len=100,
    encoder_type='transformer',
    encoder_dim=256
)

# 输入数据
x = torch.randn(32, 100, 1)  # [batch, seq_len, features]

# 训练
outputs = model(x, return_loss=True)
loss = outputs['loss']
loss.backward()

# 提取表征
representation = model.get_representation(x)  # [batch, encoder_dim]
```

### 生成新样本

```python
model.eval()
with torch.no_grad():
    # 无条件生成
    generated = model.generate(batch_size=10, device='cuda')
    
    # 条件生成
    condition = model.encode(reference_data)
    generated = model.generate(
        batch_size=10, 
        condition=condition,
        device='cuda'
    )
```

### 重构时间序列

```python
with torch.no_grad():
    # 部分去噪重构
    reconstructed = model.reconstruct(x, num_steps=500)
    
    # 完全去噪重构
    reconstructed = model.reconstruct(x, num_steps=1000)
```

## 🔧 自定义数据集

```python
from data import TimeSeriesDataset
import numpy as np

# 加载您的数据
data = np.load('your_data.npy')  # shape: [num_samples, feature_dim]

# 创建数据集
dataset = TimeSeriesDataset(
    data=data,
    seq_len=100,
    stride=1,
    normalize=True
)

# 创建数据加载器
from data import create_dataloader
loader = create_dataloader(dataset, batch_size=32)

# 使用数据加载器训练
for batch in loader:
    x = batch['sequence']
    # ... 训练代码
```

## 📊 监控训练

### 检查点文件

训练过程会在 `checkpoint_dir` 中保存：

- `latest_checkpoint.pt`: 最新检查点
- `best_model.pt`: 验证集上最佳模型
- `checkpoint_epoch_N.pt`: 每个epoch的检查点

### 加载检查点继续训练

```python
from utils import Trainer

trainer = Trainer(...)
trainer.load_checkpoint('./checkpoints/latest_checkpoint.pt')
trainer.train(num_epochs=50)  # 继续训练
```

## 🎓 进阶用法

### 1. 自定义编码器

```python
import torch.nn as nn
from models import ConditionalDiffusionTS

class CustomEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 创建模型时使用自定义编码器
model = ConditionalDiffusionTS(...)
model.encoder = CustomEncoder(input_dim=1, output_dim=256)
```

### 2. 调整扩散过程

```python
# 使用余弦噪声调度
model = ConditionalDiffusionTS(
    schedule_type='cosine',  # 'linear', 'cosine', 'quadratic'
    num_timesteps=1000,
    beta_start=1e-4,
    beta_end=0.02
)
```

### 3. 对比学习

```python
from utils import ContrastiveTrainer
from data import ContrastiveTimeSeriesDataset

# 包装数据集
train_dataset = ContrastiveTimeSeriesDataset(
    base_dataset=your_dataset,
    augmentation_prob=0.8
)

# 使用对比学习训练器
trainer = ContrastiveTrainer(
    model=model,
    train_loader=train_loader,
    temperature=0.07  # 对比学习温度参数
)
```

## ❓ 常见问题

### Q: 训练很慢怎么办？

A: 尝试以下方法：
- 减少 `num_timesteps`（例如从1000减到500）
- 使用卷积编码器而不是Transformer
- 减少 `encoder_layers`
- 增加 `batch_size`（如果GPU内存允许）

### Q: 如何提升生成质量？

A: 建议：
- 使用余弦噪声调度（`schedule_type='cosine'`）
- 增加训练轮数
- 使用对比学习
- 增加模型容量（`encoder_dim`, `encoder_layers`）

### Q: 如何应用到下游任务？

A: 使用表征进行分类/回归：

```python
# 提取表征
representations = model.get_representation(x)

# 训练分类器
classifier = nn.Linear(encoder_dim, num_classes)
logits = classifier(representations)
```

## 📚 更多资源

- [完整文档](README.md)
- [模型架构说明](README.md#模型架构)
- [配置文件说明](README.md#配置说明)

## 🎉 开始您的实验！

现在您已经准备好开始实验了。祝您使用愉快！

如有问题，请查看完整文档或提交Issue。
