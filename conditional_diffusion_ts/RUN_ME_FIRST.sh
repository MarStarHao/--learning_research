#!/bin/bash

echo "=========================================="
echo "条件扩散模型用于时间序列表征学习"
echo "快速启动脚本"
echo "=========================================="
echo ""

# 检查Python环境
echo "1. 检查Python环境..."
python3 --version
echo ""

# 安装依赖
echo "2. 安装依赖（如需要）..."
echo "   运行: pip install -r requirements.txt"
echo ""

# 快速示例
echo "3. 快速体验（推荐首次使用）..."
echo "   运行: python examples/basic_usage.py"
echo ""

# 训练
echo "4. 开始训练..."
echo "   运行: python train.py --config configs/default_config.yaml"
echo ""

# 评估
echo "5. 评估模型..."
echo "   运行: python evaluate.py --checkpoint ./checkpoints/best_model.pt"
echo ""

echo "=========================================="
echo "文档位置："
echo "  - 项目总览: ../conditional_diffusion_ts_overview.md"
echo "  - 快速入门: ./QUICKSTART.md"
echo "  - 完整文档: ./README.md"
echo "=========================================="
echo ""
echo "开始您的实验之旅吧！ 🚀"
echo ""
