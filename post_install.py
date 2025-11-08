#!/usr/bin/env python3
"""
DeepSeekQuant 环境安装后配置脚本
"""

import os
import sys
import subprocess
import nltk


def download_nltk_data():
    """下载NLTK数据"""
    print("下载NLTK数据...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✓ NLTK数据下载完成")
    except Exception as e:
        print(f"✗ NLTK数据下载失败: {e}")


def validate_environment():
    """验证环境配置"""
    print("验证环境配置...")

    # 检查关键包
    packages = [
        'numpy', 'pandas', 'tensorflow', 'torch', 'sklearn',
        'yfinance', 'sqlalchemy', 'fastapi'
    ]

    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} 导入成功")
        except ImportError as e:
            print(f"✗ {package} 导入失败: {e}")

    # 检查TensorFlow版本
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow版本: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow未安装")

    # 检查PyTorch版本
    try:
        import torch
        print(f"✓ PyTorch版本: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch未安装")


def setup_environment_variables():
    """设置环境变量"""
    print("设置环境变量...")

    # 在用户目录创建.env文件
    env_content = """
# DeepSeekQuant 环境配置
DEEPSEEKQUANT_ENV=development
PYTHONPATH=.
PYTHONUNBUFFERED=1

# 数据库配置（根据需要修改）
DATABASE_URL=sqlite:///deepseekquant.db

# API密钥配置（根据需要添加）
# ALPHA_VANTAGE_API_KEY=your_key_here
# YAHOO_API_KEY=your_key_here
"""

    env_file = os.path.expanduser("~/.deepseekquant_env")
    with open(env_file, 'w') as f:
        f.write(env_content)

    print(f"✓ 环境变量文件已创建: {env_file}")


def main():
    """主函数"""
    print("=" * 50)
    print("DeepSeekQuant 环境安装后配置")
    print("=" * 50)

    download_nltk_data()
    print()

    validate_environment()
    print()

    setup_environment_variables()
    print()

    print("=" * 50)
    print("安装后配置完成！")
    print("=" * 50)

    # 使用说明
    print("\n使用说明:")
    print("1. 激活环境: conda activate deepseekquant")
    print("2. 设置环境变量: source ~/.deepseekquant_env")
    print("3. 运行测试: python -m pytest tests/")
    print("4. 启动服务: python main.py")


if __name__ == "__main__":
    main()