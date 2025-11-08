#!/usr/bin/env python3
"""
验证 DeepSeekQuant 环境安装
"""


def check_package(package_name):
    try:
        __import__(package_name)
        return True, f"✓ {package_name}"
    except ImportError as e:
        return False, f"✗ {package_name}: {e}"


def main():
    packages = [
        'numpy', 'pandas', 'tensorflow', 'torch', 'sklearn',
        'yfinance', 'alpha_vantage', 'backtrader', 'sqlalchemy',
        'fastapi', 'pydantic', 'pytest'
    ]

    print("验证包安装...")
    results = []
    for package in packages:
        results.append(check_package(package))

    for success, message in results:
        print(message)

    # 检查版本
    try:
        import tensorflow as tf
        print(f"TensorFlow版本: {tf.__version__}")
    except:
        print("TensorFlow未安装")

    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
    except:
        print("PyTorch未安装")


if __name__ == "__main__":
    main()