"""
数据获取器模块

职责：
- 批量数据获取（batch_fetcher.py）
- HTTP客户端封装（未来可扩展）
- 数据获取优化工具
"""

from .batch_fetcher import BatchDataFetcher

__all__ = [
    'BatchDataFetcher',
]
