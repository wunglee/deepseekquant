"""
数据验证基础设施模块

职责：
- 提供通用的数据验证逻辑
- 验证MarketData对象的完整性和正确性
- 支持批量数据验证和清洗

从 core/data/validation 迁移而来
"""

from .validator import validate_market_data, validate_data_list, clean_market_data

__all__ = [
    'validate_market_data',
    'validate_data_list',
    'clean_market_data',
]
