"""
MarketData业务验证模块

职责：
- 验证MarketData对象的完整性和正确性
- 检查OHLC逻辑关系（金融数据特定规则）
- 支持批量数据验证和清洗

注意：
- 本模块专注于MarketData业务逻辑验证
- 通用技术验证（类型、长度、范围）请使用 infrastructure.type_validators
"""

from .validator import validate_market_data, validate_data_list, clean_market_data

__all__ = [
    'validate_market_data',
    'validate_data_list',
    'clean_market_data',
]
