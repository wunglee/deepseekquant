"""兼容旧测试：暴露core.signal.indicator_service接口"""
from core_bak_refactored.core.signal.indicator_service import TechnicalIndicators, MARKET_PARAMS

__all__ = ['TechnicalIndicators', 'MARKET_PARAMS']
