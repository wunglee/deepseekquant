"""
核心共享模块

职责：
- 提供跨模块共享的业务逻辑组件
- 统一管理共享枚举、配置和工具函数
"""

from .market.market_config import MarketConfig, MarketCode
from .market.market_enums import DataSource
from .exchange_rates import MockExchangeRateAdapter, CurrencyConverter
from .models import MarketData
from .config_manager import ConfigManager, MonitoringConfig, AlertingConfig, ProvidersConfig, SystemConfig, CacheConfig


__all__ = [
    # 市场配置
    'MarketConfig',
    'MarketCode',
    
    # 市场枚举
    'DataSource',
    
    # 汇率管理
    'MockExchangeRateAdapter',
    'CurrencyConverter',
    
    # 配置管理
    'ConfigManager',
    'MonitoringConfig',
    'AlertingConfig',
    'ProvidersConfig',
    'SystemConfig',
    'CacheConfig',
    
    # 数据模型
    'MarketData',
]
