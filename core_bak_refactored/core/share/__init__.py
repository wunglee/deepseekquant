"""
核心共享模块

职责：
- 提供跨模块共享的业务逻辑组件
- 统一管理共享枚举、配置和工具函数
"""

from .market.market_config import MarketConfigManager, MarketCode
from .market.market_enums import DataSource, REGIONAL_DATA_SOURCE_PRIORITY
from .exchange_rates import MockExchangeRateAdapter, CurrencyConverter
from .enums import DataSourceType, DataFrequency
from .models import MarketData
from .alert_enums import AlertSeverity, AlertChannel, DataQualityIssueType
from .config_manager import ConfigManager, MonitoringConfig, AlertingConfig, DataConfig, SystemConfig
from .performance_stats import PerformanceStatsManager, PerformanceMetrics

__all__ = [
    # 市场配置
    'MarketConfigManager',
    'MarketCode',
    
    # 市场枚举
    'DataSource',
    'REGIONAL_DATA_SOURCE_PRIORITY',
    
    # 汇率管理
    'MockExchangeRateAdapter',
    'CurrencyConverter',
    
    # 数据枚举
    'DataSourceType',
    'DataFrequency',
    
    # 告警枚举
    'AlertSeverity',
    'AlertChannel',
    'DataQualityIssueType',
    
    # 配置管理
    'ConfigManager',
    'MonitoringConfig',
    'AlertingConfig',
    'DataConfig',
    'SystemConfig',
    
    # 性能统计
    'PerformanceStatsManager',
    'PerformanceMetrics',
    
    # 数据模型
    'MarketData',
]
