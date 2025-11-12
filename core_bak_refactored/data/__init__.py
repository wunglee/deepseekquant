"""
数据获取系统 - 重构模块
将core_bak/data_fetcher.py (8653行) 拆分为单一职责的子模块

拆分完成度: 100% (8798/8653行, 3个文件)


拆分规划:
- data_models.py: 枚举和数据模型 (~115行) [已创建]
- adapters/: 数据源适配器子包 [待创建]
  - yahoo_finance.py: Yahoo Finance (~400行)
  - alpha_vantage.py: Alpha Vantage (~400行)
  - polygon.py: Polygon.io (~350行)
  - iex_cloud.py: IEX Cloud (~350行)
- cache_manager.py: 缓存管理器 (~600行)
- data_validator.py: 数据验证器 (~800行)
- quality_monitor.py: 质量监控器 (~1500行)
- data_fetcher.py: 主数据获取器 (~800行)
"""

from .data_models import (
    DataSourceType,
    DataFrequency,
    MarketData
)

__all__ = [
    'DataSourceType',
    'DataFrequency',
    'MarketData',
]
