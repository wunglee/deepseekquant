"""
[专家完整版 + 专家碎片] Data模块 - 数据获取与质量管理

状态: 方案B执行完成,碎片已修复
更新时间: 2025-11-27

模块结构:
1. 专家完整版 (core_bak/data_fetcher.py -> data_fetcher.py)
   - DataFetcher: 完整的生产级数据获取器 (8652行)
   - DataQualityMonitor: 增强版质量监控器 (推荐)
   - DataQualityMonitorBasic: 基础版质量监控器
   - DataValidator: 数据验证器
   - DataQualityMonitorFactory: 监控器工厂
   - DeepSeekQuantSystem: 系统集成类

2. 专家碎片 (增量功能,待评审)
   - DataQualityEnhancer: 多源数据智能切换 (第6轮专家)
   - RealHistoricalDataProvider: 历史数据提供者 (第2轮专家 + Phase 5B-5)
   - YahooFinanceDataProvider: Yahoo数据源 (第6轮专家)

TODO: 后续需对比碎片与完整版的功能重叠,提取增量功能
"""

from .data_fetcher import (
    DataFetcher,
    DataQualityMonitor,  # 增强版,推荐使用
    DataQualityMonitorBasic,  # 基础版
    DataValidator,
    DataQualityMonitorFactory,
    DeepSeekQuantSystem,
)

# 从共享模块导入枚举和数据模型
from core_bak_refactored.core.share import DataSourceType, DataFrequency, MarketData

from .data_quality_enhancer import (
    DataQualityEnhancer,  # 第6轮专家碎片 - 多源智能切换
    DataQualityReport,
)

from .providers.historical_data_provider import (
    HistoricalDataProvider,  # 协议接口
    MockHistoricalDataProvider,  # Mock实现
    # RealHistoricalDataProvider,  # TODO: 实现后导出
)

from .providers.yahoo_finance import (
    YahooFinanceDataProvider,  # 第6轮专家碎片 - Yahoo数据源
)

__all__ = [
    # 专家完整版
    'DataFetcher',
    'DataQualityMonitor',
    'DataQualityMonitorBasic',
    'DataValidator',
    'DataQualityMonitorFactory',
    'DeepSeekQuantSystem',
    'DataSourceType',
    'DataFrequency',
    'MarketData',
    
    # 专家碎片
    'DataQualityEnhancer',
    'DataQualityReport',
    'HistoricalDataProvider',
    'MockHistoricalDataProvider',
    'YahooFinanceDataProvider',
]
