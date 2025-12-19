"""
[重构新架构 + 专家碎片] Data模块 - 数据获取与质量管理

状态: ABCD重构流程执行中
更新时间: 2025-11-29

模块结构:
1. 重构新架构 (SOLID原则,职责单一) - 推荐使用
   - DataFetcherOrchestrator: 数据获取编排器 (仅负责路由与编排)
   - CacheManager: 三层缓存管理器 (L1内存+L2 LRU+L3 Redis)
   - MarketStatusService: 市场状态评估服务
   - FundamentalDataService: 基本面数据服务
   - AkshareProvider: AKShare数据源

2. 遗留组件 (data_fetcher.py) - 待删除
   - DataFetcher: 遗留的单体类，已被新架构替代
   - DataQualityMonitor: 质量监控器（待评估是否独立提取）
   - DataValidator: 数据验证器（待评估是否独立提取）
   注意：请直接使用 DataFetcherOrchestrator，不再导出 DataFetcher

3. 专家碎片 (增量功能,待评审)
   - DataQualityEnhancer: 多源数据智能切换 (第6轮专家)
   - RealHistoricalDataProvider: 历史数据提供者 (第2轮专家 + Phase 5B-5)
   - YahooFinanceDataProvider: Yahoo数据源 (第6轮专家)

TODO: 
- [ ] 删除 data_fetcher.py 中的 DataFetcher 类
- [ ] 评估 DataQualityMonitor/DataValidator 是否独立提取
- [ ] 对比碎片与完整版的功能重叠
"""

# 重构新架构 - SOLID原则（推荐使用）
from .fetcher.fetcher_orchestrator import DataFetcherOrchestrator
# 💚 修复：三层缓存架构（内部自动使用，不导出）
from core_bak_refactored.core.share.market.market_status_service import MarketStatusService


# 遗留组件（仅用于测试，不再导出）
# from .data_fetcher import DataFetcher  # 已被 DataFetcherOrchestrator 替代

# 从本模块导入枚举
from .enums import DataSourceType, DataFrequency, DataType, DataInterval, DataPeriod, DataFormat, DataQualityIssueType

# 从共享模块导入数据模型
from core_bak_refactored.core.share import MarketData

from .quality.data_quality_enhancer import (
    DataQualityEnhancer,  # 第6轮专家碎片 - 多源智能切换
    DataQualityReport,
)

from .providers.protocols import (
    HistoricalDataProvider,  # 协议接口（2025-12-02重构：从 historical_data_provider.py 提取）
)

from .providers.yahoo_provider import (
    YahooFinanceDataProvider,  # 第6轮专家碎片 - Yahoo数据源
)

# MarketData业务验证（从 infrastructure 迁移而来）
from .validation import (
    validate_market_data,
    validate_data_list,
    clean_market_data
)

__all__ = [
    # 重构新架构（推荐使用）
    'DataFetcherOrchestrator',
    # 💚 三层缓存架构（内部自动使用）
    'MarketStatusService',
    
    # 数据枚举
    'DataSourceType',
    'DataFrequency',
    'DataType',
    'DataInterval',
    'DataPeriod',
    'DataFormat',
    'DataQualityIssueType',
    
    # 共享模型
    'MarketData',
    
    # 专家碎片
    'DataQualityEnhancer',
    'DataQualityReport',
    'HistoricalDataProvider',  # Protocol接口
   # Deleted: 'RealHistoricalDataProvider' 已删除
    'YahooFinanceDataProvider',
    
    # MarketData业务验证
    'validate_market_data',
    'validate_data_list',
    'clean_market_data',
]
