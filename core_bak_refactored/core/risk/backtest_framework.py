"""
历史回测框架 - 压力测试场景验证
从第5轮专家指导实施
职责: 历史事件回测、合成组合构造、预测误差验证

设计原则：
- 数据抽象层：预留真实数据集成点
- 配置驱动：Provider 选择由配置文件决定，不在代码中硬编码
- 接口稳定：真实数据集成时无需修改业务逻辑

使用示例：
    # 使用工厂模式 + 配置驱动
    from core_bak_refactored.core.data.providers.factory import get_global_factory
    from core_bak_refactored.core.share.config_manager import ConfigManager
    
    # 方式1: 直接指定 provider ID
    factory = get_global_factory()
    provider = factory.get('akshare')
    
    # 方式2: 从配置获取（推荐）
    config_manager = ConfigManager()
    market_config = config_manager.get_market_config()
    # 根据市场选择 provider
    provider_id = market_config.market_sources.get('CN', 'akshare')
    provider = factory.get(provider_id)
"""

import logging
from typing import Protocol

import pandas as pd

logger = logging.getLogger('DeepSeekQuant.BacktestFramework')


# =============================================================================
# 数据接口抽象层（预留真实数据集成点）
# =============================================================================

class HistoricalDataProvider(Protocol):
    """
    历史数据提供者接口（抽象层）
    
    设计目的：
    - 解耦业务逻辑与数据来源
    - 支持模拟数据（当前）和真实数据（未来）无缝切换
    - 为core_bak_refactored/core/data模块集成预留标准接口
    """
    
    def get_index_prices(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        获取指数价格数据
        
        Args:
            symbol: 指数代码（如'000300.SH'沪深300）
            start_date: 开始日期 (pd.Timestamp)
            end_date: 结束日期 (pd.Timestamp)
        
        Returns:
            DataFrame with columns: ['date', 'close', 'volume']
        """
        ...
    
    def get_index_returns(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        """
        获取指数收益率序列
        
        Args:
            symbol: 指数代码
            start_date: 开始日期 (pd.Timestamp)
            end_date: 结束日期 (pd.Timestamp)
        
        Returns:
            Series with date index and return values
        """
        ...


# =============================================================================
# 模拟数据提供者（Phase 3A实现）
# =============================================================================

# 注意：生产代码不再直接依赖测试目录中的 Mock 实现。
# 在测试环境中，可通过 monkeypatch 或依赖注入将 MockHistoricalDataProvider
# 作为 HistoricalDataProvider 的具体实现传入业务逻辑。


# =============================================================================
# 数据提供者工厂（统一使用 factory.py）
# =============================================================================

# ❌ 已废弃: create_data_provider() 函数（硬编码 provider 选择逻辑）
# ✅ 新方式: 直接使用 factory.get() + 配置驱动
#
# 迁移指南:
#   旧代码: provider = create_data_provider('auto')
#   新代码: 
#     from core_bak_refactored.core.data.providers.factory import get_global_factory
#     from core_bak_refactored.core.share.config_manager import ConfigManager
#     
#     config_manager = ConfigManager()
#     market_config = config_manager.get_market_config()
#     provider_id = market_config.market_sources.get('CN', 'akshare')  # 从配置读取
#     factory = get_global_factory()
#     provider = factory.get(provider_id)


# =============================================================================
# 合成组合构造器（基于专家answer.md 1.3节）
# =============================================================================


# =============================================================================
# 事件窗口回测引擎（基于专家answer.md 1.3节）
# =============================================================================


# =============================================================================
# 回测报告生成器
# =============================================================================

