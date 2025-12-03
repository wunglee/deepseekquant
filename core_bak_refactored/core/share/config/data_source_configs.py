"""
数据源配置

职责：
- 重导出区域数据源优先级配置
- 提供统一的配置访问接口

说明：
- REGIONAL_DATA_SOURCE_PRIORITY 实际定义在 core.share.market.market_enums
- 这里仅作为配置模块的统一导出点
"""

from core_bak_refactored.core.share.config_manager import ConfigManager

# 从YAML配置中加载区域数据源优先级
mgr = ConfigManager()
REGIONAL_DATA_SOURCE_PRIORITY = mgr.get('regional_data_source', {})

__all__ = ['REGIONAL_DATA_SOURCE_PRIORITY']
