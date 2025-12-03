"""
共享配置模块

职责：
- 集中管理业务配置数据
- 按业务模块内聚组织配置
- 支持配置的读取和验证
"""

from .event_configs import EVENT_WINDOW_CONFIGS, HISTORICAL_EVENT_PARAMS
from .data_source_configs import REGIONAL_DATA_SOURCE_PRIORITY

__all__ = [
    'EVENT_WINDOW_CONFIGS',
    'HISTORICAL_EVENT_PARAMS',
    'REGIONAL_DATA_SOURCE_PRIORITY',
]
