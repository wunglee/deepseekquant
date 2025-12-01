"""
连接管理基础设施模块

职责：
- 管理数据源连接的创建、关闭和健康检查
- 提供连接池和重连机制
- 支持多种连接类型

从 core/data/connection 迁移而来
"""

from .data_connection_manager import DataConnectionManager

__all__ = [
    'DataConnectionManager',
]
