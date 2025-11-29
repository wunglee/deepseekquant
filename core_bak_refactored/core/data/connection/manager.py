"""
数据连接管理器（已迁移到 infrastructure/connection）

⚠️ 此模块已废弃，请使用：
    from core_bak_refactored.infrastructure.connection import DataConnectionManager

保留此文件仅为向后兼容，将在未来版本中移除。
"""
import warnings
from core_bak_refactored.infrastructure.connection import DataConnectionManager

warnings.warn(
    "core.data.connection.manager 已迁移到 infrastructure.connection，"
    "请更新导入路径：from core_bak_refactored.infrastructure.connection import DataConnectionManager",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['DataConnectionManager']
