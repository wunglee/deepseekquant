"""
Redis缓存设置模块（已迁移到 infrastructure/cache）

⚠️ 此模块已废弃，请使用：
    from core_bak_refactored.infrastructure.cache import setup_redis_cache

保留此文件仅为向后兼容，将在未来版本中移除。
"""
import warnings
from core_bak_refactored.infrastructure.cache import (
    setup_redis_cache,
    test_redis_connection,
    close_redis_connection
)

warnings.warn(
    "core.data.cache.redis_setup 已迁移到 infrastructure.cache，"
    "请更新导入路径：from core_bak_refactored.infrastructure.cache import setup_redis_cache",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['setup_redis_cache', 'test_redis_connection', 'close_redis_connection']