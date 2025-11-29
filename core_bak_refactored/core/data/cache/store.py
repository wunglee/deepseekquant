"""
缓存存储操作模块（已迁移到 infrastructure/cache）

⚠️ 此模块已废弃，请使用：
    from core_bak_refactored.infrastructure.cache import get_cached_data, cache_data

保留此文件仅为向后兼容，将在未来版本中移除。
"""
import warnings
from core_bak_refactored.infrastructure.cache import get_cached_data, cache_data

warnings.warn(
    "core.data.cache.store 已迁移到 infrastructure.cache，"
    "请更新导入路径：from core_bak_refactored.infrastructure.cache import get_cached_data, cache_data",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['get_cached_data', 'cache_data']