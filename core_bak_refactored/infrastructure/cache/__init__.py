"""
缓存基础设施模块

职责：
- 提供多级缓存机制（Memory, LRU, Redis）
- 管理缓存的读写和过期策略
- 支持数据序列化和压缩

从 core/data/cache 迁移而来
"""

from .memory import MemoryTTLCache
from .redis_adapter import RedisCacheAdapter
from .redis_setup import setup_redis_cache, test_redis_connection, close_redis_connection
from .store import get_cached_data, cache_data

__all__ = [
    # 内存缓存
    'MemoryTTLCache',
    
    # Redis缓存
    'RedisCacheAdapter',
    'setup_redis_cache',
    'test_redis_connection',
    'close_redis_connection',
    
    # 缓存存储操作
    'get_cached_data',
    'cache_data',
]
