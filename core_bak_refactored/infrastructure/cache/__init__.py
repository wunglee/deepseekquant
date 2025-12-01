"""
缓存基础设施模块

职责：
- 提供多级缓存机制（Memory, LRU, Redis）
- 管理缓存的读写和过期策略
- 支持数据序列化和压缩

从 core/data/cache 迁移而来
"""

from .cache_manager import CacheManager
from .memory import MemoryTTLCache
from .redis_adapter import RedisCacheAdapter
from .redis_setup import setup_redis_cache, check_redis_connection, close_redis_connection
from .store import get_cached_data, cache_data
from .invalidation import SmartInvalidationManager, InvalidationRule, get_smart_invalidation_manager
from .key_generator import CacheKeyGenerator

__all__ = [
    # 统一缓存管理器（推荐使用）
    'CacheManager',
    
    # 内存缓存（简单实现，用于测试）
    'MemoryTTLCache',
    
    # Redis缓存
    'RedisCacheAdapter',
    'setup_redis_cache',
    'check_redis_connection',
    'close_redis_connection',
    
    # 缓存存储操作（向后兼容）
    'get_cached_data',
    'cache_data',
    
    # 智能失效管理（从 cache_service 迁移）
    'SmartInvalidationManager',
    'InvalidationRule',
    'get_smart_invalidation_manager',
    
    # 缓存键生成器（从 cache_service 迁移）
    'CacheKeyGenerator',
]
