"""
缓存管理器（共享模块）

职责：提供标准化的缓存管理接口
用途：统一管理多层缓存操作
"""

from typing import Any, Dict, Optional
import pickle
import zlib
import hashlib
import logging

logger = logging.getLogger('DeepSeekQuant.Core.Share.CacheManager')


class CacheManager:
    """
    缓存管理器
    
    职责：提供标准化的缓存管理接口
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.cache_enabled = config.get('cache_enabled', False)
        self.cache_duration = config.get('cache_ttl', 300)
        self.memory_cache: Dict[str, Any] = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        
        # Redis（可选）
        self.redis_client = None
        redis_conf = config.get('redis', {})
        if redis_conf.get('enabled', False):
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=redis_conf.get('host', 'localhost'),
                    port=redis_conf.get('port', 6379),
                    db=redis_conf.get('db', 0),
                    password=redis_conf.get('password'),
                    decode_responses=False,
                    socket_timeout=redis_conf.get('socket_timeout', 5),
                    retry_on_timeout=True
                )
                self.redis_client.ping()
                logger.info("Redis缓存已启用")
            except Exception as e:
                logger.warning(f"Redis连接失败，降级为本地缓存: {e}")
                self.redis_client = None
    
    def generate_key(self, *args: Any, **kwargs: Any) -> str:
        """
        生成缓存键
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键字符串
        """
        key_data = '_'.join(str(arg) for arg in args)
        if kwargs:
            key_data += '_' + '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, cache_key: str) -> Optional[Any]:
        """
        获取缓存数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存数据或None
        """
        if not self.cache_enabled:
            return None
        
        try:
            # 内存缓存
            if cache_key in self.memory_cache:
                self.cache_stats['hits'] += 1
                return self.memory_cache[cache_key]
            
            # Redis缓存
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(f"deepseekquant:{cache_key}")
                    if cached_data:
                        data = pickle.loads(zlib.decompress(cached_data))
                        self.memory_cache[cache_key] = data
                        self.cache_stats['hits'] += 1
                        return data
                except Exception as e:
                    logger.debug(f"Redis读取失败: {e}")
            
            self.cache_stats['misses'] += 1
            return None
        except Exception as e:
            logger.error(f"缓存读取失败: {e}")
            return None
    
    async def set(self, cache_key: str, data: Any) -> None:
        """
        设置缓存数据
        
        Args:
            cache_key: 缓存键
            data: 缓存数据
        """
        if not self.cache_enabled:
            return
        
        try:
            # 内存缓存
            self.memory_cache[cache_key] = data
            
            # Redis缓存
            if self.redis_client:
                try:
                    serialized_data = pickle.dumps(data)
                    compressed_data = zlib.compress(serialized_data)
                    self.redis_client.setex(
                        f"deepseekquant:{cache_key}",
                        self.cache_duration,
                        compressed_data
                    )
                    self.cache_stats['size'] += len(compressed_data)
                except Exception as e:
                    logger.debug(f"Redis写入失败: {e}")
        except Exception as e:
            logger.error(f"缓存写入失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计
        
        Returns:
            缓存统计字典
        """
        return self.cache_stats.copy()
    
    def clear(self) -> None:
        """
        清空所有缓存
        """
        self.memory_cache.clear()
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis清空失败: {e}")
        self.cache_stats = {'hits': 0, 'misses': 0, 'size': 0}