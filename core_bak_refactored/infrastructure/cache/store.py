from typing import Any, Dict, Optional
import pickle
import zlib
import logging

logger = logging.getLogger(__name__)


async def get_cached_data(fetcher: Any, cache_key: str) -> Optional[Dict]:
    """
    从缓存中获取数据（从 DataFetcher._get_cached_data 迁移而来）。
    
    实现三级缓存策略：
    1. Memory Cache (内存TTL缓存) - 最快
    2. LRU Cache (最近最少使用) - 次快
    3. Redis Cache (持久化缓存) - 远程，压缩存储
    
    命中后会回填到更快的缓存层。
    
    Args:
        fetcher: DataFetcher 实例，包含缓存属性 (memory_cache, lru_cache, redis_client)
        cache_key: 缓存键
    
    Returns:
        缓存的数据字典，如果没有命中则返回 None
    
    Raises:
        不抛出异常，所有异常均被捕获并记录日志
    """
    try:
        # 第一级：检查内存缓存 (TTLCache)
        if cache_key in fetcher.memory_cache:
            logger.debug(f"内存缓存命中: {cache_key}")
            return fetcher.memory_cache[cache_key]

        # 第二级：检查LRU缓存
        if cache_key in fetcher.lru_cache:
            data = fetcher.lru_cache[cache_key]
            # 回填到内存缓存
            fetcher.memory_cache[cache_key] = data
            logger.debug(f"LRU缓存命中: {cache_key}")
            return data

        # 第三级：检查Redis缓存
        if getattr(fetcher, 'redis_client', None):
            try:
                cached_data = fetcher.redis_client.get(f"deepseekquant:{cache_key}")
                if cached_data:
                    # 解压缩和反序列化
                    data = pickle.loads(zlib.decompress(cached_data))
                    # 回填到内存与LRU缓存
                    fetcher.memory_cache[cache_key] = data
                    fetcher.lru_cache[cache_key] = data
                    logger.debug(f"Redis缓存命中: {cache_key}")
                    return data
            except Exception as e:
                logger.warning(f"Redis缓存读取失败: {e}")

        # 所有缓存层都未命中
        logger.debug(f"缓存未命中: {cache_key}")
        return None

    except Exception as e:
        logger.error(f"缓存读取失败: {e}")
        return None


async def cache_data(fetcher: Any, cache_key: str, data: Dict) -> None:
    """
    缓存数据到多级缓存（从 DataFetcher._cache_data 迁移而来）。
    
    同时写入三级缓存：
    1. Memory Cache - 写入TTL缓存
    2. LRU Cache - 写入LRU缓存
    3. Redis Cache - 序列化+压缩后写入，带过期时间
    
    Args:
        fetcher: DataFetcher 实例
        cache_key: 缓存键
        data: 要缓存的数据字典
    
    Returns:
        None
    
    Raises:
        不抛出异常，所有异常均被捕获并记录日志
    """
    try:
        # 第一级：写入内存缓存
        fetcher.memory_cache[cache_key] = data
        
        # 第二级：写入LRU缓存
        fetcher.lru_cache[cache_key] = data

        # 第三级：写入Redis缓存
        if getattr(fetcher, 'redis_client', None):
            try:
                # 序列化并压缩数据
                serialized_data = pickle.dumps(data)
                compressed_data = zlib.compress(serialized_data)

                # 设置Redis缓存，使用配置的缓存时间
                fetcher.redis_client.setex(
                    f"deepseekquant:{cache_key}",
                    fetcher.cache_duration,
                    compressed_data
                )

                # 更新缓存统计
                fetcher.cache_stats['size'] += len(compressed_data)
                logger.debug(f"Redis缓存写入成功: {cache_key}, 压缩后大小: {len(compressed_data)} bytes")

            except Exception as e:
                logger.warning(f"Redis缓存写入失败: {e}")
                # 不影响主流程，继续使用内存缓存

        # 更新缓存统计
        fetcher.cache_stats['hits'] += 1
        fetcher.performance_metrics['cache_writes'] = fetcher.performance_metrics.get('cache_writes', 0) + 1
        
        logger.debug(f"数据缓存成功: {cache_key}")

    except Exception as e:
        logger.error(f"数据缓存失败: {e}")
        # 即使缓存失败也不影响主流程
