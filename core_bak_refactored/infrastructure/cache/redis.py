"""
Redis 缓存层 - 窗口级别的 Redis 缓存

职责：
1. 按窗口粒度缓存数据（symbol:period:window_key）
2. 序列化/反序列化（pickle）
3. 压缩/解压（zlib）
4. TTL 过期机制
"""

import logging
import pickle
import zlib
from typing import Optional
import pandas as pd

logger = logging.getLogger('DeepSeekQuant.RedisCache')


class RedisCache:
    """
    Redis 缓存层
    
    注意：
    - 如果 Redis 不可用，会降级为内存模拟
    - 生产环境请使用真实 Redis 客户端
    """
    
    def __init__(self, redis_client=None, ttl: int = 3600, enable_compression: bool = True):
        """
        初始化 Redis 缓存
        
        Args:
            redis_client: Redis 客户端（如为 None 则使用内存模拟）
            ttl: 缓存过期时间（秒，默认3600=1小时）
            enable_compression: 是否启用压缩（默认True）
        """
        self._client = redis_client
        self._ttl = ttl
        self._enable_compression = enable_compression
        
        # 如果没有真实 Redis，使用内存模拟
        if self._client is None:
            self._memory_store = {}
            logger.warning("⚠️ Redis 客户端未配置，使用内存模拟")
        else:
            self._memory_store = None
            logger.info(f"✅ RedisCache 初始化: ttl={ttl}s, compression={enable_compression}")
    
    def get(self, symbol: str, period: str, window_key: str) -> Optional[pd.DataFrame]:
        """
        获取单个窗口数据
        
        Args:
            symbol: 股票/指数代码
            period: 周期
            window_key: 窗口键
        
        Returns:
            DataFrame 或 None
        """
        cache_key = f"deepseekquant:window:{symbol}:{period}:{window_key}"
        
        try:
            # 内存模拟模式
            if self._memory_store is not None:
                cached_data = self._memory_store.get(cache_key)
            else:
                # 真实 Redis
                cached_data = self._client.get(cache_key)
            
            if cached_data:
                # 反序列化
                if self._enable_compression:
                    data = pickle.loads(zlib.decompress(cached_data))
                else:
                    data = pickle.loads(cached_data)
                
                logger.debug(f"✅ Redis命中: {cache_key}")
                return data
        except Exception as e:
            logger.warning(f"⚠️ Redis读取失败: {cache_key}, error={e}")
        
        return None
    
    def set(self, symbol: str, period: str, window_key: str, data: pd.DataFrame) -> None:
        """
        写入单个窗口数据
        
        Args:
            symbol: 股票/指数代码
            period: 周期
            window_key: 窗口键
            data: 数据
        """
        if data is None or data.empty:
            return
        
        cache_key = f"deepseekquant:window:{symbol}:{period}:{window_key}"
        
        try:
            # 序列化
            serialized_data = pickle.dumps(data)
            
            # 压缩
            if self._enable_compression:
                final_data = zlib.compress(serialized_data)
                size = len(final_data)
            else:
                final_data = serialized_data
                size = len(final_data)
            
            # 写入 Redis
            if self._memory_store is not None:
                # 内存模拟
                self._memory_store[cache_key] = final_data
            else:
                # 真实 Redis
                self._client.setex(cache_key, self._ttl, final_data)
            
            logger.debug(f"✅ Redis写入: {cache_key} ({len(data)} 条, {size} bytes)")
        except Exception as e:
            logger.warning(f"⚠️ Redis写入失败: {cache_key}, error={e}")
    
    def clear(self) -> None:
        """清空缓存（仅内存模拟模式）"""
        if self._memory_store is not None:
            self._memory_store.clear()
            logger.info("✅ Redis缓存已清空（内存模拟）")
