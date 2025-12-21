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
from typing import Optional, Dict
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
    
    def get(self, symbol: str, period: str, window_key: str) -> Optional[Dict]:
        """
        获取单个窗口数据（包含元数据）
        
        Args:
            symbol: 股票/指数代码
            period: 周期
            window_key: 窗口键
        
        Returns:
            Dict {
                'data': DataFrame,           # 实际数据
                'is_first_window': bool,     # 是否为起始窗口（最早数据）
                'timestamp': float           # 缓存时间戳
            } 或 None
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
                    cached_dict = pickle.loads(zlib.decompress(cached_data))
                else:
                    cached_dict = pickle.loads(cached_data)
                
                logger.debug(f"✅ Redis命中: {cache_key}")
                return cached_dict  # 返回完整字典（包含data、is_first_window、timestamp）
        except Exception as e:
            logger.warning(f"⚠️ Redis读取失败: {cache_key}, error={e}")
        
        return None
    
    def set(self, symbol: str, period: str, window_key: str, data: pd.DataFrame, is_first_window: bool = False) -> None:
        """
        写入单个窗口数据（包含元数据）
        
        Args:
            symbol: 股票/指数代码
            period: 周期
            window_key: 窗口键
            data: 数据
            is_first_window: 是否为起始窗口（最早数据）
        """
        if data is None or data.empty:
            return
        
        cache_key = f"deepseekquant:window:{symbol}:{period}:{window_key}"
        
        try:
            # 构造缓存对象（与MemoryCache保持一致）
            import time
            cached_dict = {
                'data': data.copy(),
                'is_first_window': is_first_window,
                'timestamp': time.time()
            }
            
            # 序列化
            serialized_data = pickle.dumps(cached_dict)
            
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
    
    def update_first_window_flag(self, symbol: str, period: str, window_key: str, is_first_window: bool) -> bool:
        """
        更新指定窗口的 is_first_window 标记（用于回溯更新）
        
        Args:
            symbol: 股票/指数代码
            period: 数据粒度（daily/weekly/monthly，K线类型）
            window_key: 窗口键
            is_first_window: 新的标记值
        
        Returns:
            bool: 是否成功更新（如果窗口不存在则返回False）
        """
        cache_key = f"deepseekquant:window:{symbol}:{period}:{window_key}"
        
        try:
            # 读取现有数据
            if self._memory_store is not None:
                cached_data = self._memory_store.get(cache_key)
            else:
                cached_data = self._client.get(cache_key)
            
            if cached_data:
                # 反序列化
                if self._enable_compression:
                    cached_dict = pickle.loads(zlib.decompress(cached_data))
                else:
                    cached_dict = pickle.loads(cached_data)
                
                # 更新标记
                old_flag = cached_dict.get('is_first_window', False)
                cached_dict['is_first_window'] = is_first_window
                
                # 重新序列化并写入
                serialized_data = pickle.dumps(cached_dict)
                if self._enable_compression:
                    final_data = zlib.compress(serialized_data)
                else:
                    final_data = serialized_data
                
                if self._memory_store is not None:
                    self._memory_store[cache_key] = final_data
                else:
                    self._client.setex(cache_key, self._ttl, final_data)
                
                if old_flag != is_first_window:
                    logger.info(f"🔄 回溯更新窗口标记: {cache_key} (is_first_window: {old_flag} → {is_first_window})")
                
                return True
        except Exception as e:
            logger.warning(f"⚠️ Redis更新标记失败: {cache_key}, error={e}")
        
        return False
    
    def clear(self) -> None:
        """清空缓存（仅内存模拟模式）"""
        if self._memory_store is not None:
            self._memory_store.clear()
            logger.info("✅ Redis缓存已清空（内存模拟）")
    
    def get_stats(self) -> Dict:
        """获取统计信息（内存模拟模式）"""
        if self._memory_store is not None:
            return {
                'total_windows': len(self._memory_store),
                'mode': 'memory_simulation'
            }
        else:
            return {
                'mode': 'redis_connected',
                'note': 'Redis statistics not implemented'
            }
