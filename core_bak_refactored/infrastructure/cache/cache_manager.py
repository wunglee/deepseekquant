"""
统一缓存管理器 - 整合三层缓存机制

职责：
- 管理三层缓存（Memory + LRU + Redis）
- 提供统一的读写接口
- 支持缓存键生成
- 提供缓存统计和清空功能

从 core/data/cache/cachemanager.py 迁移并优化
"""
from typing import Any, Dict, Optional, List
import pickle
import zlib
import hashlib
import logging
from cachetools import LRUCache


class CacheManager:
    """三层缓存管理器
    
    - L1: 内存缓存（Memory Cache）- 最快，无大小限制
    - L2: LRU缓存（Least Recently Used）- 中速，有大小限制
    - L3: Redis缓存（持久化）- 远程，可选，支持分布式
    
    特性：
    - 缓存回填：低层缓存命中后自动回填到高层缓存
    - 统计跟踪：记录命中率、未命中率、缓存大小
    - 灵活配置：支持禁用缓存、自定义TTL、Redis可选
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        初始化缓存管理器
        
        Args:
            config: 配置字典，包含以下键：
                - cache_enabled: 是否启用缓存（默认 False）
                - cache_ttl: 缓存过期时间（秒，默认 300）
                - lru_maxsize: LRU缓存最大条目数（默认 128）
                - redis: Redis配置字典
                    - enabled: 是否启用Redis（默认 False）
                    - host: Redis主机（默认 'localhost'）
                    - port: Redis端口（默认 6379）
                    - db: Redis数据库编号（默认 0）
                    - password: Redis密码（可选）
                    - socket_timeout: 连接超时（默认 5秒）
        """
        self.cache_enabled = config.get('cache_enabled', False)
        self.cache_duration = config.get('cache_ttl', 300)
        self.logger = logging.getLogger('DeepSeekQuant.CacheManager')
        
        # L1: 内存缓存（简单字典）
        self.memory_cache: Dict[str, Any] = {}
        
        # L2: LRU缓存（有大小限制）
        lru_maxsize = config.get('lru_maxsize', 128)
        self.lru_cache: LRUCache = LRUCache(maxsize=lru_maxsize)
        
        # L3: Redis缓存（可选）
        self.redis_client = None
        redis_conf = config.get('redis', {})
        if redis_conf.get('enabled', False):
            self.redis_client = self._setup_redis(redis_conf)
        
        # 缓存统计
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'size': 0
        }

    def _setup_redis(self, redis_conf: Dict[str, Any]) -> Optional[Any]:
        """
        设置Redis连接
        
        Args:
            redis_conf: Redis配置字典
            
        Returns:
            Redis客户端对象，失败返回None
        """
        try:
            import redis
            
            client = redis.Redis(
                host=redis_conf.get('host', 'localhost'),
                port=redis_conf.get('port', 6379),
                db=redis_conf.get('db', 0),
                password=redis_conf.get('password'),
                decode_responses=False,
                socket_timeout=redis_conf.get('socket_timeout', 5),
                retry_on_timeout=True
            )
            
            # 测试连接
            client.ping()
            self.logger.info(f"Redis缓存已启用: {redis_conf.get('host')}:{redis_conf.get('port')}")
            return client
            
        except Exception as e:
            self.logger.warning(f"Redis连接失败，降级为本地缓存: {e}")
            return None

    def generate_key(self, symbols: List[str], period: str, interval: str, 
                    data_type: str, adjustments: bool) -> str:
        """
        生成缓存键（符号顺序无关）
        
        Args:
            symbols: 股票代码列表
            period: 数据期间 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: 数据间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            data_type: 数据类型 (ohlcv, dividends, splits, all)
            adjustments: 是否调整价格（分红和拆股）
        
        Returns:
            MD5哈希值（32字符十六进制字符串）
        """
        # 排序符号以确保键的一致性（无论顺序）
        symbols_str = '_'.join(sorted(symbols))
        
        # 组合所有参数
        key_data = f"{symbols_str}_{period}_{interval}_{data_type}_{adjustments}"
        
        # 生成MD5哈希
        return hashlib.md5(key_data.encode()).hexdigest()

    def generate_key_simple(self, *args: Any, **kwargs: Any) -> str:
        """
        生成缓存键（简化版，用于通用场景）
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            MD5哈希值
        """
        key_data = '_'.join(str(arg) for arg in args)
        if kwargs:
            key_data += '_' + '_'.join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    async def get(self, cache_key: str) -> Optional[Any]:
        """
        从缓存获取数据（三层查询，自动回填）
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据，未命中返回None
        """
        if not self.cache_enabled:
            return None
        
        try:
            # L1: 内存缓存
            if cache_key in self.memory_cache:
                self.cache_stats['hits'] += 1
                self.logger.debug(f"内存缓存命中: {cache_key}")
                return self.memory_cache[cache_key]
            
            # L2: LRU缓存
            data = self.lru_cache.get(cache_key)
            if data is not None:
                # 回填到内存缓存
                self.memory_cache[cache_key] = data
                self.cache_stats['hits'] += 1
                self.logger.debug(f"LRU缓存命中: {cache_key}")
                return data
            
            # L3: Redis缓存
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(f"deepseekquant:{cache_key}")
                    if cached_data:
                        # 解压缩和反序列化
                        data = pickle.loads(zlib.decompress(cached_data))
                        # 回填到内存与LRU缓存
                        self.memory_cache[cache_key] = data
                        self.lru_cache[cache_key] = data
                        self.cache_stats['hits'] += 1
                        self.logger.debug(f"Redis缓存命中: {cache_key}")
                        return data
                except Exception as e:
                    self.logger.debug(f"Redis读取失败: {e}")
            
            # 所有缓存层都未命中
            self.cache_stats['misses'] += 1
            self.logger.debug(f"缓存未命中: {cache_key}")
            return None
            
        except Exception as e:
            self.logger.error(f"缓存读取失败: {e}")
            return None

    async def set(self, cache_key: str, data: Any) -> None:
        """
        写入缓存数据（三层同步）
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        if not self.cache_enabled:
            return
        
        try:
            # L1 & L2: 内存缓存和LRU缓存
            self.memory_cache[cache_key] = data
            self.lru_cache[cache_key] = data
            
            # L3: Redis缓存
            if self.redis_client:
                try:
                    # 序列化并压缩
                    serialized_data = pickle.dumps(data)
                    compressed_data = zlib.compress(serialized_data)
                    
                    # 写入Redis，带过期时间
                    self.redis_client.setex(
                        f"deepseekquant:{cache_key}",
                        self.cache_duration,
                        compressed_data
                    )
                    
                    # 更新缓存大小统计
                    self.cache_stats['size'] += len(compressed_data)
                    self.logger.debug(f"Redis缓存写入成功: {cache_key}, 大小: {len(compressed_data)} bytes")
                    
                except Exception as e:
                    self.logger.debug(f"Redis写入失败: {e}")
                    
        except Exception as e:
            self.logger.error(f"缓存写入失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含缓存统计的字典
        """
        return self.cache_stats.copy()

    def clear(self) -> None:
        """
        清空所有缓存层
        """
        self.memory_cache.clear()
        self.lru_cache.clear()
        
        if self.redis_client:
            try:
                self.redis_client.flushdb()
                self.logger.info("Redis缓存已清空")
            except Exception as e:
                self.logger.error(f"Redis清空失败: {e}")
        
        # 重置统计
        self.cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        self.logger.info("所有缓存已清空")

    def close(self) -> None:
        """
        关闭缓存管理器（释放Redis连接）
        """
        if self.redis_client:
            try:
                self.redis_client.close()
                self.logger.info("Redis连接已关闭")
            except Exception as e:
                self.logger.error(f"关闭Redis连接失败: {e}")
