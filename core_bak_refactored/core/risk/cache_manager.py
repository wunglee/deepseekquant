"""
多级缓存管理器

职责：
1. L1内存缓存 (请求级别, TTL=5分钟)
2. L2 Redis缓存 (应用级别, TTL=2小时)  
3. L3持久化缓存 (磁盘级别, TTL=24小时)
4. 智能缓存失效策略
5. 缓存命中率监控

设计原则：
- 三级缓存架构：L1内存 + L2 Redis + L3持久化
- 分层粒度：L1细粒度、L2中粒度、L3粗粒度
- 性能目标：命中率>70%，响应时间减少60%
"""

import hashlib
import json
from typing import Any, Dict, Optional, Callable, Tuple
from datetime import datetime, timedelta
from functools import wraps
import logging
from dataclasses import dataclass, asdict
from cachetools import TTLCache, LRUCache
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    # L1内存缓存
    l1_maxsize: int = 1000
    l1_ttl_seconds: int = 300  # 5分钟
    
    # L2 Redis缓存（暂未启用，预留接口）
    l2_enabled: bool = False
    l2_ttl_seconds: int = 7200  # 2小时
    l2_redis_host: Optional[str] = None
    l2_redis_port: int = 6379
    
    # L3持久化缓存（暂未启用，预留接口）
    l3_enabled: bool = False
    l3_ttl_seconds: int = 86400  # 24小时
    l3_storage_path: Optional[str] = None
    
    # 缓存Key版本
    cache_version: str = "v1.0"


@dataclass
class CacheMetrics:
    """缓存性能指标"""
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0
    l3_hits: int = 0
    l3_misses: int = 0
    
    @property
    def l1_hit_rate(self) -> float:
        """L1命中率"""
        total = self.l1_hits + self.l1_misses
        return self.l1_hits / total if total > 0 else 0.0
    
    @property
    def overall_hit_rate(self) -> float:
        """总体命中率"""
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        total_requests = total_hits + self.l1_misses
        return total_hits / total_requests if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            **asdict(self),
            'l1_hit_rate': self.l1_hit_rate,
            'overall_hit_rate': self.overall_hit_rate
        }


class CacheKeyGenerator:
    """
    缓存Key生成器
    
    策略：版本化哈希Key，平衡精确性和命中率
    """
    
    @staticmethod
    def generate_key(
        components: Dict,
        data_version: str = "v1.0"
    ) -> str:
        """
        生成智能缓存Key
        
        Parameters:
        components: 组件字典，包含market, symbols, model_type等
        data_version: 数据版本
        
        Returns:
        cache_key: 生成的缓存Key
        """
        # 1. 对稳定部分进行哈希
        stable_parts = {
            'market': components.get('market', 'UNKNOWN'),
            'symbols': tuple(sorted(components.get('symbols', []))),
            'model_type': components.get('model_type', 'default')
        }
        stable_hash = hashlib.md5(
            str(stable_parts).encode()
        ).hexdigest()[:12]
        
        # 2. 时间窗口对齐到最近整点（提高命中率）
        time_window = components.get('time_window')
        if time_window:
            if isinstance(time_window, datetime):
                aligned_hour = time_window.replace(
                    minute=0, second=0, microsecond=0
                )
                time_part = f"{int(aligned_hour.timestamp())}"
            else:
                time_part = str(time_window)
        else:
            time_part = "static"
        
        # 3. 参数版本化
        params = components.get('params', {})
        param_version = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"{data_version}:{stable_hash}:{time_part}:{param_version}"
    
    @staticmethod
    def generate_simple_key(prefix: str, *args, **kwargs) -> str:
        """
        生成简单缓存Key
        
        Parameters:
        prefix: Key前缀
        args: 位置参数
        kwargs: 关键字参数
        
        Returns:
        cache_key: 生成的缓存Key
        """
        parts = [prefix] + [str(arg) for arg in args]
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
        return ":".join(parts)


class L1MemoryCache:
    """
    L1内存缓存层
    
    特点：
    - 请求级别缓存
    - TTL=5分钟
    - 细粒度：收益率数据、单资产风险指标
    """
    
    def __init__(self, config: CacheConfig):
        """
        初始化L1缓存
        
        Parameters:
        config: 缓存配置
        """
        self.config = config
        self.cache = TTLCache(
            maxsize=config.l1_maxsize,
            ttl=config.l1_ttl_seconds
        )
        self.metrics = CacheMetrics()
        logger.info(
            f"L1内存缓存初始化: maxsize={config.l1_maxsize}, "
            f"ttl={config.l1_ttl_seconds}s"
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Parameters:
        key: 缓存Key
        
        Returns:
        cached_value: 缓存的值，未命中返回None
        """
        try:
            value = self.cache.get(key)
            if value is not None:
                self.metrics.l1_hits += 1
                logger.debug(f"L1缓存命中: {key}")
                return value
            else:
                self.metrics.l1_misses += 1
                logger.debug(f"L1缓存未命中: {key}")
                return None
        except Exception as e:
            logger.error(f"L1缓存获取错误: {e}")
            self.metrics.l1_misses += 1
            return None
    
    def set(self, key: str, value: Any) -> bool:
        """
        设置缓存值
        
        Parameters:
        key: 缓存Key
        value: 要缓存的值
        
        Returns:
        success: 是否成功
        """
        try:
            self.cache[key] = value
            logger.debug(f"L1缓存设置: {key}")
            return True
        except Exception as e:
            logger.error(f"L1缓存设置错误: {e}")
            return False
    
    def invalidate(self, key: str) -> bool:
        """
        失效指定缓存
        
        Parameters:
        key: 缓存Key
        
        Returns:
        success: 是否成功
        """
        try:
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"L1缓存失效: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"L1缓存失效错误: {e}")
            return False
    
    def clear(self) -> bool:
        """清空所有缓存"""
        try:
            self.cache.clear()
            logger.info("L1缓存已清空")
            return True
        except Exception as e:
            logger.error(f"L1缓存清空错误: {e}")
            return False
    
    def get_metrics(self) -> Dict:
        """获取缓存指标"""
        return {
            'cache_size': len(self.cache),
            'max_size': self.cache.maxsize,
            'ttl_seconds': self.cache.ttl,
            **self.metrics.to_dict()
        }


class RiskCacheManager:
    """
    风险计算缓存管理器
    
    统一管理多级缓存，提供装饰器接口
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化缓存管理器
        
        Parameters:
        config: 缓存配置，默认使用标准配置
        """
        self.config = config or CacheConfig()
        
        # 初始化L1内存缓存
        self.l1_cache = L1MemoryCache(self.config)
        
        # L2/L3预留（未实施）
        self.l2_cache = None  # Redis缓存（待实施）
        self.l3_cache = None  # 持久化缓存（待实施）
        
        logger.info("风险缓存管理器初始化完成")
    
    def cached(
        self,
        key_prefix: str,
        ttl: Optional[int] = None,
        key_generator: Optional[Callable] = None
    ):
        """
        缓存装饰器
        
        Parameters:
        key_prefix: 缓存Key前缀
        ttl: 生存时间（秒），None使用默认值
        key_generator: 自定义Key生成函数
        
        Usage:
        @cache_manager.cached("returns_data")
        def get_returns(market, symbols):
            # 耗时操作
            return calculate_returns(...)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存Key
                if key_generator:
                    cache_key = key_generator(*args, **kwargs)
                else:
                    cache_key = CacheKeyGenerator.generate_simple_key(
                        key_prefix, *args, **kwargs
                    )
                
                # 尝试从L1缓存获取
                cached_value = self.l1_cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # L2/L3查找（待实施）
                # ...
                
                # 缓存未命中，执行原函数
                result = func(*args, **kwargs)
                
                # 存入L1缓存
                self.l1_cache.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def cache_covariance_matrix(
        self,
        market: str,
        symbols: list,
        lookback: int
    ):
        """
        协方差矩阵专用缓存装饰器
        
        Parameters:
        market: 市场类型
        symbols: 资产列表
        lookback: 回望窗口
        
        Usage:
        @cache_manager.cache_covariance_matrix("US", ["AAPL", "GOOGL"], 252)
        def calculate_cov(...):
            return cov_matrix
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成智能缓存Key
                cache_key = CacheKeyGenerator.generate_key({
                    'market': market,
                    'symbols': symbols,
                    'model_type': 'covariance',
                    'params': {'lookback': lookback}
                })
                
                # 查询缓存
                cached_value = self.l1_cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # 执行计算
                result = func(*args, **kwargs)
                
                # 存入缓存
                self.l1_cache.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def invalidate_pattern(self, pattern: str):
        """
        按模式失效缓存
        
        Parameters:
        pattern: 匹配模式（简单前缀匹配）
        
        Example:
        invalidate_pattern("returns_US")  # 失效所有US市场收益率缓存
        """
        invalidated_count = 0
        keys_to_invalidate = [
            k for k in self.l1_cache.cache.keys()
            if str(k).startswith(pattern)
        ]
        
        for key in keys_to_invalidate:
            if self.l1_cache.invalidate(key):
                invalidated_count += 1
        
        logger.info(f"按模式'{pattern}'失效了{invalidated_count}个缓存")
        return invalidated_count
    
    def get_overall_metrics(self) -> Dict:
        """获取整体缓存指标"""
        return {
            'l1': self.l1_cache.get_metrics(),
            'l2': {'enabled': False} if self.l2_cache is None else {},
            'l3': {'enabled': False} if self.l3_cache is None else {},
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_all(self):
        """清空所有缓存"""
        self.l1_cache.clear()
        logger.info("所有缓存已清空")


# 全局缓存管理器实例（单例模式）
_global_cache_manager: Optional[RiskCacheManager] = None


def get_cache_manager(config: Optional[CacheConfig] = None) -> RiskCacheManager:
    """
    获取全局缓存管理器实例
    
    Parameters:
    config: 缓存配置，仅首次调用时生效
    
    Returns:
    cache_manager: 缓存管理器实例
    """
    global _global_cache_manager
    
    if _global_cache_manager is None:
        _global_cache_manager = RiskCacheManager(config)
        logger.info("创建全局缓存管理器实例")
    
    return _global_cache_manager
