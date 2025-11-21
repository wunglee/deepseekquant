"""增强版缓存服务 - Infrastructure层技术实现

职责：
1. 多级缓存架构 (L1内存 + L2/L3预留)
2. TTL支持和自动过期
3. 智能Key生成和命中率监控
4. 装饰器模式集成
5. 智能失效策略 (阶段C新增)

架构定位：技术基础设施层
"""

from typing import Any, Dict, Callable, Optional, List
from datetime import datetime
import hashlib
import json
import logging
from cachetools import TTLCache
from dataclasses import dataclass, asdict
from functools import wraps

try:
    from core.base_processor import BaseProcessor
except ImportError:
    class BaseProcessor:
        def __init__(self, *args, **kwargs):
            pass

from .interfaces import ICacheService

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
    
    # L3持久化缓存（暂未启用，预留接口）
    l3_enabled: bool = False
    l3_ttl_seconds: int = 86400  # 24小时
    
    # 缓存Key版本
    cache_version: str = "v1.0"
    
    # 智能TTL策略
    enable_adaptive_ttl: bool = True  # 启用自适应TTL
    min_ttl: int = 60  # 最小TTL 1分钟
    max_ttl: int = 3600  # 最大TTL 1小时


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
    """缓存Key生成器 - 版本化哈希策略"""
    
    @staticmethod
    def generate_key(
        components: Dict,
        data_version: str = "v1.0"
    ) -> str:
        """生成智能缓存Key"""
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
        """生成简单缓存Key"""
        parts = [prefix] + [str(arg) for arg in args]
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
        return ":".join(parts)


class CacheService(BaseProcessor, ICacheService):
    """增强版缓存服务 - 实现ICacheService接口"""
    
    def __init__(self, config: Optional[CacheConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config or CacheConfig()
        self.metrics = CacheMetrics()
        
        # L1内存缓存（TTL支持）
        self._l1_cache = TTLCache(
            maxsize=self.config.l1_maxsize,
            ttl=self.config.l1_ttl_seconds
        )
        
        # L2/L3预留
        self._l2_cache = None
        self._l3_cache = None
        
        logger.info(
            f"缓存服务初始化: L1(maxsize={self.config.l1_maxsize}, "
            f"ttl={self.config.l1_ttl_seconds}s)"
        )
    
    def _initialize_core(self) -> bool:
        """初始化核心"""
        return True
    
    def _process_core(self, *args, **kwargs) -> Any:
        """兼容旧接口"""
        op = kwargs.get('op', 'get')
        key = str(kwargs.get('key', ''))
        
        if op == 'get':
            value = self.get(key)
            return {'status': 'success', 'value': value}
        elif op == 'set':
            ttl = kwargs.get('ttl')
            self.set(key, kwargs.get('value'), ttl)
            return {'status': 'success'}
        elif op == 'invalidate':
            self.invalidate(key)
            return {'status': 'success'}
        
        return {'status': 'error', 'message': 'unsupported op'}
    
    # ICacheService接口实现
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        try:
            value = self._l1_cache.get(key)
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
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """设置缓存值"""
        try:
            # 注意：TTLCache的ttl是全局的，不支持单独设置
            # 如需单独TTL，需要在value中包含过期时间并手动检查
            self._l1_cache[key] = value
            logger.debug(f"L1缓存设置: {key}")
        except Exception as e:
            logger.error(f"L1缓存设置错误: {e}")
    
    def invalidate(self, key: str) -> None:
        """失效指定缓存"""
        try:
            if key in self._l1_cache:
                del self._l1_cache[key]
                logger.debug(f"L1缓存失效: {key}")
        except Exception as e:
            logger.error(f"L1缓存失效错误: {e}")
    
    def preload_pattern(self, pattern: str, loader: Callable, ttl: int = 300) -> None:
        """预加载缓存模式"""
        try:
            data = loader()
            if isinstance(data, dict):
                for key, value in data.items():
                    self.set(key, value, ttl)
                logger.info(f"预加载缓存模式'{pattern}': {len(data)}条")
        except Exception as e:
            logger.error(f"缓存预加载错误: {e}")
            if hasattr(self, 'error_handler'):
                self.error_handler.record_error(
                    e, 'cache_preload', 
                    extra_context={'pattern': pattern}
                )
    
    # 扩展功能
    
    def invalidate_pattern(self, pattern: str) -> int:
        """按模式失效缓存"""
        invalidated_count = 0
        keys_to_invalidate = [
            k for k in self._l1_cache.keys()
            if str(k).startswith(pattern)
        ]
        
        for key in keys_to_invalidate:
            self.invalidate(key)
            invalidated_count += 1
        
        logger.info(f"按模式'{pattern}'失效了{invalidated_count}个缓存")
        return invalidated_count
    
    def get_metrics(self) -> Dict:
        """获取缓存指标"""
        return {
            'cache_size': len(self._l1_cache),
            'max_size': self._l1_cache.maxsize,
            'ttl_seconds': self._l1_cache.ttl,
            **self.metrics.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
    
    def clear(self) -> None:
        """清空所有缓存"""
        try:
            self._l1_cache.clear()
            logger.info("L1缓存已清空")
        except Exception as e:
            logger.error(f"L1缓存清空错误: {e}")
    
    def _cleanup_core(self):
        """清理核心"""
        self.clear()


# 全局缓存服务单例
_global_cache_service: Optional[CacheService] = None


def get_cache_service(config: Optional[CacheConfig] = None) -> CacheService:
    """获取全局缓存服务实例（单例模式）"""
    global _global_cache_service
    
    if _global_cache_service is None:
        _global_cache_service = CacheService(config)
        logger.info("创建全局缓存服务实例")
    
    return _global_cache_service


# ============= 智能失效策略 (阶段C新增) =============

class InvalidationRule:
    """缓存失效规则"""
    
    def __init__(self, name: str, condition: Callable[[str, Any, Dict], bool]):
        """
        Args:
            name: 规则名称
            condition: 判断条件 (key, value, context) -> bool
        """
        self.name = name
        self.condition = condition
    
    def should_invalidate(self, key: str, value: Any, context: Dict) -> bool:
        """判断是否应失效"""
        try:
            return self.condition(key, value, context)
        except Exception as e:
            logger.error(f"失效规则'{self.name}'执行错误: {e}")
            return False


class SmartInvalidationManager:
    """智能失效管理器"""
    
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.rules: List[InvalidationRule] = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认失效规则"""
        # 规列1: 时间窗口变化
        self.add_rule(
            InvalidationRule(
                'time_window_change',
                lambda k, v, ctx: 'time_window' in ctx and ctx['time_window'] != self._extract_time_from_key(k)
            )
        )
        
        # 规列2: 参数版本更新
        self.add_rule(
            InvalidationRule(
                'param_version_change',
                lambda k, v, ctx: 'param_version' in ctx and ctx['param_version'] not in k
            )
        )
        
        # 规列3: 市场数据更新
        self.add_rule(
            InvalidationRule(
                'market_data_update',
                lambda k, v, ctx: ctx.get('market_data_updated', False) and 'market' in k
            )
        )
    
    def add_rule(self, rule: InvalidationRule):
        """添加失效规则"""
        self.rules.append(rule)
        logger.info(f"添加失效规则: {rule.name}")
    
    def check_and_invalidate(self, context: Dict) -> int:
        """
        检查并失效符合规则的缓存
        
        Args:
            context: 上下文信息 (如时间窗口、参数版本等)
        
        Returns:
            失效的缓存数量
        """
        invalidated_count = 0
        keys_to_check = list(self.cache_service._l1_cache.keys())
        
        for key in keys_to_check:
            try:
                value = self.cache_service._l1_cache.get(key)
                
                # 逐个检查规则
                for rule in self.rules:
                    if rule.should_invalidate(key, value, context):
                        self.cache_service.invalidate(key)
                        invalidated_count += 1
                        logger.debug(f"由于规则'{rule.name}'失效: {key}")
                        break
            except Exception as e:
                logger.error(f"检查key'{key}'时出错: {e}")
        
        if invalidated_count > 0:
            logger.info(f"智能失效: {invalidated_count}个缓存")
        
        return invalidated_count
    
    def invalidate_by_condition(self, condition: Callable[[str], bool]) -> int:
        """
        根据自定义条件失效缓存
        
        Args:
            condition: 判断条件 (key) -> bool
        
        Returns:
            失效的缓存数量
        """
        invalidated_count = 0
        keys_to_check = list(self.cache_service._l1_cache.keys())
        
        for key in keys_to_check:
            try:
                if condition(key):
                    self.cache_service.invalidate(key)
                    invalidated_count += 1
            except Exception as e:
                logger.error(f"检查key'{key}'时出错: {e}")
        
        return invalidated_count
    
    def schedule_preload(self, keys: List[str], loader: Callable[[str], Any]):
        """
        计划预加载缓存
        
        Args:
            keys: 要预加载的key列表
            loader: 加载器函数 (key) -> value
        """
        success_count = 0
        for key in keys:
            try:
                value = loader(key)
                self.cache_service.set(key, value)
                success_count += 1
            except Exception as e:
                logger.error(f"预加载key'{key}'失败: {e}")
        
        logger.info(f"缓存预加载完成: {success_count}/{len(keys)}")
        return success_count
    
    @staticmethod
    def _extract_time_from_key(key: str) -> Optional[str]:
        """从key中提取时间信息"""
        parts = key.split(':')
        for part in parts:
            if part.isdigit() and len(part) >= 10:  # 时间戳
                return part
        return None


def get_smart_invalidation_manager(
    cache_service: Optional[CacheService] = None
) -> SmartInvalidationManager:
    """获取智能失效管理器"""
    if cache_service is None:
        cache_service = get_cache_service()
    return SmartInvalidationManager(cache_service)
