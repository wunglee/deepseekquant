"""
智能缓存失效管理器

职责：
- 基于规则的缓存失效策略
- 条件性失效和批量失效
- 缓存预加载调度

从 infrastructure/cache_service.py 迁移而来
"""
from typing import Any, Dict, Callable, List, Optional
import logging

logger = logging.getLogger(__name__)


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
    
    def __init__(self, cache_manager):
        """
        Args:
            cache_manager: CacheManager 实例
        """
        self.cache_manager = cache_manager
        self.rules: List[InvalidationRule] = []
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认失效规则"""
        # 规则1: 时间窗口变化
        self.add_rule(
            InvalidationRule(
                'time_window_change',
                lambda k, v, ctx: 'time_window' in ctx and ctx['time_window'] != self._extract_time_from_key(k)
            )
        )
        
        # 规则2: 参数版本更新
        self.add_rule(
            InvalidationRule(
                'param_version_change',
                lambda k, v, ctx: 'param_version' in ctx and ctx['param_version'] not in k
            )
        )
        
        # 规则3: 市场数据更新
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
        
        # 从 memory_cache 获取所有键（同步访问）
        keys_to_check = list(self.cache_manager.memory_cache.keys())
        
        for key in keys_to_check:
            try:
                value = self.cache_manager.memory_cache.get(key)
                
                # 逐个检查规则
                for rule in self.rules:
                    if rule.should_invalidate(key, value, context):
                        # 调用 clear 方法清除指定键
                        self._invalidate_single_key(key)
                        invalidated_count += 1
                        logger.debug(f"由于规则'{rule.name}'失效: {key}")
                        break
            except Exception as e:
                logger.error(f"检查key'{key}'时出错: {e}")
        
        if invalidated_count > 0:
            logger.info(f"智能失效: {invalidated_count}个缓存")
        
        return invalidated_count
    
    def _invalidate_single_key(self, key: str):
        """失效单个缓存键（从所有缓存层删除）"""
        try:
            # L1: Memory
            if key in self.cache_manager.memory_cache:
                del self.cache_manager.memory_cache[key]
            
            # L2: LRU
            if key in self.cache_manager.lru_cache:
                del self.cache_manager.lru_cache[key]
            
            # L3: Redis
            if self.cache_manager.redis_client:
                try:
                    self.cache_manager.redis_client.delete(f"deepseekquant:{key}")
                except Exception as e:
                    logger.debug(f"Redis删除失败: {e}")
        except Exception as e:
            logger.error(f"失效key'{key}'时出错: {e}")
    
    def invalidate_by_condition(self, condition: Callable[[str], bool]) -> int:
        """
        根据自定义条件失效缓存
        
        Args:
            condition: 判断条件 (key) -> bool
        
        Returns:
            失效的缓存数量
        """
        invalidated_count = 0
        keys_to_check = list(self.cache_manager.memory_cache.keys())
        
        for key in keys_to_check:
            try:
                if condition(key):
                    self._invalidate_single_key(key)
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
                # 使用同步方式设置（直接写入memory_cache）
                self.cache_manager.memory_cache[key] = value
                self.cache_manager.lru_cache[key] = value
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


def get_smart_invalidation_manager(cache_manager=None):
    """
    获取智能失效管理器
    
    Args:
        cache_manager: CacheManager 实例，如果为 None 则需要外部传入
    
    Returns:
        SmartInvalidationManager 实例
    """
    if cache_manager is None:
        raise ValueError("cache_manager 不能为 None，请传入 CacheManager 实例")
    
    return SmartInvalidationManager(cache_manager)
