"""
缓存键生成器

职责：
- 版本化哈希策略
- 智能键生成（时间对齐、参数版本化）
- 简单键生成

从 infrastructure/cache_service.py 迁移而来
"""
import hashlib
import json
from typing import Dict, Any
from datetime import datetime


class CacheKeyGenerator:
    """缓存Key生成器 - 版本化哈希策略"""
    
    @staticmethod
    def generate_key(
        components: Dict,
        data_version: str = "v1.0"
    ) -> str:
        """
        生成智能缓存Key
        
        Args:
            components: 组件字典，包含:
                - market: 市场代码
                - symbols: 符号列表
                - model_type: 模型类型
                - time_window: 时间窗口
                - params: 参数字典
            data_version: 数据版本（默认 v1.0）
        
        Returns:
            格式化的缓存键: "{version}:{stable_hash}:{time}:{param_hash}"
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
        
        Args:
            prefix: 键前缀
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            格式化的缓存键: "{prefix}:{arg1}:{arg2}:...:{key1=val1}:..."
        """
        parts = [prefix] + [str(arg) for arg in args]
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
        return ":".join(parts)
