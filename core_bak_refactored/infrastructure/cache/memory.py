"""
内存缓存层 - 窗口级别的 LRU + TTL 缓存

职责：
1. 按窗口粒度缓存数据（symbol:period:window_key）
2. LRU 淘汰策略（OrderedDict）
3. TTL 过期机制
4. 窗口级别的读写

注意：
- 缓存键必须包含 period（数据粒度/K线类型）
- period 是数据的本质属性（daily/weekly/monthly）
- period 必须 ≤ window_size
"""

import logging
import time
from typing import Optional, Dict
from collections import OrderedDict
import pandas as pd

logger = logging.getLogger('DeepSeekQuant.MemoryCache')


class MemoryCache:
    """内存缓存层（LRU + TTL）"""
    
    def __init__(self, max_windows: int = 1000, ttl: int = 300):
        """
        初始化内存缓存
        
        Args:
            max_windows: 最大缓存窗口数（默认1000）
            ttl: 缓存过期时间（秒，默认300=5分钟）
        """
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._max_windows = max_windows
        self._ttl = ttl
        logger.info(f"✅ MemoryCache 初始化: max_windows={max_windows}, ttl={ttl}s")
    
    def get(self, symbol: str, period: str, window_key: str) -> Optional[Dict]:
        """
        获取单个窗口数据（包含元数据）
        
        Args:
            symbol: 股票/指数代码
            period: 数据粒度（daily/weekly/monthly，K线类型）
            window_key: 窗口键
        
        Returns:
            Dict {
                'data': DataFrame,           # 实际数据
                'is_first_window': bool,     # 是否为起始窗口（最早数据）
                'timestamp': float           # 缓存时间戳
            } 或 None
        """
        cache_key = f"{symbol}:{period}:{window_key}"
        
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            
            # 检查过期
            if time.time() - cached['timestamp'] < self._ttl:
                # 移到末尾（LRU更新）
                self._cache.move_to_end(cache_key)
                logger.debug(f"✅ 内存命中: {cache_key}")
                return cached
            else:
                # 过期删除
                del self._cache[cache_key]
                logger.debug(f"🗑️ 缓存过期: {cache_key}")
        
        return None
    
    def set(self, symbol: str, period: str, window_key: str, data: pd.DataFrame, is_first_window: bool = False) -> None:
        """
        写入单个窗口数据（包含元数据）
        
        Args:
            symbol: 股票/指数代码
            period: 数据粒度（daily/weekly/monthly，K线类型）
            window_key: 窗口键
            data: 数据
            is_first_window: 是否为起始窗口（最早数据）
        """
        if data is None or data.empty:
            return
        
        cache_key = f"{symbol}:{period}:{window_key}"
        
        # LRU淘汰
        if len(self._cache) >= self._max_windows:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"🗑️ LRU淘汰: {oldest_key}")
        
        self._cache[cache_key] = {
            'data': data.copy(),
            'is_first_window': is_first_window,
            'timestamp': time.time()
        }
        
        first_flag = "🅰️ " if is_first_window else ""
        logger.debug(f"✅ 内存写入: {first_flag}{cache_key} ({len(data)} 条)")
    
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
        cache_key = f"{symbol}:{period}:{window_key}"
        
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            
            # 检查过期
            if time.time() - cached['timestamp'] < self._ttl:
                # 更新标记
                old_flag = cached['is_first_window']
                cached['is_first_window'] = is_first_window
                
                if old_flag != is_first_window:
                    logger.info(f"🔄 回溯更新窗口标记: {cache_key} (is_first_window: {old_flag} → {is_first_window})")
                
                return True
            else:
                # 过期删除
                del self._cache[cache_key]
                logger.debug(f"🗑️ 缓存过期: {cache_key}")
        
        return False
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        logger.info("✅ 内存缓存已清空")
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_windows': len(self._cache),
            'max_windows': self._max_windows,
            'usage_percent': len(self._cache) / self._max_windows * 100
        }
