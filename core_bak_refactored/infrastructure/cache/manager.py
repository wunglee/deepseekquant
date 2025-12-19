"""
三层缓存管理器 - 统一管理 Memory/Redis（二选一）→ DB → 外部数据源

核心策略：
1. 按时间窗口粒度缓存（月/周，最小粒度为周）
2. 三层缓存顺序：Memory/Redis（二选一）→ DB → 外部API
3. 逐窗口查询，精细化缓存利用
4. 回写机制：所有新数据自动写入各层缓存

职责：
- 管理三层缓存的协调工作
- 提供统一的数据获取接口
- 处理缺失窗口的逐层查询
- Memory 和 Redis 互斥，通过 cache_mode 配置选择
"""

import logging
from typing import Dict, List, Callable
import pandas as pd

from .window_manager import WindowManager
from .memory import MemoryCache
from .redis import RedisCache
from .db import DBCache

logger = logging.getLogger('DeepSeekQuant.CacheManager')


class ThreeLayerCacheManager:
    """
    三层窗口化缓存管理器
    
    缓存层级：
    1. Memory/Redis (二选一，由 cache_mode 决定)
    2. Database (持久化)
    3. External API (最后调用)
    
    窗口粒度：最小为周（weekly），避免过度碎片化
    """
    
    def __init__(
        self,
        db_service=None,
        redis_client=None,
        cache_mode: str = 'memory',
        window_size: int = 1,  # 修改：缓存窗口大小（period的整数倍，默认1）
        memory_max_windows: int = 1000,
        memory_ttl: int = 300,
        redis_ttl: int = 3600
    ):
        """
        初始化三层缓存管理器
        
        Args:
            db_service: 数据库服务实例
            redis_client: Redis 客户端（可选）
            cache_mode: 缓存模式 ('memory' 或 'redis')，二选一
            window_size: 缓存窗口大小（period的整数倍，默认7）
                        例如：period=daily, window_size=7 → 7天一个窗口
                              period=weekly, window_size=4 → 4周一个窗口
            memory_max_windows: 内存最大窗口数
            memory_ttl: 内存缓存TTL（秒）
            redis_ttl: Redis缓存TTL（秒）
        """
        # 窗口管理器
        self._window_mgr = WindowManager()
        
        # 缓存窗口大小（静态配置，period的整数倍）
        if not isinstance(window_size, int) or window_size < 1:
            raise ValueError(f"无效的 window_size: {window_size}，必须是正整数")
        self._window_size = window_size
        
        # 缓存模式：互斥，二选一
        self._cache_mode = cache_mode
        
        # 根据模式初始化对应的缓存层
        if cache_mode == 'memory':
            self._fast_cache = MemoryCache(max_windows=memory_max_windows, ttl=memory_ttl)
            logger.info(f"✅ 使用内存缓存: max_windows={memory_max_windows}, ttl={memory_ttl}s")
        elif cache_mode == 'redis':
            self._fast_cache = RedisCache(redis_client=redis_client, ttl=redis_ttl)
            logger.info(f"✅ 使用Redis缓存: ttl={redis_ttl}s")
        else:
            raise ValueError(f"无效的 cache_mode: {cache_mode}，必须是 'memory' 或 'redis'")
        
        # 数据库缓存
        self._db_cache = DBCache(db_service=db_service)
        
        logger.info(f"✅ ThreeLayerCacheManager 初始化完成: cache_mode={cache_mode}, window_size={window_size}")
    
    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = 'daily',
        db_fetch_func: Callable[[str, str], pd.DataFrame] = None,
        api_fetch_func: Callable[[str, str], pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        获取数据（三层缓存核心方法）
        
        流程：
        1. 生成所需的所有窗口键（使用 self._window_size）
        2. 从快速缓存获取已有窗口
        3. 检测当前周：
           - 如果是当前周，强制重新查询（每天更新）
           - 即使查询结束日期 < 本周末（如上市日），也必须刷新
           - 原因：今天之后的查询可能包含更多天，避免缓存过时
        4. 对缺失窗口逐个进行三层查询：
           - 先查 DB（使用 db_fetch_func）
           - DB 未命中调用外部 API（使用 api_fetch_func）
        5. 所有新数据写入各层缓存
        6. 合并所有窗口并返回
        
        Args:
            symbol: 股票/指数代码
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            period: 数据粒度/K线类型 (daily/weekly/monthly，默认 daily)
                    注意：period 必须 ≤ window_size
            db_fetch_func: 数据库查询函数，签名为 func(start_date, end_date) -> DataFrame
            api_fetch_func: API查询函数，签名为 func(start_date, end_date) -> DataFrame
        
        Returns:
            完整的 DataFrame
        """
        logger.debug(f"📋 三层缓存查询: {symbol}, {start_date} ~ {end_date}, window_size={self._window_size}, mode={self._cache_mode}")
        
        # ========== 第1步：生成所需的所有窗口键 ==========
        window_keys = self._window_mgr.generate_window_keys(start_date, end_date, period, self._window_size)
        logger.info(f"📦 需要 {len(window_keys)} 个窗口 (period={period}, window_size={self._window_size})")
        
        # ========== 第2步：从快速缓存获取已有窗口 ==========
        cached_windows = {}  # {window_key: DataFrame}
        missing_windows = []  # [window_key, ...]
        first_window_key = None  # 记录起始窗口（最早数据）
        
        # 计算当前窗口键（用于检测未完成窗口）
        import datetime
        today = datetime.datetime.now().date()
        today_ts = pd.Timestamp(today)
        current_window_key = self._window_mgr.make_window_key(today_ts, period, self._window_size)
        logger.debug(f"📅 当前窗口: {current_window_key}")
        
        for window_key in window_keys:
            cached_value = self._fast_cache.get(symbol, period, window_key)
            
            if cached_value is not None:
                cached_df = cached_value.get('data')
                is_first = cached_value.get('is_first_window', False)
                
                # 记录起始窗口
                if is_first:
                    first_window_key = window_key
                    logger.info(f"🅰️ 检测到起始窗口: {window_key} (最早数据)")
                
                # 检查是否为当前窗口（未完成窗口）
                is_current_window = (window_key == current_window_key)
                
                if not is_current_window:
                    # 已完成的窗口，直接使用缓存
                    cached_windows[window_key] = cached_df
                else:
                    # 当前窗口，需要重新查询
                    logger.info(f"🔄 当前窗口 {window_key} 需要更新（窗口尚未结束）")
                    missing_windows.append(window_key)
            else:
                # 缺失窗口
                missing_windows.append(window_key)
        
        # 检查是否有比起始窗口更早的查询
        if first_window_key is not None:
            # 移除所有早于起始窗口的请求
            original_count = len(missing_windows)
            missing_windows = [w for w in missing_windows if w >= first_window_key]
            removed_count = original_count - len(missing_windows)
            
            if removed_count > 0:
                logger.info(f"🚫 已忽略 {removed_count} 个早于起始窗口 {first_window_key} 的查询")
        
        logger.info(f"✅ {self._cache_mode}命中: {len(cached_windows)}/{len(window_keys)} 个窗口")
        
        # ========== 第3步：处理缺失窗口（逐个三层查询）==========
        if missing_windows:
            logger.info(f"🔍 缺失 {len(missing_windows)} 个窗口，开始三层查询")
            
            for window_key in missing_windows:
                # 计算窗口的日期范围
                window_start, window_end = self._window_mgr.window_key_to_date_range(window_key, period)
                
                logger.debug(f"🔍 查询窗口: {window_key} ({window_start} ~ {window_end})")
                
                # 3.1 尝试从数据库获取
                db_df = None
                if db_fetch_func:
                    db_df = db_fetch_func(window_start, window_end)
                
                if db_df is not None and not db_df.empty:
                    # 数据库命中，回写快速缓存
                    logger.info(f"✅ 数据库命中: {window_key} ({len(db_df)} 条)")
                    self._fast_cache.set(symbol, period, window_key, db_df)
                    cached_windows[window_key] = db_df
                    continue
                
                # 3.2 数据库也未命中，调用外部 API
                logger.info(f"🌐 API查询: {window_key}")
                
                try:
                    # 调用外部API获取该窗口数据
                    api_df = None
                    if api_fetch_func:
                        api_df = api_fetch_func(window_start, window_end)
                    
                    if api_df is not None and not api_df.empty:
                        logger.info(f"✅ API返回: {window_key} ({len(api_df)} 条)")
                        
                        # 判断是否为起始窗口
                        # 核心逻辑：查询条件要求的最早日期 < 数据源返回的最早日期
                        # 说明查询要求更早的数据但数据源无法提供，返回的就是最早数据
                        is_first_window = False
                        if 'date' in api_df.columns:
                            actual_start = pd.to_datetime(api_df['date'].min())
                            query_start = pd.to_datetime(start_date)
                            
                            # 检查是否为当前窗口（需要排除当前窗口）
                            is_current_window = (window_key == current_window_key)
                            
                            # 起始窗口判断条件：
                            # 1. 非当前窗口（当前窗口会每天刷新，不需要标记）
                            # 2. 查询起始 < 数据源返回的最早日期
                            #    → 说明这是数据源的最早数据（如上市日）
                            if not is_current_window and query_start < actual_start:
                                is_first_window = True
                                logger.info(f"🅰️ 检测到起始窗口: {window_key} (查询从 {query_start.date()}，但数据源最早从 {actual_start.date()} 开始)")
                                
                                # 回溯更新：检查是否有其他窗口包含这个起始日期但未标记
                                # 场景：首次查询正好从上市日开始，未检测到起始窗口
                                #       第二次查询包含更早日期，检测到起始，需回溯更新第一次的缓存
                                self._backfill_first_window_flag(symbol, period, actual_start, window_keys)
                        
                        # 写入快速缓存（数据库由 api_fetch_func 自行写入）
                        self._fast_cache.set(symbol, period, window_key, api_df, is_first_window=is_first_window)
                        cached_windows[window_key] = api_df
                    else:
                        logger.warning(f"⚠️ API无数据: {window_key}")
                except Exception as e:
                    logger.error(f"❌ API查询失败: {window_key}, error={e}")
        
        # ========== 第4步：合并所有窗口数据并返回 ==========
        if not cached_windows:
            logger.warning(f"⚠️ 所有窗口都无数据: {symbol} {start_date}~{end_date}")
            return pd.DataFrame()
        
        # 按窗口键排序并合并
        sorted_keys = sorted(cached_windows.keys())
        result_dfs = [cached_windows[key] for key in sorted_keys]
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        # 精确筛选日期范围
        if 'date' in result_df.columns:
            result_df['date'] = pd.to_datetime(result_df['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            result_df = result_df[(result_df['date'] >= start_dt) & (result_df['date'] <= end_dt)]
        
        logger.info(f"✅ 返回数据: {len(result_df)} 条 (来自 {len(cached_windows)} 个窗口)")
        return result_df
    
    def _backfill_first_window_flag(self, symbol: str, period: str, actual_start: pd.Timestamp, current_window_keys: list) -> None:
        """
        回溯更新起始窗口标记
        
        场景：
        - 首次查询正好从上市日开始（如 2025-01-08 ~ 2025-01-12）
        - query_start == actual_start，未被标记为起始窗口
        - 第二次查询包含更早日期（如 2025-01-06 ~ 2025-01-12）
        - 检测到 query_start < actual_start，确认为起始窗口
        - 需要回溯更新之前缓存中包含 actual_start 的窗口
        
        Args:
            symbol: 股票/指数代码
            period: 数据粒度/K线类型
            actual_start: 数据源返回的实际最早日期（上市日）
            current_window_keys: 当前查询涉及的窗口键列表（避免重复更新）
        """
        # 生成包含 actual_start 的窗口键
        first_data_window_key = self._window_mgr.make_window_key(actual_start, period, self._window_size)
        
        # 如果当前查询已经处理过这个窗口，无需回溯
        if first_data_window_key in current_window_keys:
            return
        
        # 尝试更新该窗口的标记
        if hasattr(self._fast_cache, 'update_first_window_flag'):
            updated = self._fast_cache.update_first_window_flag(
                symbol, period, first_data_window_key, is_first_window=True
            )
            
            if updated:
                logger.info(f"✅ 回溯成功: 更新窗口 {first_data_window_key} 为起始窗口")
    
    def clear_all_cache(self) -> None:
        """清空所有层级的缓存"""
        self._fast_cache.clear()
        logger.info(f"✅ 所有缓存已清空 (cache_mode={self._cache_mode})")
    
    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            'cache_mode': self._cache_mode,
            self._cache_mode: self._fast_cache.get_stats()
        }
