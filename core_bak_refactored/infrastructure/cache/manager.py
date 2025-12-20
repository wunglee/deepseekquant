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
from typing import Dict, List, Callable, Optional
import pandas as pd
from jinja2.utils import missing

from .window_manager import WindowManager
from .memory import MemoryCache
from .redis import RedisCache
from .db import DBCache
from core_bak_refactored.core.share.market.trading_calendar_service import get_trading_calendar_service
from core_bak_refactored.core.share.market.market_utils import MarketUtils
from core_bak_refactored.core.share.market.market_enums import MarketCode

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
        
        # 交易日历服务（用于判断连续性）
        self._calendar_service = get_trading_calendar_service()
        
        logger.info(f"✅ ThreeLayerCacheManager 初始化完成: cache_mode={cache_mode}, window_size={window_size}")
    
    def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = 'daily',
        market_code: Optional[MarketCode] = None,
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
            market_code: 市场代码枚举 (MarketCode.CN/US/HK/JP/EU/SG)，用于交易日历判断，如为None则从 symbol 推断
            db_fetch_func: 数据库查询函数，签名为 func(start_date, end_date) -> DataFrame
            api_fetch_func: API查询函数，签名为 func(start_date, end_date) -> DataFrame
        
        Returns:
            完整的 DataFrame
        """
        logger.debug(f"📋 三层缓存查询: {symbol}, {start_date} ~ {end_date}, window_size={self._window_size}, mode={self._cache_mode}")
        
        # 推断市场代码（用于交易日历）
        if market_code is None:
            market_code = MarketUtils.infer_market_from_symbol(symbol)
        logger.debug(f"🌏 使用市场代码: {market_code}")
        
        # ========== 第1步：生成所需的所有窗口键 ==========
        window_keys = self._window_mgr.generate_window_keys(start_date, end_date, period, self._window_size, market_code)
        logger.info(f"📦 需要 {len(window_keys)} 个窗口 (period={period}, window_size={self._window_size})")
        
        # ========== 第2步：从快速缓存获取已有窗口 ==========
        cached_windows = {}  # {window_key: DataFrame}
        missing_windows = []  # [window_key, ...]
        first_window_key = None  # 记录起始窗口（最早数据）
        
        # 计算当前窗口键（用于检测未完成窗口）
        import datetime
        today = datetime.datetime.now().date()
        today_ts = pd.Timestamp(today)
        current_window_key = self._window_mgr.make_window_key(today_ts, period, self._window_size, market_code)
        logger.debug(f"📅 当前窗口: {current_window_key}")
        for window_key in window_keys:
            cached_value = self._fast_cache.get(symbol, period, window_key)
            
            if cached_value is not None:
                # 两种缓存现在都返回dict：{'data': df, 'is_first_window': bool, 'timestamp': float}
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
        
        # ========== 第3步：处理缺失窗口（合并连续窗口，批量查询）==========
        if missing_windows:
            logger.info(f"🔍 缺失 {len(missing_windows)} 个窗口，开始三层查询")
            
            # 🔧 关键优化：合并连续未命中窗口，减少网络请求次数
            merged_ranges = self._merge_continuous_windows(missing_windows, period, market_code)
            logger.info(f"🔧 合并后: {len(merged_ranges)} 个连续范围 (原 {len(missing_windows)} 个窗口)")
            
            for range_info in merged_ranges:
                range_start = range_info['start']
                range_end = range_info['end']
                range_windows = range_info['windows']  # 该范围包含的窗口键列表
                
                logger.info(f"📊 批量查询: {range_start} ~ {range_end} (包含 {len(range_windows)} 个窗口)")
                
                # 3.1 尝试从数据库获取大范围数据
                db_df = None
                if db_fetch_func:
                    # 检查函数是否支持period参数
                    import inspect
                    sig = inspect.signature(db_fetch_func)
                    if 'period' in sig.parameters:
                        db_df = db_fetch_func(range_start, range_end, period=period)
                    else:
                        db_df = db_fetch_func(range_start, range_end)
                
                if db_df is not None and not db_df.empty:
                    logger.info(f"✅ 数据库批量命中: {range_start} ~ {range_end} ({len(db_df)} 条)")
                    # 分配数据到各个穗口
                    self._distribute_data_to_windows(symbol, period, db_df, range_windows, cached_windows, current_window_key, start_date, market_code)
                    continue
                
                # 3.2 数据库也未命中，调用外部 API
                logger.info(f"🌐 API批量查询: {range_start} ~ {range_end}")
                
                try:
                    api_df = None
                    if api_fetch_func:
                        # 检查函数是否支持period参数
                        import inspect
                        sig = inspect.signature(api_fetch_func)
                        if 'period' in sig.parameters:
                            api_df = api_fetch_func(range_start, range_end, period=period)
                        else:
                            api_df = api_fetch_func(range_start, range_end)
                    
                    if api_df is not None and not api_df.empty:
                        logger.info(f"✅ API批量返回: {range_start} ~ {range_end} ({len(api_df)} 条)")
                        # 分配数据到各个穗口
                        self._distribute_data_to_windows(symbol, period, api_df, range_windows, cached_windows, current_window_key, start_date, market_code)
                    else:
                        logger.warning(f"⚠️ API无数据: {range_start} ~ {range_end}")
                except Exception as e:
                    logger.error(f"❌ API批量查询失败: {range_start} ~ {range_end}, error={e}")
        
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
    
    def _merge_continuous_windows(self, window_keys: list, period: str, market_code: MarketCode) -> list:
        """
        合并连续的窗口键，减少网络请求次数
        
        Args:
            window_keys: 缺失窗口键列表 (已排序)
            period: 数据粒度
            market_code: 市场代码，用于交易日历判断
        
        Returns:
            合并后的连续范围列表，每个元素包含:
            - start: 范围起始日期
            - end: 范围结束日期
            - windows: 该范围包含的窗口键列表
        
        Example:
            Input: ['2024-01_01', '2024-02_02', '2024-03_03', '2024-05_05', '2024-06_06']
            Output: [
                {'start': '2024-01-01', 'end': '2024-03-31', 'windows': ['2024-01_01', '2024-02_02', '2024-03_03']},
                {'start': '2024-05-01', 'end': '2024-06-30', 'windows': ['2024-05_05', '2024-06_06']}
            ]
        """
        if not window_keys:
            return []
        
        # 按窗口键排序
        sorted_keys = sorted(window_keys)
        
        merged_ranges = []
        current_range_windows = [sorted_keys[0]]
        
        for i in range(1, len(sorted_keys)):
            prev_key = sorted_keys[i-1]
            curr_key = sorted_keys[i]
            
            # 检查是否连续：下一个窗口紧跟上一个窗口
            if self._is_consecutive_windows(prev_key, curr_key, period, market_code):
                # 连续，加入当前范围
                current_range_windows.append(curr_key)
            else:
                # 不连续，保存当前范围，开始新范围
                range_start, _ = self._window_mgr.window_key_to_date_range(current_range_windows[0], period)
                _, range_end = self._window_mgr.window_key_to_date_range(current_range_windows[-1], period)
                merged_ranges.append({
                    'start': range_start,
                    'end': range_end,
                    'windows': current_range_windows.copy()
                })
                current_range_windows = [curr_key]
        
        # 添加最后一个范围
        range_start, _ = self._window_mgr.window_key_to_date_range(current_range_windows[0], period)
        _, range_end = self._window_mgr.window_key_to_date_range(current_range_windows[-1], period)
        merged_ranges.append({
            'start': range_start,
            'end': range_end,
            'windows': current_range_windows.copy()
        })
        
        return merged_ranges
    
    def _is_consecutive_windows(self, key1: str, key2: str, period: str, market_code: MarketCode) -> bool:
        """
        判断两个窗口是否连续（基于不同周期的判断逻辑）
        
        Args:
            key1: 第一个窗口键
            key2: 第二个窗口键
            period: 数据粒度 (daily/weekly/monthly)
            market_code: 市场代码枚举（MarketCode.CN/US/HK/JP/EU/SG）
        
        Returns:
            True 表示连续，False 表示不连续
            
        逻辑：
            - daily: 使用交易日历判断连续性（考虑节假日）
            - weekly: 判断ISO周号是否连续
            - monthly: 判断月份是否连续
        """
        from datetime import datetime
        
        if period == 'daily':
            # Daily周期：使用交易日历判断连续性
            # 获取两个窗口的日期范围
            _, end1 = self._window_mgr.window_key_to_date_range(key1, period)
            start2, _ = self._window_mgr.window_key_to_date_range(key2, period)
            
            # 转换为datetime对象
            end1_dt = datetime.strptime(end1, '%Y-%m-%d')
            start2_dt = datetime.strptime(start2, '%Y-%m-%d')
            
            # 使用交易日历服务判断连续性
            return self._calendar_service.is_consecutive_trading_days(market_code, end1_dt, start2_dt)
        
        elif period == 'weekly':
            # Weekly周期：判断ISO周号是否连续
            # 窗口键格式：YYYY-Www_Www（例如：2025-W01_W01）
            try:
                # 解析key1的结束周
                parts1 = key1.split('_')
                year1_week = parts1[0]  # YYYY-Www
                end_week1 = parts1[1]   # Www
                year1 = int(year1_week.split('-W')[0])
                week1_end = int(end_week1.replace('W', ''))
                
                # 解析key2的起始周
                parts2 = key2.split('_')
                year2_week = parts2[0]  # YYYY-Www
                start_week2_str = year2_week.split('-W')[1]
                year2 = int(year2_week.split('-W')[0])
                week2_start = int(start_week2_str)
                
                # 判断连续性
                if year1 == year2:
                    # 同一年：周号连续（week1_end + 1 == week2_start）
                    return (week1_end + 1) == week2_start
                elif year1 + 1 == year2:
                    # 跨年：year1的最后一周 → year2的第1周
                    # year1的最后一周号通常是52或53
                    return week1_end >= 52 and week2_start == 1
                else:
                    return False
            except (ValueError, IndexError) as e:
                logger.warning(f"⚠️ 解析周窗口键失败: {key1}, {key2}, error: {e}")
                return False
        
        elif period == 'monthly':
            # Monthly周期：判断月份是否连续
            # 窗口键格式：YYYY-MM_MM（例如：2025-01_03）
            try:
                # 解析key1的结束月
                parts1 = key1.split('_')
                year_month1 = parts1[0]  # YYYY-MM
                end_month1_str = parts1[1]  # MM
                year1 = int(year_month1.split('-')[0])
                month1_end = int(end_month1_str)
                
                # 解析key2的起始月
                parts2 = key2.split('_')
                year_month2 = parts2[0]  # YYYY-MM
                month2_start_str = year_month2.split('-')[1]
                year2 = int(year_month2.split('-')[0])
                month2_start = int(month2_start_str)
                
                # 判断连续性
                if year1 == year2:
                    # 同一年：月份连续（month1_end + 1 == month2_start）
                    return (month1_end + 1) == month2_start
                elif year1 + 1 == year2:
                    # 跨年：year1的12月 → year2的1月
                    return month1_end == 12 and month2_start == 1
                else:
                    return False
            except (ValueError, IndexError) as e:
                logger.warning(f"⚠️ 解析月窗口键失败: {key1}, {key2}, error: {e}")
                return False
        
        else:
            logger.warning(f"⚠️ 不支持的周期: {period}")
            return False
    
    def _distribute_data_to_windows(self, symbol: str, period: str, data: pd.DataFrame, 
                                   window_keys: list, cached_windows: dict, 
                                   current_window_key: str, query_start_date: str, market_code: MarketCode) -> None:
        """
        将大范围数据分配到各个窗口，并写入缓存
        
        Args:
            symbol: 股票/指数代码
            period: 数据粒度
            data: 大范围查询返回的数据
            window_keys: 需要分配的窗口键列表
            cached_windows: 已缓存窗口字典 (输出参数)
            current_window_key: 当前窗口键
            query_start_date: 查询起始日期（用于判断起始窗口）
            market_code: 市场代码枚举，用于交易日历判断
        """
        if data.empty:
            return
        
        # 确保数据有 date 列
        if 'date' not in data.columns:
            logger.warning("⚠️ 数据缺少 date 列，无法分配到窗口")
            return
        
        data['date'] = pd.to_datetime(data['date'])
        actual_start = data['date'].min()
        actual_end = data['date'].max()
        query_start = pd.to_datetime(query_start_date)
        
        # 🔧 关键修复：将 query_start 调整到下一个交易日（如果不是交易日）
        query_start_dt = query_start.to_pydatetime()
        if not self._calendar_service.is_trading_day(market_code, query_start_dt):
            next_trading_day = self._calendar_service.get_next_trading_day(market_code, query_start_dt)
            if next_trading_day:
                query_start = pd.Timestamp(next_trading_day)
                logger.debug(f"📅 调整查询起始日期: {query_start_dt.date()} (非交易日) → {next_trading_day.date()} (交易日)")
        
        # 分配数据到各个穗口
        for window_key in window_keys:
            window_start, window_end = self._window_mgr.window_key_to_date_range(window_key, period)
            window_start_dt = pd.to_datetime(window_start)
            window_end_dt = pd.to_datetime(window_end)
            
            # 筛选该窗口的数据
            window_data = data[(data['date'] >= window_start_dt) & (data['date'] <= window_end_dt)].copy()

            if not window_data.empty:
                # 判断是否为起始窗口
                is_current_window = (window_key == current_window_key)
                is_first_window = False
                
                if not is_current_window and query_start < actual_start:
                    # 查询起始 < 数据源返回的最早日期，检查是否包含最早数据
                    if actual_start >= window_start_dt and actual_start <= window_end_dt:
                        is_first_window = True
                        logger.info(f"🅰️ 检测到起始窗口: {window_key} (查询从 {query_start.date()}，但数据源最早从 {actual_start.date()} 开始)")
                
                # 写入缓存（两种缓存现在都支持is_first_window参数）
                self._fast_cache.set(symbol, period, window_key, window_data, is_first_window=is_first_window)
                cached_windows[window_key] = window_data
                logger.debug(f"  ✅ 窗口 {window_key}: {len(window_data)} 条")


    
    def _backfill_first_window_flag(self, symbol: str, period: str, actual_start: pd.Timestamp, current_window_keys: list,market_code: MarketCode=MarketCode.CN) -> None:
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
        first_data_window_key = self._window_mgr.make_window_key(actual_start, period, self._window_size, market_code)
        
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
    

