"""
历史数据获取编排模块（从 DataFetcher.get_historical_data 和相关方法迁移而来）

职责：
1. 编排历史数据获取流程
2. 管理并发请求
3. 处理缓存和备用数据源
4. 更新性能指标
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


async def get_historical_data(
    fetcher: Any,
    symbols: List[str],
    period: str = '1y',
    interval: str = '1d',
    data_type: str = 'ohlcv',
    adjustments: bool = True
) -> Dict[str, List[Any]]:
    """
    获取历史数据 - 完整生产实现（从 DataFetcher.get_historical_data 迁移而来）。
    
    编排完整的数据获取流程：
    1. 生成缓存键并检查缓存
    2. 并发获取所有符号的数据
    3. 处理失败符号的备用数据源
    4. 缓存成功结果
    5. 更新性能指标
    
    Args:
        fetcher: DataFetcher实例
        symbols: 股票代码列表
        period: 数据期间 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: 数据间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        data_type: 数据类型 (ohlcv, dividends, splits, all)
        adjustments: 是否调整价格（分红和拆股）
    
    Returns:
        包含市场数据的字典，键为符号，值为数据列表
    
    Example:
        >>> data = await get_historical_data(
        ...     fetcher, ['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True
        ... )
        >>> # {'AAPL': [...], 'MSFT': [...]}
    """
    start_time = time.time()
    results = {}
    failed_symbols = []
    
    try:
        # 1. 生成缓存键（使用 fetcher 的 cache_manager）
        cache_manager = getattr(fetcher, 'cache_manager', None)
        if cache_manager:
            cache_key = cache_manager.generate_key(symbols, period, interval, data_type, adjustments)
        else:
            # 向后兼容：如果 fetcher 没有 cache_manager，使用旧的 generate_key 函数
            from core_bak_refactored.infrastructure.cache.cache_manager import CacheManager
            temp_manager = CacheManager({'cache_enabled': False})
            cache_key = temp_manager.generate_key(symbols, period, interval, data_type, adjustments)
        
        # 2. 检查缓存
        from core_bak_refactored.core.data.cache.store import get_cached_data
        cached_data = await get_cached_data(fetcher, cache_key)
        
        if cached_data and fetcher.cache_enabled:
            fetcher.cache_stats['hits'] += 1
            fetcher.performance_metrics['cache_hits'] += 1
            logger.debug(f"缓存命中: {cache_key}")
            return cached_data
        
        fetcher.cache_stats['misses'] += 1
        fetcher.performance_metrics['cache_misses'] += 1
        
        # 3. 并发获取所有符号的数据
        tasks = []
        for symbol in symbols:
            task = fetch_symbol_data(
                fetcher, symbol, period, interval, data_type, adjustments
            )
            tasks.append(task)
        
        # 等待所有任务完成
        symbol_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 4. 处理结果
        for i, result in enumerate(symbol_results):
            symbol = symbols[i]
            
            if isinstance(result, Exception):
                logger.error(f"获取 {symbol} 数据失败: {result}")
                failed_symbols.append(symbol)
                continue
            
            if result:
                results[symbol] = result
                fetcher.performance_metrics['data_points_processed'] += len(result)
        
        # 5. 如果主数据源失败，尝试备用数据源
        if failed_symbols and hasattr(fetcher, 'fallback_sources') and fetcher.fallback_sources:
            logger.warning(f"主数据源失败，尝试备用数据源: {failed_symbols}")
            
            from core_bak_refactored.core.data.fallback.orchestrator import try_fallback_sources
            fallback_results = await try_fallback_sources(
                fetcher, failed_symbols, period, interval, data_type, adjustments
            )
            results.update(fallback_results)
        
        # 6. 缓存结果
        if results and fetcher.cache_enabled:
            from core_bak_refactored.core.data.cache.store import cache_data
            await cache_data(fetcher, cache_key, results)
        
        # 7. 更新性能指标
        duration = time.time() - start_time
        fetcher.performance_metrics['requests_total'] += len(symbols)
        fetcher.performance_metrics['requests_failed'] += len(failed_symbols)
        fetcher.performance_metrics['avg_response_time'] = (
            fetcher.performance_metrics.get('avg_response_time', 0) * 0.9 + duration * 0.1
        )
        fetcher.performance_metrics['last_update'] = datetime.now().isoformat()
        
        logger.info(
            f"历史数据获取完成: {len(results)} 成功, {len(failed_symbols)} 失败, "
            f"耗时: {duration:.2f}s"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"历史数据获取失败: {e}")
        raise


async def fetch_symbol_data(
    fetcher: Any,
    symbol: str,
    period: str,
    interval: str,
    data_type: str,
    adjustments: bool
) -> Optional[List[Any]]:
    """
    获取单个符号的数据（从 DataFetcher._fetch_symbol_data 迁移而来）。
    
    按优先级尝试数据源：
    1. 首先尝试主数据源
    2. 主数据源失败时，按顺序尝试备用数据源
    
    Args:
        fetcher: DataFetcher实例
        symbol: 股票代码
        period: 数据期间
        interval: 数据间隔
        data_type: 数据类型
        adjustments: 是否调整价格
    
    Returns:
        数据列表，失败返回None
    """
    try:
        # 首先尝试主数据源
        primary_source = getattr(fetcher, 'primary_source', 'yahoo_finance')
        data_sources = getattr(fetcher, 'data_sources', {})
        
        primary_source_func = data_sources.get(primary_source)
        if primary_source_func:
            data = await primary_source_func(symbol, period, interval, data_type, adjustments)
            if data:
                return data
        
        # 如果主数据源失败，按优先级尝试备用数据源
        fallback_sources = getattr(fetcher, 'fallback_sources', [])
        for fallback_source in fallback_sources:
            if fallback_source in data_sources:
                fallback_func = data_sources[fallback_source]
                try:
                    data = await fallback_func(symbol, period, interval, data_type, adjustments)
                    if data:
                        logger.info(f"备用数据源 {fallback_source} 成功获取 {symbol} 数据")
                        return data
                except Exception as e:
                    logger.warning(f"备用数据源 {fallback_source} 获取 {symbol} 失败: {e}")
                    continue
        
        return None
        
    except Exception as e:
        logger.error(f"获取 {symbol} 数据失败: {e}")
        return None


def calculate_success_rate(
    total_symbols: int,
    successful_symbols: int
) -> Dict[str, Any]:
    """
    计算数据获取成功率。
    
    Args:
        total_symbols: 总符号数
        successful_symbols: 成功符号数
    
    Returns:
        成功率统计字典
    """
    if total_symbols == 0:
        return {
            'total_symbols': 0,
            'successful_symbols': 0,
            'failed_symbols': 0,
            'success_rate': 0,
            'failure_rate': 0
        }
    
    failed_symbols = total_symbols - successful_symbols
    success_rate = successful_symbols / total_symbols
    failure_rate = failed_symbols / total_symbols
    
    return {
        'total_symbols': total_symbols,
        'successful_symbols': successful_symbols,
        'failed_symbols': failed_symbols,
        'success_rate': success_rate,
        'failure_rate': failure_rate
    }
