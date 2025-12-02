from typing import Any, Dict, List
import asyncio
import logging

logger = logging.getLogger(__name__)


async def try_fallback_sources(
    fetcher: Any, 
    symbols: List[str], 
    period: str, 
    interval: str,
    data_type: str, 
    adjustments: bool
) -> Dict[str, List[Any]]:
    """
    尝试备用数据源（从 DataFetcher._try_fallback_sources 迁移而来）。
    
    实现策略：
    1. 顺序遍历备用数据源列表
    2. 对每个数据源，并发获取所有失败符号的数据
    3. 成功获取的符号从失败列表中移除
    4. 如果所有符号都成功，提前退出
    5. 否则继续尝试下一个备用源
    
    Args:
        fetcher: DataFetcher 实例，包含 fallback_sources 和 data_sources 属性
        symbols: 失败符号列表
        period: 数据期间
        interval: 数据间隔
        data_type: 数据类型
        adjustments: 是否调整价格
    
    Returns:
        成功获取的数据字典，键为符号，值为数据列表
    
    Example:
        >>> fallback_results = await try_fallback_sources(
        ...     fetcher, ['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True
        ... )
        >>> # {'AAPL': [MarketData(...), ...], 'MSFT': [MarketData(...), ...]}
    """
    fallback_results: Dict[str, List[Any]] = {}
    failed_symbols = symbols.copy()  # 复制列表避免修改原始参数

    # 遍历所有备用数据源
    for fallback_source in fetcher.fallback_sources:
        if not failed_symbols:  # 所有符号都已成功获取
            logger.info("所有符号已通过备用数据源成功获取")
            break

        try:
            # 获取数据源函数
            source_func = fetcher.data_sources.get(fallback_source)
            if not source_func:
                logger.debug(f"备用数据源 {fallback_source} 未注册，跳过")
                continue

            logger.info(
                f"尝试备用数据源: {fallback_source}，"
                f"剩余符号: {len(failed_symbols)} ({', '.join(failed_symbols[:5])}{'...' if len(failed_symbols) > 5 else ''})"
            )

            # 并发获取所有失败符号的数据
            tasks = []
            for symbol in failed_symbols:
                task = source_func(symbol, period, interval, data_type, adjustments)
                tasks.append(task)

            # 等待所有任务完成（包括异常）
            symbol_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理结果
            successful_symbols = []
            for i, result in enumerate(symbol_results):
                symbol = failed_symbols[i]

                if isinstance(result, Exception):
                    logger.debug(f"备用数据源 {fallback_source} 获取 {symbol} 失败: {result}")
                    continue

                if result:  # 成功获取数据
                    fallback_results[symbol] = result  # type: ignore[assignment]
                    successful_symbols.append(symbol)
                    logger.info(f"备用数据源 {fallback_source} 成功获取 {symbol} 数据")

            # 从失败列表中移除成功的符号
            failed_symbols = [s for s in failed_symbols if s not in successful_symbols]

            # 记录成功率
            if successful_symbols:
                logger.info(
                    f"备用数据源 {fallback_source} 成功获取 {len(successful_symbols)}/{len(symbol_results)} 个符号"
                )

            # 如果所有符号都已获取成功，提前退出
            if not failed_symbols:
                logger.info(f"备用数据源 {fallback_source} 已获取所有剩余符号")
                break

        except Exception as e:
            logger.warning(f"备用数据源 {fallback_source} 整体失败: {e}")
            continue

    # 最终统计
    if failed_symbols:
        logger.warning(
            f"所有备用数据源尝试完毕，仍有 {len(failed_symbols)} 个符号失败: {', '.join(failed_symbols)}"
        )
    else:
        logger.info(f"所有符号均通过备用数据源获取成功")

    return fallback_results
