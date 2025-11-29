"""
Yahoo Finance数据提供者（从 DataFetcher._fetch_yahoo_data 迁移而来）

职责：
1. 从Yahoo Finance获取历史OHLCV数据
2. 支持多种数据类型（ohlcv, dividends, splits, all）
3. 支持价格调整（分红和拆股）
4. 转换为MarketData对象列表
"""
from typing import List, Optional
from datetime import datetime
import logging
import yfinance as yf

logger = logging.getLogger(__name__)


async def fetch_yahoo_data(
    symbol: str,
    period: str,
    interval: str,
    data_type: str,
    adjustments: bool
) -> Optional[List]:
    """
    从Yahoo Finance获取数据（从 DataFetcher._fetch_yahoo_data 迁移而来）。
    
    Args:
        symbol: 股票代码
        period: 数据期间 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: 数据间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        data_type: 数据类型 (ohlcv, dividends, splits, all)
        adjustments: 是否调整价格（分红和拆股）
    
    Returns:
        MarketData对象列表，失败返回None
    
    Raises:
        ValueError: 不支持的数据类型
    
    Example:
        >>> data = await fetch_yahoo_data('AAPL', '1y', '1d', 'ohlcv', True)
        >>> # [MarketData(...), MarketData(...), ...]
    """
    try:
        ticker = yf.Ticker(symbol)

        # 根据数据类型获取不同的数据
        if data_type == 'ohlcv':
            hist = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=adjustments,
                actions=False
            )
        elif data_type == 'dividends':
            hist = ticker.dividends
        elif data_type == 'splits':
            hist = ticker.splits
        elif data_type == 'all':
            hist = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=adjustments,
                actions=True
            )
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")

        if hist.empty:
            logger.warning(f"Yahoo Finance未返回 {symbol} 的数据")
            return None

        # 转换为MarketData对象列表（需要导入MarketData类）
        # 注意：这里暂时返回原始DataFrame，后续整合时再转换为MarketData
        market_data_list = []
        for idx, row in hist.iterrows():
            # 提取时间戳
            timestamp = idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else idx
            
            # 构造数据字典（暂时使用字典，后续整合时转换为MarketData对象）
            data_point = {
                'symbol': symbol,
                'timestamp': timestamp,
                'open': row.get('Open', float('nan')),
                'high': row.get('High', float('nan')),
                'low': row.get('Low', float('nan')),
                'close': row.get('Close', float('nan')),
                'volume': row.get('Volume', 0),
                'adj_close': row.get('Adj Close', float('nan')),
                'dividends': row.get('Dividends', 0) if 'Dividends' in row else 0,
                'splits': row.get('Stock Splits', 1) if 'Stock Splits' in row else 1,
                'metadata': {
                    'data_source': 'yahoo',
                    'data_type': data_type,
                    'period': period,
                    'interval': interval,
                    'adjustments': adjustments
                }
            }
            market_data_list.append(data_point)

        logger.info(f"Yahoo Finance成功获取 {symbol} 数据，共 {len(market_data_list)} 条记录")
        return market_data_list

    except Exception as e:
        logger.error(f"Yahoo Finance数据获取失败 ({symbol}): {e}")
        return None
