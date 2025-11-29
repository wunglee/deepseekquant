from typing import List
import hashlib


def generate_key(symbols: List[str], period: str, interval: str, data_type: str, adjustments: bool) -> str:
    """
    生成缓存键（从 DataFetcher._generate_cache_key 迁移而来）。
    
    Args:
        symbols: 股票代码列表
        period: 数据期间 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: 数据间隔 (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        data_type: 数据类型 (ohlcv, dividends, splits, all)
        adjustments: 是否调整价格（分红和拆股）
    
    Returns:
        MD5哈希值（32字符十六进制字符串）
    
    Example:
        >>> generate_key(['AAPL', 'MSFT'], '1y', '1d', 'ohlcv', True)
        'a1b2c3d4e5f6...'  # MD5 hash
    """
    # 排序符号以确保键的一致性（无论顺序）
    symbols_str = '_'.join(sorted(symbols))
    
    # 组合所有参数
    key_data = f"{symbols_str}_{period}_{interval}_{data_type}_{adjustments}"
    
    # 生成MD5哈希
    return hashlib.md5(key_data.encode()).hexdigest()
