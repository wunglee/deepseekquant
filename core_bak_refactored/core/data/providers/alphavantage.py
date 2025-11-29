"""
Alpha Vantage数据提供者（从 DataFetcher._fetch_alpha_vantage_data 迁移而来）

职责：
1. 从Alpha Vantage获取历史OHLCV数据
2. 支持多种时间间隔（分钟级、日线、周线、月线）
3. 处理API密钥认证
4. 解析JSON响应为统一数据格式
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlencode
import logging

logger = logging.getLogger(__name__)


async def fetch_alpha_vantage_data(
    symbol: str,
    period: str,
    interval: str,
    data_type: str,
    adjustments: bool,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[List[Dict]]:
    """
    从Alpha Vantage获取数据（从 DataFetcher._fetch_alpha_vantage_data 迁移而来）。
    
    Args:
        symbol: 股票代码
        period: 数据期间 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: 数据间隔 (1m, 5m, 15m, 30m, 60m, daily, weekly, monthly)
        data_type: 数据类型 (ohlcv)
        adjustments: 是否调整价格
        api_credentials: API凭证字典，包含api_key和base_url
        aiohttp_session: aiohttp会话对象
    
    Returns:
        数据字典列表，失败返回None
    
    Raises:
        ValueError: API密钥未配置或不支持的间隔
    
    Example:
        >>> credentials = {'api_key': 'YOUR_KEY', 'base_url': 'https://...'}
        >>> data = await fetch_alpha_vantage_data('AAPL', '1y', 'daily', 'ohlcv', True, credentials, session)
        >>> # [{'symbol': 'AAPL', 'timestamp': ..., 'open': ..., ...}, ...]
    """
    try:
        # 验证API凭证
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("Alpha Vantage API密钥未配置")

        # 根据时间间隔选择API函数
        if interval in ['1m', '5m', '15m', '30m', '60m']:
            function = 'TIME_SERIES_INTRADAY'
            interval_param = interval
        elif interval in ['1d', 'daily']:
            function = 'TIME_SERIES_DAILY'
            interval_param = None
        elif interval in ['1wk', 'weekly']:
            function = 'TIME_SERIES_WEEKLY'
            interval_param = None
        elif interval in ['1mo', 'monthly']:
            function = 'TIME_SERIES_MONTHLY'
            interval_param = None
        else:
            raise ValueError(f"不支持的间隔: {interval}")

        # 构建API参数
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': api_credentials['api_key'],
            'outputsize': 'full' if period in ['max', '10y', '5y'] else 'compact',
            'datatype': 'json'
        }

        # 添加间隔参数（仅分钟级数据需要）
        if interval_param:
            params['interval'] = interval_param

        # 添加调整参数
        if adjustments:
            params['adjusted'] = 'true'

        # 构建完整URL
        base_url = api_credentials.get('base_url', 'https://www.alphavantage.co/query')
        url = f"{base_url}?{urlencode(params)}"

        # 发送HTTP请求
        async with aiohttp_session.get(url) as response:
            if response.status != 200:
                raise ValueError(f"API请求失败，HTTP状态码: {response.status}")

            data = await response.json()

            # 检查API错误响应
            if 'Error Message' in data:
                raise ValueError(f"API返回错误: {data['Error Message']}")
            
            if 'Note' in data:
                logger.warning(f"API限流提示: {data['Note']}")
                return None

            # 解析时间序列数据
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break

            if not time_series_key:
                raise ValueError(f"未找到时间序列数据，响应键: {list(data.keys())}")

            time_series = data[time_series_key]
            market_data_list = []

            # 遍历时间序列数据
            for timestamp_str, values in time_series.items():
                # 解析时间戳（支持多种格式）
                try:
                    if len(timestamp_str) > 10:  # 包含时间
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    else:  # 仅日期
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d')
                except ValueError:
                    logger.warning(f"无法解析时间戳: {timestamp_str}")
                    continue

                # 构造数据点
                data_point = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(values.get('1. open', 0)),
                    'high': float(values.get('2. high', 0)),
                    'low': float(values.get('3. low', 0)),
                    'close': float(values.get('4. close', 0)),
                    'volume': float(values.get('5. volume', 0)),
                    'metadata': {
                        'data_source': 'alpha_vantage',
                        'data_type': data_type,
                        'function': function,
                        'interval': interval,
                        'adjustments': adjustments
                    }
                }
                market_data_list.append(data_point)

            # 按时间排序（从旧到新）
            market_data_list.sort(key=lambda x: x['timestamp'])

            logger.info(f"Alpha Vantage成功获取 {symbol} 数据，共 {len(market_data_list)} 条记录")
            return market_data_list

    except Exception as e:
        logger.error(f"Alpha Vantage数据获取失败 ({symbol}): {e}")
        return None
