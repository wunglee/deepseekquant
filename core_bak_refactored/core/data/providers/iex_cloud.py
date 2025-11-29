"""
IEX Cloud数据提供者

职责：
1. 从IEX Cloud获取历史OHLCV数据
2. 支持多种数据类型（quote、historical、stats等）
3. 处理API认证和版本控制
4. 支持批量请求优化
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from urllib.parse import urlencode
import logging

logger = logging.getLogger(__name__)


async def fetch_iex_cloud_data(
    symbol: str,
    period: str,
    interval: str,
    data_type: str,
    adjustments: bool,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[List[Dict]]:
    """
    从IEX Cloud获取数据。
    
    Args:
        symbol: 股票代码
        period: 数据期间
        interval: 数据间隔
        data_type: 数据类型
        adjustments: 是否调整价格
        api_credentials: API凭证
        aiohttp_session: aiohttp会话
    
    Returns:
        数据字典列表，失败返回None
    """
    try:
        # 验证API密钥
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("IEX Cloud API密钥未配置")
        
        # 确定API端点
        endpoint = _determine_endpoint(interval, period)
        
        # 构建API URL
        base_url = api_credentials.get('base_url', 'https://cloud.iexapis.com/stable')
        api_key = api_credentials['api_key']
        
        url = f"{base_url}/stock/{symbol}/{endpoint}?token={api_key}"
        
        # 添加查询参数
        params = {}
        if endpoint == 'chart':
            range_param = _map_period_to_iex_range(period)
            url += f"&range={range_param}"
            if interval == '1m' or interval == '5m':
                url += "&chartByDay=true"
        
        # 发送HTTP请求
        async with aiohttp_session.get(url) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"API请求失败，HTTP状态码: {response.status}, 错误: {error_text}")
            
            data = await response.json()
            
            # IEX Cloud返回数组
            if not isinstance(data, list) or len(data) == 0:
                logger.warning(f"IEX Cloud未返回 {symbol} 的数据")
                return None
            
            # 转换为标准格式
            market_data_list = []
            for item in data:
                # IEX Cloud的日期格式
                date_str = item.get('date')
                if not date_str:
                    continue
                
                # 解析时间戳
                try:
                    if 'minute' in item:
                        # 分钟级数据
                        timestamp = datetime.strptime(f"{date_str} {item['minute']}", '%Y-%m-%d %H:%M')
                    else:
                        # 日线数据
                        timestamp = datetime.strptime(date_str, '%Y-%m-%d')
                except (ValueError, KeyError):
                    logger.warning(f"无法解析时间戳: {date_str}")
                    continue
                
                # 构造数据点
                data_point = {
                    'symbol': symbol,
                    'timestamp': timestamp,
                    'open': float(item.get('open', 0)),
                    'high': float(item.get('high', 0)),
                    'low': float(item.get('low', 0)),
                    'close': float(item.get('close', 0)),
                    'volume': float(item.get('volume', 0)),
                    'unadjusted_volume': float(item.get('uVolume', 0)),
                    'change': float(item.get('change', 0)),
                    'change_percent': float(item.get('changePercent', 0)),
                    'metadata': {
                        'data_source': 'iex_cloud',
                        'data_type': data_type,
                        'interval': interval,
                        'adjustments': adjustments,
                        'change_over_time': item.get('changeOverTime', 0)
                    }
                }
                market_data_list.append(data_point)
            
            logger.info(f"IEX Cloud成功获取 {symbol} 数据，共 {len(market_data_list)} 条记录")
            return market_data_list
            
    except Exception as e:
        logger.error(f"IEX Cloud数据获取失败 ({symbol}): {e}")
        return None


def _determine_endpoint(interval: str, period: str) -> str:
    """
    根据间隔和期间确定API端点。
    
    Args:
        interval: 时间间隔
        period: 数据期间
    
    Returns:
        API端点字符串
    """
    if interval in ['1m', '5m', '15m', '30m', '1h']:
        return 'intraday-prices'
    else:
        return 'chart'


def _map_period_to_iex_range(period: str) -> str:
    """
    映射期间到IEX Cloud的range参数。
    
    Args:
        period: 期间字符串
    
    Returns:
        IEX Cloud的range值
    """
    mapping = {
        '1d': '1d',
        '5d': '5d',
        '1mo': '1m',
        '3mo': '3m',
        '6mo': '6m',
        '1y': '1y',
        '2y': '2y',
        '5y': '5y',
        'ytd': 'ytd',
        'max': 'max'
    }
    
    return mapping.get(period, '1y')


async def fetch_iex_quote(
    symbol: str,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[Dict]:
    """
    从IEX Cloud获取实时报价。
    
    Args:
        symbol: 股票代码
        api_credentials: API凭证
        aiohttp_session: aiohttp会话
    
    Returns:
        报价字典，失败返回None
    """
    try:
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("IEX Cloud API密钥未配置")
        
        base_url = api_credentials.get('base_url', 'https://cloud.iexapis.com/stable')
        api_key = api_credentials['api_key']
        
        url = f"{base_url}/stock/{symbol}/quote?token={api_key}"
        
        async with aiohttp_session.get(url) as response:
            if response.status != 200:
                return None
            
            quote = await response.json()
            
            return {
                'symbol': symbol,
                'latest_price': quote.get('latestPrice'),
                'latest_time': quote.get('latestTime'),
                'open': quote.get('open'),
                'high': quote.get('high'),
                'low': quote.get('low'),
                'close': quote.get('close'),
                'volume': quote.get('volume'),
                'market_cap': quote.get('marketCap'),
                'pe_ratio': quote.get('peRatio'),
                'week_52_high': quote.get('week52High'),
                'week_52_low': quote.get('week52Low'),
                'ytd_change': quote.get('ytdChange')
            }
            
    except Exception as e:
        logger.error(f"IEX Cloud报价获取失败 ({symbol}): {e}")
        return None
