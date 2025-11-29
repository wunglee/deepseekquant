"""
Twelve Data数据提供者

职责：
1. 从Twelve Data获取股票数据
2. 支持多种时间间隔和数据类型
3. 提供全球市场覆盖
4. 处理复杂API参数和认证
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


async def fetch_twelve_data(
    symbol: str,
    period: str,
    interval: str,
    data_type: str,
    adjustments: bool,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[List[Dict]]:
    """
    从Twelve Data获取时间序列数据。
    
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
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("Twelve Data API密钥未配置")
        
        # 构建API URL
        base_url = api_credentials.get('base_url', 'https://api.twelvedata.com')
        api_key = api_credentials['api_key']
        
        # 映射时间间隔
        interval_param = _map_interval_to_twelve_data(interval)
        
        # 计算输出大小
        outputsize = _calculate_outputsize(period, interval)
        
        # 构建查询参数
        params = {
            'symbol': symbol,
            'interval': interval_param,
            'apikey': api_key,
            'outputsize': outputsize,
            'format': 'JSON'
        }
        
        # 是否返回调整后数据
        if adjustments:
            params['type'] = 'stock'
        
        url = f"{base_url}/time_series"
        
        # 发送HTTP请求
        async with aiohttp_session.get(url, params=params) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"API请求失败，HTTP状态码: {response.status}, 错误: {error_text}")
            
            data = await response.json()
            
            # 检查错误响应
            if 'status' in data and data['status'] == 'error':
                logger.warning(f"Twelve Data错误: {data.get('message', 'Unknown error')}")
                return None
            
            # 解析时间序列数据
            values = data.get('values', [])
            if not values:
                logger.warning(f"Twelve Data未返回 {symbol} 的数据")
                return None
            
            # 转换为标准格式
            market_data_list = []
            for item in values:
                try:
                    # Twelve Data的时间格式
                    timestamp_str = item.get('datetime')
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    
                    data_point = {
                        'symbol': symbol,
                        'timestamp': timestamp,
                        'open': float(item.get('open', 0)),
                        'high': float(item.get('high', 0)),
                        'low': float(item.get('low', 0)),
                        'close': float(item.get('close', 0)),
                        'volume': float(item.get('volume', 0)),
                        'metadata': {
                            'data_source': 'twelve_data',
                            'data_type': data_type,
                            'interval': interval,
                            'adjustments': adjustments
                        }
                    }
                    market_data_list.append(data_point)
                except (ValueError, KeyError) as e:
                    logger.warning(f"解析数据点失败: {e}")
                    continue
            
            logger.info(f"Twelve Data成功获取 {symbol} 数据，共 {len(market_data_list)} 条记录")
            return market_data_list
            
    except Exception as e:
        logger.error(f"Twelve Data数据获取失败 ({symbol}): {e}")
        return None


async def fetch_twelve_data_quote(
    symbol: str,
    api_credentials: Dict[str, Any],
    aiohttp_session: Any
) -> Optional[Dict]:
    """
    从Twelve Data获取实时报价。
    
    Args:
        symbol: 股票代码
        api_credentials: API凭证
        aiohttp_session: aiohttp会话
    
    Returns:
        报价字典，失败返回None
    """
    try:
        if not api_credentials or not api_credentials.get('api_key'):
            raise ValueError("Twelve Data API密钥未配置")
        
        base_url = api_credentials.get('base_url', 'https://api.twelvedata.com')
        api_key = api_credentials['api_key']
        
        params = {
            'symbol': symbol,
            'apikey': api_key
        }
        
        url = f"{base_url}/quote"
        
        async with aiohttp_session.get(url, params=params) as response:
            if response.status != 200:
                return None
            
            quote = await response.json()
            
            if 'status' in quote and quote['status'] == 'error':
                return None
            
            return {
                'symbol': symbol,
                'name': quote.get('name'),
                'exchange': quote.get('exchange'),
                'currency': quote.get('currency'),
                'open': float(quote.get('open', 0)),
                'high': float(quote.get('high', 0)),
                'low': float(quote.get('low', 0)),
                'close': float(quote.get('close', 0)),
                'volume': float(quote.get('volume', 0)),
                'previous_close': float(quote.get('previous_close', 0)),
                'change': float(quote.get('change', 0)),
                'percent_change': float(quote.get('percent_change', 0)),
                'average_volume': float(quote.get('average_volume', 0)),
                'fifty_two_week': {
                    'low': float(quote.get('fifty_two_week', {}).get('low', 0)),
                    'high': float(quote.get('fifty_two_week', {}).get('high', 0)),
                    'change': float(quote.get('fifty_two_week', {}).get('change', 0)),
                    'change_percent': float(quote.get('fifty_two_week', {}).get('change_percent', 0))
                },
                'timestamp': datetime.fromisoformat(quote.get('datetime', '').replace('Z', '+00:00')) if quote.get('datetime') else None
            }
            
    except Exception as e:
        logger.error(f"Twelve Data报价获取失败 ({symbol}): {e}")
        return None


def _map_interval_to_twelve_data(interval: str) -> str:
    """
    映射时间间隔到Twelve Data格式。
    
    支持的间隔:
    - 1min, 5min, 15min, 30min, 45min
    - 1h, 2h, 4h
    - 1day, 1week, 1month
    
    Args:
        interval: 时间间隔字符串
    
    Returns:
        Twelve Data间隔字符串
    """
    mapping = {
        '1m': '1min',
        '5m': '5min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '1d': '1day',
        '1wk': '1week',
        '1mo': '1month'
    }
    
    return mapping.get(interval, '1day')


def _calculate_outputsize(period: str, interval: str) -> int:
    """
    根据期间和间隔计算输出大小。
    
    Args:
        period: 期间字符串
        interval: 间隔字符串
    
    Returns:
        输出大小（数据点数量）
    """
    # 期间到天数的映射
    period_days = {
        '1d': 1,
        '5d': 5,
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825
    }
    
    # 间隔到每天数据点数的映射
    points_per_day = {
        '1m': 390,   # 6.5小时 * 60分钟
        '5m': 78,
        '15m': 26,
        '30m': 13,
        '1h': 6.5,
        '1d': 1,
        '1wk': 1/5,
        '1mo': 1/21
    }
    
    days = period_days.get(period, 365)
    points = points_per_day.get(interval, 1)
    
    # 计算总数据点，限制在5000以内（API限制）
    total_points = int(days * points)
    return min(total_points, 5000)
