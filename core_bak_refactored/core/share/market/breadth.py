"""
市场广度模块（从 DataFetcher._get_advance_decline 迁移而来）

职责：
1. 获取涨跌家数数据
2. 计算涨跌比率
3. 评估市场广度指标
"""
from typing import Dict, List, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


async def get_advance_decline(fetcher: Any, symbols: List[str] = None) -> Dict[str, Any]:
    """
    获取涨跌家数数据（从 DataFetcher._get_advance_decline 迁移而来）。
    
    Args:
        fetcher: DataFetcher实例，包含get_real_time_data方法
        symbols: 要分析的股票代码列表，默认None表示使用主要指数成分股
    
    Returns:
        涨跌家数统计字典，包含：
        - advances: 上涨家数
        - declines: 下跌家数
        - unchanged: 持平家数
        - advance_decline_ratio: 涨跌比率
        - total_issues: 总数
        - timestamp: 时间戳
    
    Example:
        >>> result = await get_advance_decline(fetcher)
        >>> # {'advances': 120, 'declines': 80, 'unchanged': 10, ...}
    """
    try:
        # 如果未提供符号，获取主要指数成分股
        if symbols is None:
            # 获取SPY成分股（这里简化处理，实际应调用专门的方法）
            symbols = _get_default_symbols(fetcher)

        # 限制数量以避免API过载
        symbols = symbols[:100]

        # 获取实时价格数据
        realtime_data = await fetcher.get_real_time_data(symbols)

        # 计算涨跌家数
        advances = 0
        declines = 0
        unchanged = 0

        for symbol, data in realtime_data.items():
            # 检查是否有涨跌数据
            if hasattr(data, 'metadata') and 'change' in data.metadata:
                change = data.metadata['change']
            elif isinstance(data, dict) and 'change' in data:
                change = data['change']
            else:
                # 无法获取涨跌数据，跳过
                continue

            # 分类统计
            if change > 0:
                advances += 1
            elif change < 0:
                declines += 1
            else:
                unchanged += 1

        # 计算涨跌比率
        if declines > 0:
            advance_decline_ratio = advances / declines
        else:
            advance_decline_ratio = float('inf') if advances > 0 else 0

        total_issues = advances + declines + unchanged

        result = {
            'advances': advances,
            'declines': declines,
            'unchanged': unchanged,
            'advance_decline_ratio': advance_decline_ratio,
            'total_issues': total_issues,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"市场广度统计完成: 涨 {advances}, 跌 {declines}, 平 {unchanged}, "
            f"涨跌比 {advance_decline_ratio:.2f}"
        )

        return result

    except Exception as e:
        logger.warning(f"获取涨跌家数失败: {e}")
        return {
            'advances': 0,
            'declines': 0,
            'unchanged': 0,
            'advance_decline_ratio': 0,
            'total_issues': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def _get_default_symbols(fetcher: Any) -> List[str]:
    """
    获取默认的股票符号列表（主要指数成分股）。
    
    Args:
        fetcher: DataFetcher实例
    
    Returns:
        股票代码列表
    """
    # 简化实现：返回一些常见的大盘股作为示例
    # 实际生产中应该从配置或API获取真实的指数成分股
    default_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'TSLA', 'NVDA', 'JPM', 'V', 'JNJ',
        'WMT', 'PG', 'MA', 'HD', 'DIS',
        'PYPL', 'NFLX', 'ADBE', 'CRM', 'CSCO'
    ]
    
    return default_symbols
