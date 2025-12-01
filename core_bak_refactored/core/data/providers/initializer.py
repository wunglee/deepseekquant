"""
数据源初始化模块（从 DataFetcher._initialize_data_sources 迁移而来）

职责：
1. 注册所有可用的数据源
2. 支持依赖注入（外部custom_sources）
3. 验证数据源方法存在性
4. 记录初始化日志
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def initialize_data_sources(fetcher: Any, custom_sources: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    初始化数据源连接（从 DataFetcher._initialize_data_sources 迁移而来）。
    
    采用渐进式注册：只注册实现了的数据源方法
    支持依赖注入：优先使用外部注入的custom_sources
    
    Args:
        fetcher: DataFetcher 实例，包含数据源方法
        custom_sources: 外部注入的自定义数据源字典（用于测试Mock或扩展）
    
    Returns:
        数据源字典，键为数据源类型，值为数据源方法
    
    Example:
        >>> sources = initialize_data_sources(fetcher, {'custom': custom_fetch_func})
        >>> # {'custom': <function>, 'yahoo_finance': <bound method>, ...}
    """
    sources = {}

    # 1. 注入外部自定义数据源（用于测试Mock或扩展）
    if custom_sources:
        sources.update(custom_sources)
        logger.info(f"注入自定义数据源: {list(custom_sources.keys())}")

    # 2. 注册已实现的内置数据源（只注册存在的方法）
    # 定义数据源类型到方法名的映射
    source_mapping = {
        'yahoo_finance': '_fetch_yahoo_data',
        'alpha_vantage': '_fetch_alpha_vantage_data',
        'iex_cloud': '_fetch_iex_cloud_data',
        'polygon': '_fetch_polygon_data',
        'twelve_data': '_fetch_twelve_data',
        'finnhub': '_fetch_finnhub_data',
        'tiingo': '_fetch_tiingo_data',
        'quandl': '_fetch_quandl_data',
        'intrinio': '_fetch_intrinio_data',
        'eod_historical': '_fetch_eod_historical_data',
        'custom_api': '_fetch_custom_api_data',
        'database': '_fetch_database_data',
        'broker_api': '_fetch_broker_api_data',
    }

    # 只注册已实现的方法（避免AttributeError）
    registered_count = 0
    for source_type, method_name in source_mapping.items():
        if source_type not in sources and hasattr(fetcher, method_name):
            sources[source_type] = getattr(fetcher, method_name)
            registered_count += 1
        elif source_type not in sources:
            logger.debug(f"数据源 {source_type} 未实现，跳过注册")

    logger.info(f"已注册 {registered_count} 个内置数据源: {list(sources.keys())}")
    
    return sources
