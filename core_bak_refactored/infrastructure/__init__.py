# Package marker for core_bak_refactored.infrastructure

# 导出通用工具（2025-11-28 彻底优化新增）
from .error_handling import (
    safe_execute,
    safe_numeric_operation,
    ErrorContext,
    validate_and_execute
)

from .data_validators import (
    LengthValidator,
    TypeValidator,
    NumericValidator,
    DataQualityValidator
)

from .config_utils import (
    ConfigExtractor,
    ConfigValidator,
    ThresholdManager
)

from .numeric_utils import (
    SafeNumericConverter,
    NumericCleaner,
    StatisticalNormalizer,
    RatioCalculator
)

from .statistical_calculators import StatisticalCalculator
from .timeseries_calculator import TimeSeriesCalculator, TechnicalIndicators

# HTTP客户端基础设施（2025-11-28 从core/data/http迁移）
from .http import setup_http_client, close_http_client

# 缓存基础设施（2025-11-28 从core/data/cache迁移）
from .cache import (
    MemoryTTLCache,
    RedisCacheAdapter,
    setup_redis_cache,
    get_cached_data,
    cache_data
)

# 数据验证基础设施（2025-11-28 从core/data/validation迁移）
from .validation import (
    validate_market_data,
    validate_data_list,
    clean_market_data
)

# 连接管理基础设施（2025-11-28 从core/data/connection迁移）
from .connection import DataConnectionManager

__all__ = [
    # 错误处理
    'safe_execute',
    'safe_numeric_operation',
    'ErrorContext',
    'validate_and_execute',
    
    # 数据验证
    'LengthValidator',
    'TypeValidator',
    'NumericValidator',
    'DataQualityValidator',
    
    # 配置管理
    'ConfigExtractor',
    'ConfigValidator',
    'ThresholdManager',
    
    # 数值工具
    'SafeNumericConverter',
    'NumericCleaner',
    'StatisticalNormalizer',
    'RatioCalculator',
    
    # 统计与时序计算
    'StatisticalCalculator',
    'TimeSeriesCalculator',
    'TechnicalIndicators',  # 别名兼容
    
    # HTTP客户端
    'setup_http_client',
    'close_http_client',
    
    # 缓存系统
    'MemoryTTLCache',
    'RedisCacheAdapter',
    'setup_redis_cache',
    'get_cached_data',
    'cache_data',
    
    # 数据验证
    'validate_market_data',
    'validate_data_list',
    'clean_market_data',
    
    # 连接管理
    'DataConnectionManager',
]
