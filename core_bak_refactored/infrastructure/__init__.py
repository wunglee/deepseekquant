# Package marker for core_bak_refactored.infrastructure

# 导出通用工具（2025-11-28 彻底优化新增）
from .error_handling import (
    safe_execute,
    safe_numeric_operation,
    ErrorContext,
    validate_and_execute
)

from .type_validators import (
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

# 新增的数据质量计算工具
from .statistical_quality_metrics import StatisticalQualityMetrics

# 新增的系统健康度计算工具
from .system_health_calculators import SystemHealthCalculators

# 新增的质量分析计算工具
from .quality_analysis_calculators import QualityAnalysisCalculators

# 新增的异常检测器
from .anomaly_detectors import ZScoreDetector, IQRDetector, RollingStdDetector, AnomalyDetectionManager, AnomalyResult

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

# 连接管理基础设施（2025-11-28 从core/data/connection迁移）
from .connection import DataConnectionManager

# 数据库层（2025-12-03新增）
from .database import (
    get_database,
    SQLiteDatabase,
    MarketDataRepository,
    DatabaseProtocol
)

# 注意：MarketData业务验证已迁移到 core.data.validation
# 通用技术验证请使用 type_validators（LengthValidator, TypeValidator等）

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
    
    # 数据质量计算工具
    'StatisticalQualityMetrics',
    
    # 系统健康度计算工具
    'SystemHealthCalculators',
    
    # 质量分析计算工具
    'QualityAnalysisCalculators',
    
    # 异常检测器
    'ZScoreDetector',
    'IQRDetector',
    'RollingStdDetector',
    'AnomalyDetectionManager',
    'AnomalyResult',
    
    # HTTP客户端
    'setup_http_client',
    'close_http_client',
    
    # 缓存系统
    'MemoryTTLCache',
    'RedisCacheAdapter',
    'setup_redis_cache',
    'get_cached_data',
    'cache_data',
    
    # 连接管理
    'DataConnectionManager',
    
    # 数据库（2025-12-03新增）
    'get_database',
    'SQLiteDatabase',
    'MarketDataRepository',
    'DatabaseProtocol',
]
