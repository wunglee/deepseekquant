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
]
