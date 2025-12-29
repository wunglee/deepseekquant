"""数据质量模块

职责：
- 数据质量检查（DataQualityChecker）
- 数据质量类型定义（quality_types）
- 数据质量指标计算（metrics）
- ML异常检测（anomaly_detectors）- 可选

设计原则：
- 不依赖data_fetcher，保持独立性
- 可被历史数据提供者、实时数据流等多处使用
- 支持自定义检查规则和阈值
- ML异常检测可选启用（避免强依赖sklearn）
"""
from core_bak_refactored.core.data.quality.data_quality_enhancer import DataQualityEnhancer
from core_bak_refactored.core.data.quality.data_quality_checker import (
    DataQualityChecker,
    CrossValidationResult
)
from core_bak_refactored.core.data.quality.quality_types import (
    DataQualityReport,
    QualityReport,
    QualityFlags,
    Issue
)
from core_bak_refactored.core.data.quality.metrics import (
    check_dataframe_quality,
    get_quality_summary
)

# ML异常检测（可选导入，避免强依赖）
try:
    from core_bak_refactored.core.data.quality.anomaly_detectors import (
        AnomalyDetector,
        AnomalyResult,
        ZScoreDetector,
        IQRDetector,
        RollingStdDetector,
        IsolationForestDetector,
        LOFDetector,
        AnomalyDetectorManager
    )
    _ML_DETECTORS_AVAILABLE = True
except ImportError:
    _ML_DETECTORS_AVAILABLE = False

__all__ = [
    # 核心检查器
    'DataQualityChecker',
    'CrossValidationResult',
    
    # 类型定义
    'DataQualityReport',
    'QualityReport',
    'QualityFlags',
    'Issue',
    
    # 工具函数
    'check_dataframe_quality',
    'get_quality_summary',
    
    # 增强器
    'DataQualityEnhancer'
]

# ML检测器（可选）
if _ML_DETECTORS_AVAILABLE:
    __all__.extend([
        'AnomalyDetector',
        'AnomalyResult',
        'ZScoreDetector',
        'IQRDetector',
        'RollingStdDetector',
        'IsolationForestDetector',
        'LOFDetector',
        'AnomalyDetectorManager',
    ])
