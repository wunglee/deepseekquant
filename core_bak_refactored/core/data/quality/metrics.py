"""数据质量指标模块

职责：
- 提供数据质量指标的聚合和计算
- 不依赖data_fetcher，使用DataQualityChecker
"""

from typing import Any, Dict, List
import pandas as pd


from core_bak_refactored.core.data.quality.data_quality_checker import DataQualityChecker
from core_bak_refactored.core.data.quality.quality_types import DataQualityReport


def check_dataframe_quality(df: pd.DataFrame,
                           index_id: str = 'unknown',
                           expected_days: int = None,
                           market: str = None) -> DataQualityReport:
    """检查DataFrame的数据质量
    
    Args:
        df: 包含date, close, volume列的DataFrame
        index_id: 指数代码
        expected_days: 期望天数
        market: 市场代码
        
    Returns:
        数据质量报告
    """
    checker = DataQualityChecker()
    return checker.check_quality(df, index_id, expected_days, market)


def get_quality_summary(reports: List[DataQualityReport]) -> Dict[str, Any]:
    """获取多个质量报告的汇总信息
    
    Args:
        reports: 质量报告列表
        
    Returns:
        汇总信息字典
    """
    if not reports:
        return {
            'total_reports': 0,
            'avg_score': 0.0,
            'pass_rate': 0.0,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    total_score = sum(r.overall_score for r in reports)
    passed_count = sum(1 for r in reports if r.passed)
    
    return {
        'total_reports': len(reports),
        'avg_score': total_score / len(reports),
        'pass_rate': passed_count / len(reports),
        'avg_completeness': sum(r.completeness_score for r in reports) / len(reports),
        'avg_consistency': sum(r.consistency_score for r in reports) / len(reports),
        'avg_accuracy': sum(r.accuracy_score for r in reports) / len(reports),
        'total_outliers': sum(r.outliers_detected for r in reports),
        'total_issues': sum(len(r.issues) for r in reports),
        'timestamp': pd.Timestamp.now().isoformat()
    }
