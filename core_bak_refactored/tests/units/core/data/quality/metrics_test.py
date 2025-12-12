"""数据质量指标测试

测试覆盖：
- check_dataframe_quality: 检查DataFrame质量
- get_quality_summary: 获取质量报告汇总
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from core_bak_refactored.core.data.quality.metrics import (
    check_dataframe_quality,
    get_quality_summary
)
from core_bak_refactored.core.data.quality import DataQualityReport


def test_check_dataframe_quality():
    """测试DataFrame质量检查"""
    df = pd.DataFrame({
        'date': pd.date_range('2015-06-01', periods=100),
        'close': np.linspace(100, 110, 100),
        'volume': [1000000] * 100
    })
    
    report = check_dataframe_quality(df, index_id='000300.SH', expected_days=100)
    
    assert isinstance(report, DataQualityReport)
    assert report.overall_score >= 0.9
    assert report.passed == True


def test_check_dataframe_quality_with_market():
    """测试带市场参数的质量检查"""
    df = pd.DataFrame({
        'date': pd.date_range('2015-06-01', periods=50),
        'close': np.linspace(100, 105, 50),
        'volume': [5000000] * 50
    })
    
    report = check_dataframe_quality(df, index_id='HSI', market='HK')
    
    assert isinstance(report, DataQualityReport)
    assert report.overall_score > 0.0


def test_get_quality_summary_empty():
    """测试空报告列表的汇总"""
    summary = get_quality_summary([])
    
    assert summary['total_reports'] == 0
    assert summary['avg_score'] == 0.0
    assert summary['pass_rate'] == 0.0
    assert 'timestamp' in summary


def test_get_quality_summary_multiple_reports():
    """测试多个报告的汇总"""
    df1 = pd.DataFrame({
        'date': pd.date_range('2015-06-01', periods=100),
        'close': np.linspace(100, 110, 100),
        'volume': [1000000] * 100
    })
    
    df2 = pd.DataFrame({
        'date': pd.date_range('2015-06-01', periods=80),
        'close': np.linspace(100, 108, 80),
        'volume': [800000] * 80
    })
    
    report1 = check_dataframe_quality(df1, expected_days=100)
    report2 = check_dataframe_quality(df2, expected_days=100)
    
    summary = get_quality_summary([report1, report2])
    
    assert summary['total_reports'] == 2
    assert 0.0 <= summary['avg_score'] <= 1.0
    assert 0.0 <= summary['pass_rate'] <= 1.0
    assert 'avg_completeness' in summary
    assert 'avg_consistency' in summary
    assert 'avg_accuracy' in summary  # 更新为 accuracy
    assert 'total_outliers' in summary  # 更新为 outliers
    assert 'total_issues' in summary
