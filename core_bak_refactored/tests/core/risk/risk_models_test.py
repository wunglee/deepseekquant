"""
风险数据模型测试
"""

import os
import sys
import pytest
from datetime import datetime, timedelta

# 确保可以从项目根导入模块
sys.path.insert(0, os.path.abspath('.'))

from core_bak_refactored.core.risk.risk_models import (
    RiskLevel,
    RiskType,
    RiskMetric,
    RecommendationType,
    TimeHorizon,
    CalculationMethod,
    ImpactLevel,
    LimitBreach,
    Recommendation,
)


def test_risk_level_from_score_boundaries():
    assert RiskLevel.from_score(0) == RiskLevel.VERY_LOW
    assert RiskLevel.from_score(19.9) == RiskLevel.VERY_LOW
    assert RiskLevel.from_score(20) == RiskLevel.LOW
    assert RiskLevel.from_score(39.9) == RiskLevel.LOW
    assert RiskLevel.from_score(40) == RiskLevel.MODERATE
    assert RiskLevel.from_score(59.9) == RiskLevel.MODERATE
    assert RiskLevel.from_score(60) == RiskLevel.HIGH
    assert RiskLevel.from_score(79.9) == RiskLevel.HIGH
    assert RiskLevel.from_score(80) == RiskLevel.VERY_HIGH
    assert RiskLevel.from_score(94.9) == RiskLevel.VERY_HIGH
    assert RiskLevel.from_score(95) == RiskLevel.EXTREME


def test_risk_level_legacy_black_swan_maps_to_extreme():
    assert RiskLevel.from_legacy_value('black_swan') == RiskLevel.EXTREME
    # 非black_swan的字符串应能正常映射到枚举
    assert RiskLevel.from_legacy_value('high') == RiskLevel.HIGH


def test_limit_breach_serialization_roundtrip():
    lb = LimitBreach(
        limit_id='LIM-001',
        risk_type=RiskType.MARKET_RISK,
        metric=RiskMetric.VALUE_AT_RISK,
        current_value=1.23,
        threshold=1.00,
        breach_amount=0.23,
        timestamp=datetime(2024, 11, 14, 12, 0, 0),
        severity=RiskLevel.HIGH,
        breach_duration_seconds=60,
    )
    d = lb.to_dict()
    lb2 = LimitBreach.from_dict(d)
    assert lb2.limit_id == 'LIM-001'
    assert lb2.risk_type == RiskType.MARKET_RISK
    assert lb2.metric == RiskMetric.VALUE_AT_RISK
    assert isinstance(lb2.timestamp, datetime)
    assert lb2.severity == RiskLevel.HIGH
    assert lb2.breach_duration_seconds == 60


def test_limit_breach_from_dict_with_strings_and_old_field():
    ts = datetime(2024, 11, 14, 10, 30, 0).isoformat()
    data = {
        'limit_id': 'LIM-002',
        'risk_type': 'market_risk',
        'metric': 'value_at_risk',
        'current_value': 2.5,
        'threshold': 2.0,
        'breach_amount': 0.5,
        'timestamp': ts,
        'severity': 'high',
        'breach_duration': 120,  # 旧字段名，应被映射到breach_duration_seconds
    }
    lb = LimitBreach.from_dict(data)
    assert lb.limit_id == 'LIM-002'
    assert lb.risk_type == RiskType.MARKET_RISK
    assert lb.metric == RiskMetric.VALUE_AT_RISK
    assert isinstance(lb.timestamp, datetime)
    assert lb.severity == RiskLevel.HIGH
    assert lb.breach_duration_seconds == 120


def test_recommendation_priority_validation_and_roundtrip():
    # 合法priority
    rec = Recommendation(
        type=RecommendationType.HEDGE,
        priority=3,
        description='建议进行对冲',
        action_items=['hedge with futures'],
        estimated_impact=0.1,
    )
    d = rec.to_dict()
    rec2 = Recommendation.from_dict(d)
    assert rec2.type == RecommendationType.HEDGE
    assert rec2.priority == 3
    assert rec2.status in ('pending', 'approved', 'rejected', 'completed')
    assert isinstance(rec2.created_at, datetime)

    # 非法priority应抛出ValueError
    with pytest.raises(ValueError):
        Recommendation(
            type=RecommendationType.MONITOR,
            priority=11,
            description='非法优先级',
            action_items=[],
        )


def test_time_horizon_properties():
    assert TimeHorizon.DAILY.display_name == '每日'
    assert TimeHorizon.WEEKLY.display_name == '每周'
    assert TimeHorizon.MONTHLY.display_name == '每月'
    assert TimeHorizon.YEARLY.display_name == '每年'

    assert TimeHorizon.DAILY.timedelta == timedelta(days=1)
    assert TimeHorizon.WEEKLY.timedelta == timedelta(weeks=1)
    assert TimeHorizon.MONTHLY.timedelta == timedelta(days=30)
    assert TimeHorizon.YEARLY.timedelta == timedelta(days=365)

if __name__ == '__main__':
    unittest.main()
