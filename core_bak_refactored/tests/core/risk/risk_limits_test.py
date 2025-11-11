import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
from typing import Dict

from core.risk.risk_limits import RiskLimitsManager
from core.risk.risk_models import RiskLimit, RiskType, RiskMetric, RiskControlAction


class DummyAlloc:
    def __init__(self, weight: float, sector: str = 'general', metadata: Dict = None):
        self.weight = weight
        self.sector = sector
        self.metadata = metadata or {}


class DummyPortfolioState:
    def __init__(self, allocations):
        self.allocations = allocations
        self.total_value = 1000000.0
        self.leveraged_value = 1000000.0


class RiskLimitsManagerTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            'risk_limits': [
                {
                    'risk_type': {'value': 'market_risk'},
                    'metric': {'value': 'value_at_risk'},
                    'threshold': 0.001,
                    'time_horizon': '1d'
                },
                {
                    'risk_type': {'value': 'market_risk'},
                    'metric': {'value': 'expected_shortfall'},
                    'threshold': 0.0015,
                    'time_horizon': '1d'
                }
            ],
            'position_limits': {}
        }
        self.manager = RiskLimitsManager(self.config)
        # 直接设置风险限额，避免枚举反序列化问题
        self.manager.risk_limits = {
            'market_risk_value_at_risk': RiskLimit(
                risk_type=RiskType.MARKET_RISK,
                metric=RiskMetric.VALUE_AT_RISK,
                threshold=0.001,
                time_horizon='1d'
            ),
            'market_risk_expected_shortfall': RiskLimit(
                risk_type=RiskType.MARKET_RISK,
                metric=RiskMetric.EXPECTED_SHORTFALL,
                threshold=0.0015,
                time_horizon='1d'
            )
        }
        # 放宽行业限额，避免行业集中度误报
        self.manager.config['sector_limits'] = {'general': 1.0}
        self.portfolio_state = DummyPortfolioState({
            'A': DummyAlloc(0.6),
            'B': DummyAlloc(0.4)
        })

    def test_market_risk_limits_breach_on_var_and_es(self):
        # 构造风险指标：使用 var_95 / cvar_95 键名
        risk_metrics = {
            'var_95': 0.01,
            'cvar_95': 0.02
        }
        breaches = self.manager.check_limits(self.portfolio_state, risk_metrics)
        # 应至少包含VaR和ES两类违规
        types = [b['limit_type'] for b in breaches]
        self.assertIn('value_at_risk', types)
        self.assertIn('expected_shortfall', types)
        # 验证metric键名映射
        metrics = {b['limit_type']: b['metric'] for b in breaches}
        self.assertEqual(metrics['value_at_risk'], 'var_95')
        self.assertEqual(metrics['expected_shortfall'], 'cvar_95')

    def test_empty_portfolio_no_limit_breach(self):
        """空仓组合不应触发限额违规"""
        empty_state = DummyPortfolioState({})
        risk_metrics = {'var_95': 0.0, 'cvar_95': 0.0}
        breaches = self.manager.check_limits(empty_state, risk_metrics)
        # 空仓没有风险，不应违规
        self.assertEqual(len(breaches), 0)

    def test_risk_limit_from_dict_with_enum_objects(self):
        """枚举对象形式的配置加载"""
        data = {
            'risk_type': RiskType.MARKET_RISK,
            'metric': RiskMetric.VALUE_AT_RISK,
            'threshold': 0.05,
            'action': RiskControlAction.WARN
        }
        limit = RiskLimit.from_dict(data)
        self.assertEqual(limit.risk_type, RiskType.MARKET_RISK)
        self.assertEqual(limit.metric, RiskMetric.VALUE_AT_RISK)

    def test_risk_limit_from_dict_with_string_enums(self):
        """字符串形式的枚举配置加载"""
        data = {
            'risk_type': 'market_risk',
            'metric': 'value_at_risk',
            'threshold': 0.05,
            'action': 'warn'
        }
        limit = RiskLimit.from_dict(data)
        self.assertEqual(limit.risk_type, RiskType.MARKET_RISK)
        self.assertEqual(limit.metric, RiskMetric.VALUE_AT_RISK)
        self.assertEqual(limit.action, RiskControlAction.WARN)

    def test_risk_limit_from_dict_with_dict_enums(self):
        """字典形式的枚举配置加载（兼容旧配置）"""
        data = {
            'risk_type': {'value': 'market_risk'},
            'metric': {'value': 'expected_shortfall'},
            'threshold': 0.08,
            'action': {'value': 'reject'}
        }
        limit = RiskLimit.from_dict(data)
        self.assertEqual(limit.risk_type, RiskType.MARKET_RISK)
        self.assertEqual(limit.metric, RiskMetric.EXPECTED_SHORTFALL)
        self.assertEqual(limit.action, RiskControlAction.REJECT)


if __name__ == '__main__':
    unittest.main(verbosity=2)
