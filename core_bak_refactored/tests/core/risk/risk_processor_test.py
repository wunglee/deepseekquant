import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest
import numpy as np
import pandas as pd
from typing import Dict

from core.risk.risk_processor import RiskProcessor


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


class RiskProcessorTest(unittest.TestCase):
    def setUp(self):
        # 配置包含压力测试场景（即使失败也回退值）
        self.config = {
            'trading_days_per_year': 252,
            'default_confidence_level': 0.95,
            'risk_free_rate': 0.03,
            'stress_test_scenarios': [
                {
                    'scenario_id': 'crash_1',
                    'name': 'market_crash',
                    'description': 'simulated crash',
                    'parameters': {'type': 'market_crash'},
                    'probability': 0.1,
                    'impact_level': {'value': 'high'},
                    'duration': '1d',
                    'triggers': [],
                    'mitigation_strategies': []
                }
            ],
            'portfolio_id': 'test_portfolio'
        }
        self.processor = RiskProcessor(self.config)
        self.portfolio_state = DummyPortfolioState({
            'A': DummyAlloc(0.5),
            'B': DummyAlloc(0.5)
        })
        # 构造收益序列
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0, 0.02, 300))
        self.data = {
            'returns': returns.values.tolist(),
            'portfolio_state': self.portfolio_state,
            'market_data': {}
        }

    def test_process_outputs_include_var95_and_scenario_analysis(self):
        result = self.processor.process(self.data)
        self.assertTrue(result['success'])
        assessment = result['assessment']
        # 验证VaR/ES键名映射
        self.assertIn('value_at_risk', assessment.__dict__)
        self.assertIn('expected_shortfall', assessment.__dict__)
        # 验证情景分析包含字典
        self.assertIsInstance(assessment.scenario_analysis, dict)
        # 验证风险贡献与集中度（风险贡献可能为空，但字段应存在）
        self.assertIsInstance(assessment.risk_contributions, dict)


if __name__ == '__main__':
    unittest.main(verbosity=2)
