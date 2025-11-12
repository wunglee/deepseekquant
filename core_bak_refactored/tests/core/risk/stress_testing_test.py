import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest

from core.risk.stress_testing import StressTester
from core.risk.risk_models import StressTestScenario, RiskLevel


class DummyAlloc:
    def __init__(self, weight: float):
        self.weight = weight


class DummyPortfolioState:
    def __init__(self, allocations):
        self.allocations = allocations


class StressTesterTest(unittest.TestCase):
    """测试压力测试器"""

    def setUp(self):
        self.config = {
            'trading_days_per_year': 252,
            'stress_test_scenarios': [
                {
                    'scenario_id': 'crash_test',
                    'name': 'Market Crash',
                    'description': 'Severe market crash',
                    'parameters': {'type': 'market_crash', 'crash_magnitude': -0.30},
                    'probability': 0.05,
                    'impact_level': RiskLevel.EXTREME,
                    'duration': '1d',
                    'triggers': [],
                    'mitigation_strategies': []
                }
            ]
        }
        self.tester = StressTester(self.config)
        self.portfolio_state = DummyPortfolioState({
            'A': DummyAlloc(0.6),
            'B': DummyAlloc(0.4)
        })
        self.market_data = {}

    def test_initialize_scenarios_from_config(self):
        """场景初始化：从配置加载"""
        self.assertIn('crash_test', self.tester.scenarios)
        scenario = self.tester.scenarios['crash_test']
        self.assertEqual(scenario.name, 'Market Crash')

    def test_run_stress_tests(self):
        """运行压力测试：应返回结果字典"""
        results = self.tester.run_stress_tests(self.portfolio_state, self.market_data)
        self.assertIsInstance(results, dict)
        self.assertIn('crash_test', results)
        # 市场崩盘应为负值
        self.assertLess(results['crash_test'], 0)

    def test_simulate_market_crash(self):
        """市场崩盘场景：30%下跌"""
        scenario = StressTestScenario(
            scenario_id='crash',
            name='crash',
            description='crash',
            parameters={'type': 'market_crash', 'crash_magnitude': -0.30},
            probability=0.05,
            impact_level=RiskLevel.EXTREME,
            duration='1d',
            triggers=[],
            mitigation_strategies=[]
        )
        result = self.tester._simulate_market_crash(scenario, self.portfolio_state, self.market_data)
        # 总敞口1.0 * -0.30 = -0.30
        self.assertAlmostEqual(result, -0.30, places=2)

    def test_simulate_liquidity_crisis(self):
        """流动性危机场景"""
        scenario = StressTestScenario(
            scenario_id='liquidity',
            name='liquidity',
            description='liquidity crisis',
            parameters={'type': 'liquidity_crisis', 'liquidity_cost_multiplier': 3.0},
            probability=0.1,
            impact_level=RiskLevel.HIGH,
            duration='3d',
            triggers=[],
            mitigation_strategies=[]
        )
        result = self.tester._simulate_liquidity_crisis(scenario, self.portfolio_state, self.market_data)
        # 应为负值（流动性成本）
        self.assertLess(result, 0)

    def test_simulate_interest_rate_shock(self):
        """利率冲击场景"""
        scenario = StressTestScenario(
            scenario_id='rate',
            name='rate shock',
            description='rate shock',
            parameters={'type': 'interest_rate_shock', 'rate_shock_bps': 200, 'portfolio_duration': 3.0},
            probability=0.2,
            impact_level=RiskLevel.MODERATE,
            duration='1d',
            triggers=[],
            mitigation_strategies=[]
        )
        result = self.tester._simulate_interest_rate_shock(scenario, self.portfolio_state, self.market_data)
        # 利率上升导致损失
        self.assertLess(result, 0)

    def test_simulate_correlation_breakdown(self):
        """相关性崩溃场景"""
        scenario = StressTestScenario(
            scenario_id='corr',
            name='correlation breakdown',
            description='correlation breakdown',
            parameters={'type': 'correlation_breakdown', 'correlation_increase': 0.8},
            probability=0.1,
            impact_level=RiskLevel.HIGH,
            duration='1w',
            triggers=[],
            mitigation_strategies=[]
        )
        result = self.tester._simulate_correlation_breakdown(scenario, self.portfolio_state, self.market_data)
        # 多元化失效，风险增加
        self.assertLess(result, 0)

    def test_run_scenario_analysis(self):
        """情景分析：应返回多种场景"""
        results = self.tester.run_scenario_analysis(self.portfolio_state, self.market_data)
        self.assertIsInstance(results, dict)
        # 应包含标准情景
        self.assertIn('recession_mild', results)
        self.assertIn('recession_severe', results)
        self.assertIn('market_rally', results)

    def test_simulate_market_downturn(self):
        """市场下行场景"""
        params = {'growth_shock': -0.02, 'volatility_shock': 0.5}
        result = self.tester._simulate_market_downturn(params, self.portfolio_state, self.market_data)
        # 应包含直接损失与波动率冲击
        self.assertLess(result, 0)

    def test_simulate_volatility_spike(self):
        """波动率飙升场景"""
        params = {'volatility_shock': 2.0}
        result = self.tester._simulate_volatility_spike(params, self.portfolio_state, self.market_data)
        # 波动率增加导致风险增加
        self.assertLess(result, 0)
    
    def test_builtin_scenarios_loaded(self):
        """
        P1增强测试：内置场景库加载
        验证5个内置场景（专家answer.md线108-141）
        """
        # 应该加载5个内置 + 1个自定义 = 6个
        self.assertGreaterEqual(len(self.tester.scenarios), 5)
        
        # 验证全球市场场景
        self.assertIn('2008_financial_crisis', self.tester.scenarios)
        self.assertIn('covid_19_pandemic', self.tester.scenarios)
        
        # 验证A股特有场景
        self.assertIn('2015_china_market_crash', self.tester.scenarios)
        self.assertIn('circuit_breaker_2016', self.tester.scenarios)
        self.assertIn('thousand_stocks_limit_down', self.tester.scenarios)
    
    def test_builtin_scenario_parameters(self):
        """
        P1增强测试：内置场景参数正确性
        验证专家提供的关键参数
        """
        # 2008金融危机：decline=-0.40, volatility_spike=3.5
        crisis_2008 = self.tester.scenarios['2008_financial_crisis']
        self.assertEqual(crisis_2008.parameters['decline'], -0.40)
        self.assertEqual(crisis_2008.parameters['volatility_spike'], 3.5)
        self.assertEqual(crisis_2008.parameters['correlation_break'], 0.8)
        
        # 2015A股大跌：decline=-0.30, liquidity_dry_up=0.8
        crash_2015 = self.tester.scenarios['2015_china_market_crash']
        self.assertEqual(crash_2015.parameters['decline'], -0.30)
        self.assertEqual(crash_2015.parameters['liquidity_dry_up'], 0.8)
        self.assertEqual(crash_2015.parameters['limit_hit_frequency'], 0.3)
    
    def test_builtin_scenarios_runnable(self):
        """
        P1增强测试：内置场景可执行性
        确保所有内置场景能正常运行
        """
        results = self.tester.run_stress_tests(self.portfolio_state, self.market_data)
        
        # 所有内置场景应该都有结果
        self.assertIn('2008_financial_crisis', results)
        self.assertIn('2015_china_market_crash', results)
        self.assertIn('covid_19_pandemic', results)
        self.assertIn('circuit_breaker_2016', results)
        self.assertIn('thousand_stocks_limit_down', results)
        
        # 所有场景结果应为负值（损失）
        for scenario_id, result in results.items():
            self.assertLessEqual(result, 0, f"{scenario_id}应该产生损失")


if __name__ == '__main__':
    unittest.main(verbosity=2)
