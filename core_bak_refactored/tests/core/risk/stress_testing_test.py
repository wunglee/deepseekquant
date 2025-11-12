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
    
    # =========================================================================
    # P1-2增强测试：完整场景参数使用 & 组合场景测试
    # =========================================================================
    
    def test_market_crash_with_all_parameters(self):
        """
        P1-2增强测试：市场崩盘场景完整使用所有参数
        验证decline/volatility_spike/correlation_break/recovery_period参数生效
        """
        # 运裌2008金融危机场景（包含所有参数）
        results = self.tester.run_stress_tests(self.portfolio_state, self.market_data)
        
        # 2008危机应该比COVID损失更大（decline: -0.40 vs -0.20）
        self.assertLess(results['2008_financial_crisis'], results['covid_19_pandemic'],
                        "2008危机损失应大于COVID")
        
        # 波动率冲击应该增加损失（volatility_spike=3.5）
        self.assertLess(results['2008_financial_crisis'], -0.35,
                        "波动率冲击应该增加损失至超35%以上")
    
    def test_liquidity_crisis_with_china_specific_parameters(self):
        """
        P1-2增强测试：流动性危机A股特有参数处理
        验证liquidity_dry_up/limit_hit_frequency/margin_call_cascade参数生效
        """
        # 运衁2015A股大跌场景（包含liquidity_dry_up和limit_hit_frequency）
        scenario_2015 = self.tester.scenarios['2015_china_market_crash']
        result_2015 = self.tester._simulate_market_crash(scenario_2015, self.portfolio_state, self.market_data)
        
        # 运行千股跌停场景（包含margin_call_cascade）
        scenario_limit_down = self.tester.scenarios['thousand_stocks_limit_down']
        result_limit_down = self.tester._simulate_liquidity_crisis(
            scenario_limit_down, self.portfolio_state, self.market_data
        )
        
        # 流动性危机应产生显著损失
        self.assertLess(result_limit_down, -0.1, "流动性危机应产生至少10%损失")
    
    def test_combined_stress_tests_sequential(self):
        """
        P1-2增强测试：顺序冲击测试（危机传导）
        验证多个场景顺序发生时的传导效应
        """
        # 配置顺序场景
        self.config['stress_testing'] = {
            'enable_sequential_test': True,
            'sequential_scenarios': [['2008_financial_crisis', '2015_china_market_crash']],
            'propagation_factor': 0.3
        }
        self.tester = StressTester(self.config)
        
        # 运行组合测试
        results = self.tester.run_combined_stress_tests(self.portfolio_state, self.market_data)
        
        # 应该返回顺序测试结果
        self.assertIn('sequential', results)
        self.assertTrue(len(results['sequential']) > 0, "应该有顺序测试结果")
        
        # 顺序损失应该大于单个场景（因为有传导效应）
        sequential_loss = list(results['sequential'].values())[0]
        self.assertLess(sequential_loss, -0.5, "顺序冲击损失应超过50%")
    
    def test_combined_stress_tests_concurrent(self):
        """
        P1-2增强测试：并发冲击测试（系统性风险）
        验证多个场景同时发生时的相关性影响
        """
        # 配置并发场景
        self.config['stress_testing'] = {
            'enable_concurrent_test': True,
            'concurrent_scenarios': [['2008_financial_crisis', '2015_china_market_crash']],
            'systemic_premium': 0.2
        }
        self.tester = StressTester(self.config)
        
        # 运行组合测试
        results = self.tester.run_combined_stress_tests(self.portfolio_state, self.market_data)
        
        # 应该返回并发测试结果
        self.assertIn('concurrent', results)
        self.assertTrue(len(results['concurrent']) > 0, "应该有并发测试结果")
        
        # 并发损失应该显著（考虑相关性和系统性溢价）
        concurrent_loss = list(results['concurrent'].values())[0]
        self.assertLess(concurrent_loss, -0.4, "并发冲击损失应显著")
    
    def test_combined_stress_tests_feedback_loop(self):
        """
        P1-2增强测试：反馈循环测试（风险叠加）
        验证损失导致的反馈效应（恐慌性抛售）
        """
        # 配置反馈循环
        self.config['stress_testing'] = {
            'enable_feedback_loop_test': True,
            'feedback_loop_scenarios': ['2008_financial_crisis'],
            'feedback_factor': 0.25,
            'max_feedback_iterations': 5
        }
        self.tester = StressTester(self.config)
        
        # 运行组合测试
        results = self.tester.run_combined_stress_tests(self.portfolio_state, self.market_data)
        
        # 应该返回反馈循环结果
        self.assertIn('feedback_loop', results)
        self.assertIn('2008_financial_crisis', results['feedback_loop'])
        
        # 反馈循环损失应该大于单次冲击（因为有反馈效应）
        feedback_loss = results['feedback_loop']['2008_financial_crisis']
        single_loss = self.tester.run_stress_tests(self.portfolio_state, self.market_data)['2008_financial_crisis']
        self.assertLess(feedback_loss, single_loss * 0.95, "反馈循环应该放大损失")
    
    def test_auxiliary_methods(self):
        """
        P1-2增强测试：辅助方法功能
        验证_get_daily_volume和_get_leveraged_position方法
        """
        # 测试日成交量获取
        volume = self.tester._get_daily_volume('600000.SH', self.market_data)
        self.assertGreater(volume, 0, "日成交量应该大于0")
        
        # 测试杠杆仓位获取
        leveraged = self.tester._get_leveraged_position(self.portfolio_state)
        self.assertGreaterEqual(leveraged, 0, "杠杆仓位应该非负")


if __name__ == '__main__':
    unittest.main(verbosity=2)
