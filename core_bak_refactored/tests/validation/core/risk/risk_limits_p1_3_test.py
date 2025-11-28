"""
风险限额管理P1-3增强功能测试
测试智能阈值、组合优化、违规优先级、市场差异化限额
"""

import unittest
import sys
import os
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core_bak_refactored.core.risk.risk_limits_enhanced import (
    SmartThresholdChecker, ThresholdTier, ThresholdBreach,
    PortfolioOptimizationAdvisor, BreachPrioritizer,
    MarketSpecificLimitsChecker, MARKET_SPECIFIC_LIMITS
)


@dataclass
class MockAllocation:
    """模拟资产配置"""
    weight: float
    sector: str = 'Technology'
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class MockPortfolioState:
    """模拟组合状态"""
    allocations: dict
    total_value: float = 1000000
    leveraged_value: float = 1000000
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TestSmartThresholdChecker(unittest.TestCase):
    """测试智能阈值检查器"""
    
    def setUp(self):
        self.checker = SmartThresholdChecker()
    
    def test_green_zone_threshold(self):
        """测试绿色区域阈值（0.9）"""
        breach = self.checker.check_smart_threshold(
            metric_name='VaR',
            current_value=0.095,
            base_threshold=0.10
        )
        
        self.assertIsNotNone(breach)
        self.assertEqual(breach.tier, ThresholdTier.GREEN)
        self.assertAlmostEqual(breach.utilization_ratio, 0.95, places=2)
        self.assertTrue(10 <= breach.severity_score <= 30)
        self.assertEqual(breach.alert_level, 'info')
        self.assertIn('接近阈值', breach.recommended_actions[0])
    
    def test_yellow_zone_threshold(self):
        """测试黄色区域阈值（1.0）"""
        breach = self.checker.check_smart_threshold(
            metric_name='VaR',
            current_value=0.105,
            base_threshold=0.10
        )
        
        self.assertIsNotNone(breach)
        self.assertEqual(breach.tier, ThresholdTier.YELLOW)
        self.assertAlmostEqual(breach.utilization_ratio, 1.05, places=2)
        self.assertTrue(30 <= breach.severity_score <= 60)
        self.assertEqual(breach.alert_level, 'info')
        self.assertIn('触及正常限额', breach.recommended_actions[0])
    
    def test_orange_zone_threshold(self):
        """测试橙色区域阈值（1.2）"""
        breach = self.checker.check_smart_threshold(
            metric_name='VaR',
            current_value=0.125,
            base_threshold=0.10
        )
        
        self.assertIsNotNone(breach)
        self.assertEqual(breach.tier, ThresholdTier.ORANGE)
        self.assertAlmostEqual(breach.utilization_ratio, 1.25, places=2)
        self.assertTrue(60 <= breach.severity_score <= 85)
        self.assertEqual(breach.alert_level, 'warning')
        self.assertIn('超出正常限额', breach.recommended_actions[0])
    
    def test_red_zone_threshold(self):
        """测试红色区域阈值（1.5）"""
        breach = self.checker.check_smart_threshold(
            metric_name='VaR',
            current_value=0.160,
            base_threshold=0.10
        )
        
        self.assertIsNotNone(breach)
        self.assertEqual(breach.tier, ThresholdTier.RED)
        self.assertAlmostEqual(breach.utilization_ratio, 1.60, places=2)
        self.assertTrue(85 <= breach.severity_score <= 100)
        self.assertEqual(breach.alert_level, 'critical')
        self.assertIn('严重超限', breach.recommended_actions[0])
    
    def test_below_green_threshold(self):
        """测试低于绿色阈值（无警报）"""
        breach = self.checker.check_smart_threshold(
            metric_name='VaR',
            current_value=0.08,
            base_threshold=0.10
        )
        
        self.assertIsNone(breach)
    
    def test_breach_history_recorded(self):
        """测试违规历史记录"""
        initial_count = len(self.checker.breach_history)
        
        self.checker.check_smart_threshold('VaR', 0.10, 0.10)
        self.assertEqual(len(self.checker.breach_history), initial_count + 1)
        
        self.checker.check_smart_threshold('ES', 0.15, 0.10)
        self.assertEqual(len(self.checker.breach_history), initial_count + 2)


class TestPortfolioOptimizationAdvisor(unittest.TestCase):
    """测试投资组合优化顾问"""
    
    def setUp(self):
        self.config = {
            'risk_free_rate': 0.03,
            'target_sharpe_ratio': 1.0,
            'target_risk_return_ratio': 2.0,
            'market_return': 0.08,
            'market_volatility': 0.18
        }
        self.advisor = PortfolioOptimizationAdvisor(self.config)
        
        # 模拟组合状态
        self.portfolio_state = MockPortfolioState(
            allocations={
                'AAPL': MockAllocation(0.15, 'Technology'),
                'MSFT': MockAllocation(0.15, 'Technology'),
                'JPM': MockAllocation(0.10, 'Finance'),
                'JNJ': MockAllocation(0.10, 'Healthcare'),
                'XOM': MockAllocation(0.10, 'Energy')
            }
        )
    
    def test_sharpe_ratio_optimization(self):
        """测试夏普比率优化建议"""
        risk_metrics = {
            'expected_return': 0.06,
            'volatility': 0.12
        }
        
        recommendations = self.advisor._optimize_sharpe_ratio(risk_metrics)
        
        self.assertGreater(len(recommendations), 0)
        rec = recommendations[0]
        self.assertEqual(rec['type'], 'sharpe_optimization')
        self.assertIn('current_sharpe', rec)
        self.assertIn('target_sharpe', rec)
        self.assertIn('actions', rec)
        self.assertGreater(len(rec['actions']), 0)
    
    def test_minimum_variance_optimization(self):
        """测试最小方差组合优化"""
        risk_metrics = {
            'volatility': 0.20,
            'correlation_risk': 0.05
        }
        
        recommendations = self.advisor._optimize_minimum_variance(
            self.portfolio_state, risk_metrics
        )
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            self.assertEqual(rec['type'], 'minimum_variance')
            self.assertIn('current_volatility', rec)
            self.assertIn('theoretical_min', rec)
            self.assertIn('excess_volatility', rec)
    
    def test_efficient_frontier_analysis(self):
        """测试有效前沿分析"""
        risk_metrics = {
            'expected_return': 0.04,  # 低收益
            'volatility': 0.15
        }
        
        recommendations = self.advisor._analyze_efficient_frontier(risk_metrics)
        
        self.assertGreater(len(recommendations), 0)
        rec = recommendations[0]
        self.assertEqual(rec['type'], 'efficient_frontier')
        self.assertIn('current_position', rec)
        self.assertIn('efficient_position', rec)
        self.assertIn('return_gap', rec)
    
    def test_risk_return_balance(self):
        """测试风险收益均衡优化"""
        risk_metrics = {
            'var_95': -0.08,
            'expected_return': 0.10
        }
        
        recommendations = self.advisor._optimize_risk_return_balance(risk_metrics)
        
        self.assertGreater(len(recommendations), 0)
        rec = recommendations[0]
        self.assertEqual(rec['type'], 'risk_return_balance')
        self.assertIn('current_ratio', rec)
        self.assertIn('target_ratio', rec)
    
    def test_generate_all_recommendations(self):
        """测试生成所有优化建议"""
        risk_metrics = {
            'expected_return': 0.05,
            'volatility': 0.18,
            'var_95': -0.09
        }
        
        recommendations = self.advisor.generate_recommendations(
            self.portfolio_state, risk_metrics
        )
        
        self.assertIsInstance(recommendations, list)
        # 应该至少有2-3个建议
        self.assertGreaterEqual(len(recommendations), 2)


class TestBreachPrioritizer(unittest.TestCase):
    """测试违规优先级处理器"""
    
    def setUp(self):
        self.prioritizer = BreachPrioritizer()
    
    def test_single_breach_priority(self):
        """测试单个违规优先级计算"""
        breach = {
            'limit_type': 'value_at_risk',
            'severity': 'high',
            'threshold': 0.05,
            'current_value': 0.08,
            'time_horizon': 'immediate'
        }
        
        score = self.prioritizer._calculate_breach_priority(breach)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        # 高严重性 + immediate时间 应该有较高分数
        self.assertGreater(score, 60)
    
    def test_multiple_breaches_prioritization(self):
        """测试多个违规排序"""
        breaches = [
            {
                'limit_type': 'concentration',
                'severity': 'medium',
                'threshold': 0.30,
                'current_value': 0.35,
                'time_horizon': '1d'
            },
            {
                'limit_type': 'leverage_ratio',
                'severity': 'critical',
                'threshold': 2.0,
                'current_value': 2.8,
                'time_horizon': 'immediate'
            },
            {
                'limit_type': 'liquidity_risk',
                'severity': 'high',
                'threshold': 0.20,
                'current_value': 0.28,
                'time_horizon': '4h'
            }
        ]
        
        prioritized = self.prioritizer.prioritize_breaches(breaches)
        
        self.assertEqual(len(prioritized), 3)
        # 第一个应该是杠杆违规（critical + immediate）
        self.assertEqual(prioritized[0]['limit_type'], 'leverage_ratio')
        self.assertEqual(prioritized[0]['处理顺序'], 1)
        # 确保按priority_score降序排列
        for i in range(len(prioritized) - 1):
            self.assertGreaterEqual(
                prioritized[i]['priority_score'],
                prioritized[i+1]['priority_score']
            )
    
    def test_cascading_impact_analysis(self):
        """测试级联影响分析"""
        main_breach = {
            'limit_type': 'leverage_ratio',
            'severity': 'critical'
        }
        
        all_breaches = [
            main_breach,
            {'limit_type': 'concentration', 'severity': 'medium'},
            {'limit_type': 'liquidity', 'severity': 'high'}
        ]
        
        impact = self.prioritizer._analyze_cascading_impact(main_breach, all_breaches)
        
        self.assertIn('affected_limits', impact)
        self.assertIn('impact_score', impact)
        self.assertIn('chain_reaction', impact)
        # 杠杆违规应该影响集中度和流动性
        self.assertGreater(len(impact['affected_limits']), 0)
    
    def test_priority_level_determination(self):
        """测试优先级别判定"""
        self.assertEqual(self.prioritizer._determine_priority_level(90), 'P0-紧急')
        self.assertEqual(self.prioritizer._determine_priority_level(70), 'P1-高优先级')
        self.assertEqual(self.prioritizer._determine_priority_level(50), 'P2-中优先级')
        self.assertEqual(self.prioritizer._determine_priority_level(30), 'P3-低优先级')


class TestMarketSpecificLimitsChecker(unittest.TestCase):
    """测试市场差异化限额检查器"""
    
    def test_cn_market_checker_initialization(self):
        """测试A股市场检查器初始化"""
        checker = MarketSpecificLimitsChecker('CN')
        
        self.assertEqual(checker.market_type, 'CN')
        self.assertEqual(checker.market_limits['regulatory_framework'], 'CSRC')
        self.assertEqual(checker.market_limits['leverage_max'], 1.0)
        self.assertEqual(checker.market_limits['single_stock_max_weight'], 0.10)
    
    def test_us_market_checker_initialization(self):
        """测试美股市场检查器初始化"""
        checker = MarketSpecificLimitsChecker('US')
        
        self.assertEqual(checker.market_type, 'US')
        self.assertEqual(checker.market_limits['regulatory_framework'], 'SEC/FINRA')
        self.assertEqual(checker.market_limits['leverage_max'], 4.0)
        self.assertEqual(checker.market_limits['single_stock_max_weight'], 0.15)
    
    def test_cn_single_stock_limit_breach(self):
        """测试A股单股限额违规"""
        checker = MarketSpecificLimitsChecker('CN')
        portfolio_state = MockPortfolioState(
            allocations={
                '600519': MockAllocation(0.15),  # 超过10%限制
                '000858': MockAllocation(0.08)
            }
        )
        
        breaches = checker._check_single_stock_limits(portfolio_state)
        
        self.assertEqual(len(breaches), 1)
        self.assertEqual(breaches[0]['symbol'], '600519')
        self.assertEqual(breaches[0]['threshold'], 0.10)
        self.assertIn('CN', breaches[0]['limit_type'])
    
    def test_cn_leverage_limit_breach(self):
        """测试A股杠杆限额违规"""
        checker = MarketSpecificLimitsChecker('CN')
        portfolio_state = MockPortfolioState(
            allocations={'600519': MockAllocation(0.10)},
            total_value=1000000,
            leveraged_value=1500000  # 1.5倍杠杆，超过1.0限制
        )
        
        breaches = checker._check_leverage_limits(portfolio_state)
        
        self.assertEqual(len(breaches), 1)
        self.assertEqual(breaches[0]['threshold'], 1.0)
        self.assertAlmostEqual(breaches[0]['current_value'], 1.5, places=1)
    
    def test_cn_st_stock_limit(self):
        """测试A股ST股票限额"""
        checker = MarketSpecificLimitsChecker('CN')
        portfolio_state = MockPortfolioState(
            allocations={
                'ST600001': MockAllocation(0.04),
                'ST600002': MockAllocation(0.03),  # 总计7%，超过5%限制
                '600519': MockAllocation(0.05)
            }
        )
        
        breaches = checker._check_cn_specific_rules(portfolio_state)
        
        self.assertEqual(len(breaches), 1)
        self.assertEqual(breaches[0]['limit_type'], 'CN_st_stock_limit')
        self.assertAlmostEqual(breaches[0]['current_value'], 0.07, places=2)
    
    def test_us_pdt_rule(self):
        """测试美股PDT规则"""
        checker = MarketSpecificLimitsChecker('US')
        portfolio_state = MockPortfolioState(
            allocations={'AAPL': MockAllocation(0.10)},
            total_value=20000,  # 低于25000
            metadata={'day_trades_count': 5}  # 日内交易超过4次
        )
        
        breaches = checker._check_us_specific_rules(portfolio_state)
        
        self.assertEqual(len(breaches), 1)
        self.assertEqual(breaches[0]['limit_type'], 'US_pdt_rule')
        self.assertEqual(breaches[0]['threshold'], 25000)
    
    def test_concentration_top10_limit(self):
        """测试前10大持仓集中度限额"""
        checker = MarketSpecificLimitsChecker('CN')
        
        # 创建15只股票，前10只总权重65%（超过60%限制）
        allocations = {}
        for i in range(15):
            weight = 0.065 if i < 10 else 0.035
            allocations[f'600{i:03d}'] = MockAllocation(weight)
        
        portfolio_state = MockPortfolioState(allocations=allocations)
        breaches = checker._check_concentration_limits(portfolio_state)
        
        self.assertEqual(len(breaches), 1)
        self.assertAlmostEqual(breaches[0]['current_value'], 0.65, places=2)
        self.assertEqual(breaches[0]['threshold'], 0.60)
    
    def test_market_limits_constants(self):
        """测试市场限额常量完整性"""
        for market in ['CN', 'US', 'HK']:
            self.assertIn(market, MARKET_SPECIFIC_LIMITS)
            limits = MARKET_SPECIFIC_LIMITS[market]
            
            # 验证必需字段
            self.assertIn('description', limits)
            self.assertIn('single_stock_max_weight', limits)
            self.assertIn('sector_max_weight', limits)
            self.assertIn('leverage_max', limits)
            self.assertIn('concentration_top10', limits)
            self.assertIn('regulatory_framework', limits)


class TestP1_3Integration(unittest.TestCase):
    """P1-3集成测试"""
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        # 1. 智能阈值检查
        threshold_checker = SmartThresholdChecker()
        breach = threshold_checker.check_smart_threshold(
            metric_name='VaR',
            current_value=0.12,
            base_threshold=0.10
        )
        self.assertIsNotNone(breach)
        self.assertEqual(breach.tier, ThresholdTier.ORANGE)
        
        # 2. 组合优化建议
        config = {
            'risk_free_rate': 0.03,
            'target_sharpe_ratio': 1.0,
            'target_risk_return_ratio': 2.0,
            'market_return': 0.08,
            'market_volatility': 0.18
        }
        advisor = PortfolioOptimizationAdvisor(config)
        portfolio_state = MockPortfolioState(
            allocations={
                'AAPL': MockAllocation(0.20),
                'MSFT': MockAllocation(0.15)
            }
        )
        risk_metrics = {'expected_return': 0.05, 'volatility': 0.18, 'var_95': -0.09}
        recommendations = advisor.generate_recommendations(portfolio_state, risk_metrics)
        self.assertGreater(len(recommendations), 0)
        
        # 3. 违规优先级排序
        breaches = [
            {'limit_type': 'leverage_ratio', 'severity': 'critical', 
             'threshold': 2.0, 'current_value': 3.0, 'time_horizon': 'immediate'},
            {'limit_type': 'concentration', 'severity': 'medium',
             'threshold': 0.30, 'current_value': 0.35, 'time_horizon': '1d'}
        ]
        prioritizer = BreachPrioritizer()
        prioritized = prioritizer.prioritize_breaches(breaches)
        self.assertEqual(len(prioritized), 2)
        self.assertEqual(prioritized[0]['处理顺序'], 1)
        
        # 4. 市场特定限额检查
        market_checker = MarketSpecificLimitsChecker('CN')
        market_breaches = market_checker.check_market_limits(portfolio_state)
        self.assertIsInstance(market_breaches, list)


if __name__ == '__main__':
    unittest.main(verbosity=2)
