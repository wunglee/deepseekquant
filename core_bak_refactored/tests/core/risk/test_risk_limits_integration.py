"""
风险限额模块集成测试

测试RiskLimitsEnhanced的集成功能
"""

import unittest
from unittest.mock import Mock, MagicMock
from core_bak_refactored.core.risk.risk_limits_enhanced import (
    EnhancedRiskLimitsManager,
    EnhancedLimitsConfig,
    SmartThresholdChecker,
    BreachPrioritizer,
    MarketSpecificLimitsChecker,
    MARKET_SPECIFIC_LIMITS
)


class TestRiskLimitsIntegration(unittest.TestCase):
    """风险限额集成测试"""
    
    def setUp(self):
        """创建测试环境"""
        # 创建模拟的基础RiskLimitsManager
        self.base_manager = Mock()
        self.base_manager.limits = {
            'var_limit': 0.05,
            'max_drawdown': 0.20,
            'volatility_limit': 0.30,
            'leverage_limit': 2.0,
            'single_position_limit': 0.10
        }
        self.base_manager.check_all_limits = Mock(return_value=[])
        
        # 创建增强管理器
        self.config = EnhancedLimitsConfig(
            enable_smart_threshold=True,
            enable_breach_prioritizer=True,
            enable_market_specific=True,
            default_market='CN'
        )
        self.enhanced_manager = EnhancedRiskLimitsManager(
            self.base_manager,
            self.config
        )
    
    def test_base_manager_functionality(self):
        """测试基础管理器功能正常"""
        self.assertIsNotNone(self.base_manager)
        self.assertEqual(self.base_manager.limits['var_limit'], 0.05)
    
    def test_enhanced_manager_initialization(self):
        """测试增强管理器成功初始化"""
        self.assertIsNotNone(self.enhanced_manager)
        self.assertIsNotNone(self.enhanced_manager.base_manager)
        self.assertEqual(self.enhanced_manager.base_manager, self.base_manager)
    
    def test_feature_components_loaded(self):
        """测试所有特性组件已加载"""
        status = self.enhanced_manager.get_feature_status()
        
        self.assertTrue(status['smart_threshold'], "智能阈值应已启用")
        self.assertTrue(status['breach_prioritizer'], "违规优先级应已启用")
        self.assertTrue(status['market_specific'], "市场限额应已启用")
    
    def test_integrated_limit_check(self):
        """测试集成限额检查"""
        # 创建模拟的投资组合状态
        portfolio_state = Mock()
        portfolio_state.total_value = 1000000
        portfolio_state.leveraged_value = 1500000
        portfolio_state.allocations = {}
        
        # 创建模拟的风险指标
        risk_metrics = {
            'var_95': 0.06,  # 超过限额(0.05)
            'max_drawdown': 0.18,
            'volatility': 0.25
        }
        
        # 执行检查
        result = self.enhanced_manager.check_all_limits(portfolio_state, risk_metrics)
        
        # 验证返回结构
        self.assertIn('base_breaches', result)
        self.assertIn('enhanced_breaches', result)
        self.assertIn('prioritized_breaches', result)
        self.assertIn('market_specific_issues', result)
    
    def test_graceful_degradation(self):
        """测试优雅降级"""
        # 禁用所有增强功能
        degraded_config = EnhancedLimitsConfig(
            enable_smart_threshold=False,
            enable_breach_prioritizer=False,
            enable_market_specific=False
        )
        degraded_manager = EnhancedRiskLimitsManager(
            self.base_manager,
            degraded_config
        )
        
        # 模拟数据
        portfolio_state = Mock()
        portfolio_state.total_value = 1000000
        risk_metrics = {'var_95': 0.04}
        
        # 应仍能正常工作（使用基础功能）
        result = degraded_manager.check_all_limits(portfolio_state, risk_metrics)
        self.assertIsNotNone(result)
        self.assertIn('base_breaches', result)
    
    def test_dynamic_config_update(self):
        """测试运行时配置更新"""
        # 初始状态：智能阈值已启用
        self.assertTrue(self.enhanced_manager.config.enable_smart_threshold)
        self.assertIsNotNone(self.enhanced_manager.smart_threshold)
        
        # 动态禁用（配置已更新，但组件不会被删除）
        self.enhanced_manager.update_config(enable_smart_threshold=False)
        
        # 验证配置已更新
        self.assertFalse(self.enhanced_manager.config.enable_smart_threshold)
        # 组件仍然存在（不会被删除，仅不再使用）
        # 这是合理的设计，避免频繁创建/销毁对象
        
        # 重新启用
        self.enhanced_manager.update_config(enable_smart_threshold=True)
        
        # 验证已重新启用
        self.assertTrue(self.enhanced_manager.config.enable_smart_threshold)
        self.assertIsNotNone(self.enhanced_manager.smart_threshold)
    
    def test_market_switching(self):
        """测试市场切换"""
        # 初始CN市场
        initial_market = self.enhanced_manager.market_checker.market_type
        self.assertEqual(initial_market, 'CN')
        
        # 切换到US市场
        self.enhanced_manager.update_config(default_market='US')
        
        # 验证切换成功
        new_market = self.enhanced_manager.market_checker.market_type
        self.assertEqual(new_market, 'US')
    
    def test_end_to_end_violation_handling(self):
        """测试端到端违规处理流程"""
        # 模拟投资组合状态（多个违规）
        portfolio_state = Mock()
        portfolio_state.total_value = 1000000
        portfolio_state.leveraged_value = 2500000  # 杠杆超标
        portfolio_state.allocations = {}
        portfolio_state.daily_turnover = 0.20  # CN市场限额0.15
        
        # 风险指标（VaR超标）
        risk_metrics = {
            'var_95': 0.07,  # 超过0.05限额
            'max_drawdown': 0.15,
            'volatility': 0.28
        }
        
        # 执行检查
        result = self.enhanced_manager.check_all_limits(portfolio_state, risk_metrics)
        
        # 应该检测到违规
        prioritized = result.get('prioritized_breaches', [])
        self.assertGreater(len(prioritized), 0, "应检测到违规")
        
        # 验证优先级排序（第一个应是最高优先级）
        if len(prioritized) >= 2:
            self.assertGreaterEqual(
                prioritized[0]['priority_score'],
                prioritized[1]['priority_score'],
                "违规应按优先级降序排列"
            )


class TestComponentInteraction(unittest.TestCase):
    """组件交互测试"""
    
    def test_smart_threshold_with_prioritizer(self):
        """测试智能阈值与优先级处理器的交互"""
        # 创建智能阈值检查器
        threshold_checker = SmartThresholdChecker()
        
        # 检查多个违规
        breach1 = threshold_checker.check_smart_threshold('var_95', 0.06, 0.05)
        breach2 = threshold_checker.check_smart_threshold('max_drawdown', 0.25, 0.20)
        
        # 转换为字典
        breaches_dict = [
            {
                **breach1.__dict__,
                'limit_type': 'value_at_risk',
                'time_horizon': '1d'
            },
            {
                **breach2.__dict__,
                'limit_type': 'max_drawdown',
                'time_horizon': 'immediate'
            }
        ]
        
        # 使用优先级处理器
        prioritizer = BreachPrioritizer()
        prioritized = prioritizer.prioritize_breaches(breaches_dict)
        
        # 验证交互结果
        self.assertEqual(len(prioritized), 2)
        # immediate应优先于1d
        self.assertEqual(prioritized[0]['time_horizon'], 'immediate')
    
    def test_market_limits_with_base_limits(self):
        """测试市场限额与基础限额的协同"""
        # CN市场检查器
        cn_checker = MarketSpecificLimitsChecker('CN')
        
        # 验证市场限额不冲突
        cn_single_stock = MARKET_SPECIFIC_LIMITS['CN']['single_stock_max_weight']
        self.assertEqual(cn_single_stock, 0.10)
        
        # US市场检查器
        us_checker = MarketSpecificLimitsChecker('US')
        us_single_stock = MARKET_SPECIFIC_LIMITS['US']['single_stock_max_weight']
        self.assertEqual(us_single_stock, 0.15)
        
        # 验证不同市场有不同限额
        self.assertNotEqual(cn_single_stock, us_single_stock)


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_large_portfolio_check_performance(self):
        """测试大规模投资组合检查性能"""
        import time
        
        # 创建管理器
        base_manager = Mock()
        base_manager.limits = {'var_limit': 0.05}
        base_manager.check_all_limits = Mock(return_value=[])
        
        config = EnhancedLimitsConfig()
        enhanced_manager = EnhancedRiskLimitsManager(base_manager, config)
        
        # 模拟大规模投资组合
        portfolio_state = Mock()
        portfolio_state.total_value = 1000000000  # 10亿
        portfolio_state.leveraged_value = 1500000000
        portfolio_state.allocations = {f'stock_{i}': Mock(weight=0.01) for i in range(100)}
        
        risk_metrics = {
            'var_95': 0.06,
            'max_drawdown': 0.15,
            'volatility': 0.25
        }
        
        # 测量执行时间
        start_time = time.time()
        result = enhanced_manager.check_all_limits(portfolio_state, risk_metrics)
        elapsed_time = time.time() - start_time
        
        # 验证性能（应在合理时间内完成）
        self.assertLess(elapsed_time, 1.0, "检查应在1秒内完成")
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main(verbosity=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
