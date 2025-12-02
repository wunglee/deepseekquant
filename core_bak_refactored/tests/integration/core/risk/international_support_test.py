"""
国际化支持测试

测试不同市场（US/CN/HK/JP/EU/SG）的风险指标计算与专家参数优化验证
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core_bak_refactored.core.risk.risk_metrics_service import RiskMetricsService
from core_bak_refactored.core.share.market.market_config import MarketConfigManager


class TestInternationalSupport(unittest.TestCase):
    """测试国际化支持"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        # 生成模拟收益率数据
        self.returns_cn = pd.Series(np.random.normal(0.001, 0.025, 100))
        self.returns_us = pd.Series(np.random.normal(0.0008, 0.015, 100))
        self.returns_hk = pd.Series(np.random.normal(0.0009, 0.018, 100))
        
        # 添加市场特征
        # A股：添加涨跌停
        self.returns_cn.iloc[10] = 0.095  # 接近涨停
        self.returns_cn.iloc[50] = -0.095  # 接近跌停
        
        # 美股：添加熔断级别波动
        self.returns_us.iloc[20] = -0.068  # 接近7%熔断
        
        # 配置管理器
        self.config_manager = MarketConfigManager()
    
    def test_config_manager_initialization(self):
        """测试配置管理器初始化"""
        # 验证市场注册表
        self.assertIn('CN', self.config_manager.market_registry)
        self.assertIn('US', self.config_manager.market_registry)
        self.assertIn('HK', self.config_manager.market_registry)
        
        # 验证市场信息
        cn_info = self.config_manager.get_market_info('CN')
        self.assertEqual(cn_info['currency'], 'CNY')
        self.assertEqual(cn_info['default_trading_days'], 245)
        
        us_info = self.config_manager.get_market_info('US')
        self.assertEqual(us_info['currency'], 'USD')
        self.assertEqual(us_info['default_trading_days'], 252)
    
    def test_generate_config_template_cn(self):
        """测试生成CN市场配置模板"""
        config = self.config_manager.generate_config_template('CN')
        
        self.assertEqual(config['market_type'], 'CN')
        self.assertEqual(config['trading_days_per_year'], 245)
        self.assertTrue(config['market_configs']['CN']['has_limit_up_down'])
        self.assertIn('limit_thresholds', config['market_configs']['CN'])
        self.assertEqual(config['market_configs']['CN']['limit_thresholds']['main_board'], 0.10)
    
    def test_generate_config_template_us(self):
        """测试生成US市场配置模板"""
        config = self.config_manager.generate_config_template('US')
        
        self.assertEqual(config['market_type'], 'US')
        self.assertEqual(config['trading_days_per_year'], 252)
        self.assertFalse(config['market_configs']['US']['has_limit_up_down'])
        self.assertEqual(config['market_configs']['US']['circuit_breaker_levels'], [0.07, 0.13, 0.20])
        self.assertEqual(config['market_configs']['US']['luld_threshold'], 0.05)
    
    def test_cn_market_risk_service(self):
        """测试CN市场风险服务"""
        config = self.config_manager.generate_config_template('CN')
        service = RiskMetricsService(config)
        
        # 验证初始化
        self.assertEqual(service.market_type, 'CN')
        self.assertEqual(service.trading_days_per_year, 245)
        
        # 计算风险指标
        vol = service.calculate_volatility(self.returns_cn)
        var = service.calculate_value_at_risk(self.returns_cn, 0.95)
        sharpe = service.calculate_sharpe_ratio(self.returns_cn)
        
        self.assertGreater(vol, 0)
        self.assertGreater(var, 0)
        self.assertNotEqual(sharpe, 0)
    
    def test_us_market_risk_service(self):
        """测试US市场风险服务"""
        config = self.config_manager.generate_config_template('US')
        service = RiskMetricsService(config)
        
        # 验证初始化
        self.assertEqual(service.market_type, 'US')
        self.assertEqual(service.trading_days_per_year, 252)
        
        # 计算风险指标
        vol = service.calculate_volatility(self.returns_us)
        var = service.calculate_value_at_risk(self.returns_us, 0.95)
        sharpe = service.calculate_sharpe_ratio(self.returns_us)
        
        self.assertGreater(vol, 0)
        self.assertGreater(var, 0)
        self.assertNotEqual(sharpe, 0)
    
    def test_limit_up_down_detection_cn(self):
        """测试CN市场涨跌停检测"""
        config = self.config_manager.generate_config_template('CN')
        service = RiskMetricsService(config)
        
        # 检测涨跌停
        has_limit = service._has_limit_hit(self.returns_cn, board_type='main_board')
        self.assertTrue(has_limit)  # 应该检测到涨跌停
        
        # 正常收益率
        normal_returns = pd.Series([0.01, 0.02, -0.01, -0.02, 0.015])
        has_limit_normal = service._has_limit_hit(normal_returns, board_type='main_board')
        self.assertFalse(has_limit_normal)  # 不应该检测到涨跌停
    
    def test_circuit_breaker_detection_us(self):
        """测试US市场熔断检测"""
        config = self.config_manager.generate_config_template('US')
        service = RiskMetricsService(config)
        
        # 检测市场异常
        anomalies = service._detect_market_anomalies(self.returns_us)
        
        # 应该检测到接近熔断的情况
        self.assertGreater(len(anomalies), 0)
        # 检查是否有熔断异常
        has_circuit_breaker = any(a['type'] == 'circuit_breaker' for a in anomalies.values())
        self.assertTrue(has_circuit_breaker)
    
    def test_enhanced_sharpe_ratio(self):
        """测试增强版夏普比率"""
        config = self.config_manager.generate_config_template('US')
        service = RiskMetricsService(config)
        
        # 计算增强夏普比率
        result = service.calculate_sharpe_ratio_enhanced(
            self.returns_us,
            include_market_premium=True,
            adjust_for_anomalies=True
        )
        
        self.assertIn('standard_sharpe', result)
        self.assertIn('enhanced_sharpe', result)
        self.assertIn('adjustment_factors', result)
        self.assertEqual(result['market_type'], 'US')
        self.assertGreaterEqual(result['anomalies_detected'], 0)
    
    def test_cross_market_comparison(self):
        """测试跨市场风险对比"""
        config = self.config_manager.generate_config_template('CN')
        service = RiskMetricsService(config)
        
        # 准备多市场数据
        returns_map = {
            'CN': self.returns_cn,
            'US': self.returns_us
        }
        
        # 执行跨市场对比
        comparison = service.calculate_cross_market_risk_comparison(returns_map)
        
        self.assertEqual(len(comparison['markets_analyzed']), 2)
        self.assertIn('CN', comparison['risk_metrics'])
        self.assertIn('US', comparison['risk_metrics'])
        self.assertIn('relative_risk', comparison)
    
    def test_risk_free_rate_by_market(self):
        """测试不同市场的无风险利率"""
        # CN市场
        cn_config = self.config_manager.generate_config_template('CN')
        cn_service = RiskMetricsService(cn_config)
        cn_rf = cn_service.get_risk_free_rate()
        self.assertAlmostEqual(cn_rf, 0.03, places=3)
        
        # US市场
        us_config = self.config_manager.generate_config_template('US')
        us_service = RiskMetricsService(us_config)
        us_rf = us_service.get_risk_free_rate()
        self.assertAlmostEqual(us_rf, 0.045, places=3)
    
    def test_trading_days_by_market(self):
        """测试不同市场的年交易日"""
        # CN市场
        cn_config = self.config_manager.generate_config_template('CN')
        cn_service = RiskMetricsService(cn_config)
        self.assertEqual(cn_service.trading_days_per_year, 245)
        
        # US市场
        us_config = self.config_manager.generate_config_template('US')
        us_service = RiskMetricsService(us_config)
        self.assertEqual(us_service.trading_days_per_year, 252)
        
        # HK市场
        hk_config = self.config_manager.generate_config_template('HK')
        hk_service = RiskMetricsService(hk_config)
        self.assertEqual(hk_service.trading_days_per_year, 247)


    def test_sg_liquidity_weight_adjustment(self):
        """测试SG流动性权重调整为1.25（专家建议第15轮）"""
        config = self.config_manager.generate_config_template('SG')
        sg_config = config['market_configs']['SG']
        self.assertEqual(sg_config['liquidity_risk_weight'], 1.25,
                        "SG流动性权重应为1.25（反映良好市场基础设施）")
    
    def test_us_liquidity_weight_adjustment(self):
        """测试US流动性权重调整为0.85（专家建议第15轮）"""
        config = self.config_manager.generate_config_template('US')
        us_config = config['market_configs']['US']
        self.assertEqual(us_config['liquidity_risk_weight'], 0.85,
                        "US流动性权重应为0.85（反映近期流动性变化）")
    
    def test_brexit_risk_weight_decay_baseline(self):
        """测试Brexit权重时间衰减机制：基准日期（专家建议第15轮）"""
        # 2020年基准：应为1.15
        weight_2020 = self.config_manager._get_brexit_risk_weight(datetime(2020, 1, 31))
        self.assertAlmostEqual(weight_2020, 1.15, places=2,
                              msg="2020-01-31（Brexit生效日）权重应为1.15")
    
    def test_brexit_risk_weight_decay_progression(self):
        """测试Brexit权重时间衰减机制：衰减过程（专家建议第15轮）"""
        # 2021年：应衰减至约1.0925
        weight_2021 = self.config_manager._get_brexit_risk_weight(datetime(2021, 1, 31))
        expected_2021 = 1.15 * 0.95
        self.assertAlmostEqual(weight_2021, expected_2021, places=3,
                              msg="2021年权重应按年5%衰减")
        
        # 2022年：应继续衰减但未触及下限
        weight_2022 = self.config_manager._get_brexit_risk_weight(datetime(2022, 1, 31))
        expected_2022 = 1.15 * (0.95 ** 2)  # 1.15 * 0.9025 = 1.037875
        self.assertAlmostEqual(weight_2022, expected_2022, places=3,
                              msg="2022年权重应按年5%持续衰减")
    
    def test_brexit_risk_weight_decay_floor(self):
        """测试Brexit权重时间衰减机制：下限保护（专家建议第15轮）"""
        # 2025年及以后：应触发下限1.0
        weight_2025 = self.config_manager._get_brexit_risk_weight(datetime(2025, 1, 31))
        self.assertGreaterEqual(weight_2025, 1.0,
                               msg="权重下限应为1.0（中性水平）")
        self.assertLess(weight_2025, 1.15,
                       msg="2025年权重应已衰减")
        
        # 远期（2030年）：仍应保持在下限
        weight_2030 = self.config_manager._get_brexit_risk_weight(datetime(2030, 1, 31))
        self.assertEqual(weight_2030, 1.0,
                        msg="远期权重应稳定在下限1.0")
    
    def test_eu_config_uses_dynamic_brexit_weight(self):
        """测试EU市场配置集成动态Brexit权重（专家建议第15轮）"""
        config = self.config_manager.generate_config_template('EU')
        eu_config = config['market_configs']['EU']
        
        # 验证EU配置包含brexit_risk_weight
        self.assertIn('brexit_risk_weight', eu_config,
                     "EU配置应包含动态Brexit权重")
        
        # 验证权重在合理范围内
        brexit_weight = eu_config['brexit_risk_weight']
        self.assertGreaterEqual(brexit_weight, 1.0,
                               msg="Brexit权重应 >= 1.0")
        self.assertLessEqual(brexit_weight, 1.15,
                            msg="Brexit权重应 <= 1.15")


if __name__ == '__main__':
    unittest.main()
