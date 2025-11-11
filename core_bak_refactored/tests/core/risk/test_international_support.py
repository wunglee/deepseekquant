"""
国际化支持测试

测试不同市场（US/CN/HK）的风险指标计算
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from core.risk.risk_metrics_service import RiskMetricsService
from core.risk.international_config import MarketConfigManager


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


if __name__ == '__main__':
    unittest.main()
