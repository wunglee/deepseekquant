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

from core_bak_refactored.core.risk.risk_metrics_service import RiskMetricsService
from core_bak_refactored.core.share.market_config import MarketConfigManager


class TestInternationalSupport(unittest.TestCase):
    """测试国际化支持"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        # 生成模拟收益率数据
        self.returns_cn = pd.Series(np.random.normal(0.001, 0.025, 100))
        self.returns_us = pd.Series(np.random.normal(0.0008, 0.015, 100))
        self.returns_hk = pd.Series(np.random.normal(0.0009, 0.018, 100))
        self.returns_jp = pd.Series(np.random.normal(0.0003, 0.012, 100))  # 日本：低收益低波动
        self.returns_eu = pd.Series(np.random.normal(0.0007, 0.016, 100))  # 欧洲：政治风险
        self.returns_sg = pd.Series(np.random.normal(0.0006, 0.020, 100))  # 新加坡：外部依赖
        
        # 添加市场特征
        # A股：添加涨跌停
        self.returns_cn.iloc[10] = 0.095  # 接近涨停
        self.returns_cn.iloc[50] = -0.095  # 接近跌停
        
        # 美股：添加熔断级别波动
        self.returns_us.iloc[20] = -0.068  # 接近7%熔断
        
        # 日本：添加货币政策跳跃（黑田经济学）
        self.returns_jp.iloc[30] = 0.035  # 政策跳跃
        
        # 欧洲：添加政治事件冲击（脱欧等）
        self.returns_eu.iloc[40] = -0.045  # 政治事件冲击
        
        # 新加坡：添加全球冲击敏感性
        self.returns_sg.iloc[60] = -0.055  # 外部冲击
        
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
    
    def test_jp_market_config(self):
        """测试日本市场配置（第14轮专家补充）"""
        config = self.config_manager.generate_config_template('JP')
        jp_config = config['market_configs']['JP']
        
        # 验证专家建议的参数
        self.assertEqual(jp_config['var_method_priority'], 't_distribution')  # 货币政策主导
        self.assertEqual(jp_config['covariance_lookback'], 504)  # 2年政策持续性
        self.assertEqual(jp_config['jump_adjustment_coef'], 0.022)  # 通缩环境跳跃较小
        self.assertEqual(jp_config['evt_threshold'], 0.88)  # 尾部风险中等
        self.assertEqual(jp_config['min_required_returns'], 60)  # 通缩环境需要更多样本
        self.assertEqual(jp_config['volatility_persistence'], 0.95)  # 高度持续
        self.assertEqual(jp_config['liquidity_risk_weight'], 0.9)  # 流动性充足
        self.assertEqual(jp_config['deflation_risk_adjustment'], 0.01)  # 通缩风险调整
        
        # 验证无风险利率
        service = RiskMetricsService(config)
        jp_rf = service.get_risk_free_rate()
        self.assertAlmostEqual(jp_rf, 0.005, places=3)  # 接近零利率
    
    def test_eu_market_config(self):
        """测试欧洲市场配置（第14轮专家补充）"""
        config = self.config_manager.generate_config_template('EU')
        eu_config = config['market_configs']['EU']
        
        # 验证专家建议的参数
        self.assertEqual(eu_config['var_method_priority'], 'historical_simulation')  # 政治事件驱动
        self.assertEqual(eu_config['covariance_lookback'], 252)  # 1年政治周期
        self.assertEqual(eu_config['jump_adjustment_coef'], 0.025)  # 政治事件跳跃
        self.assertEqual(eu_config['evt_threshold'], 0.87)  # 政治尾部风险
        self.assertEqual(eu_config['min_required_returns'], 45)  # 政治事件影响估计
        self.assertEqual(eu_config['volatility_persistence'], 0.90)  # 政治事件降低持续性
        self.assertEqual(eu_config['liquidity_risk_weight'], 1.0)  # 跨国流动性差异
        self.assertEqual(eu_config['brexit_risk_weight'], 1.15)  # 英国脱欧风险
        self.assertEqual(eu_config['banking_sector_risk'], 0.008)  # 银行体系风险
        
        # 验证无风险利率
        service = RiskMetricsService(config)
        eu_rf = service.get_risk_free_rate()
        self.assertAlmostEqual(eu_rf, 0.025, places=3)  # 欧洲国债利率
    
    def test_sg_market_config(self):
        """测试新加坡市场配置（第14轮专家补充）"""
        config = self.config_manager.generate_config_template('SG')
        sg_config = config['market_configs']['SG']
        
        # 验证专家建议的参数
        self.assertEqual(sg_config['var_method_priority'], 'evt')  # 外部冲击敏感
        self.assertEqual(sg_config['covariance_lookback'], 189)  # 9个月资本流动
        self.assertEqual(sg_config['jump_adjustment_coef'], 0.028)  # 全球资本流动跳跃
        self.assertEqual(sg_config['evt_threshold'], 0.84)  # 较低阈值（敏感）
        self.assertEqual(sg_config['min_required_returns'], 40)  # 市场小但数据质量高
        self.assertEqual(sg_config['volatility_persistence'], 0.88)  # 外部依赖性强
        self.assertEqual(sg_config['liquidity_risk_weight'], 1.3)  # 市场规模小
        self.assertEqual(sg_config['trade_openness_risk'], 0.012)  # 贸易开放度风险
        self.assertEqual(sg_config['currency_risk_weight'], 1.25)  # 汇率政策风险
        
        # 验证无风险利率
        service = RiskMetricsService(config)
        sg_rf = service.get_risk_free_rate()
        self.assertAlmostEqual(sg_rf, 0.030, places=3)  # 新加坡政府债券利率
    
    def test_jp_market_risk_calculation(self):
        """测试日本市场风险计算"""
        config = self.config_manager.generate_config_template('JP')
        service = RiskMetricsService(config)
        
        # 计算风险指标
        vol = service.calculate_volatility(self.returns_jp)
        var = service.calculate_value_at_risk(self.returns_jp, 0.95)
        sharpe = service.calculate_sharpe_ratio(self.returns_jp)
        
        self.assertGreater(vol, 0)
        self.assertGreater(var, 0)
        # 日本市场低收益率环境下夏普比率可能较低
        self.assertIsNotNone(sharpe)
    
    def test_eu_market_risk_calculation(self):
        """测试欧洲市场风险计算"""
        config = self.config_manager.generate_config_template('EU')
        service = RiskMetricsService(config)
        
        # 计算风险指标
        vol = service.calculate_volatility(self.returns_eu)
        var = service.calculate_value_at_risk(self.returns_eu, 0.95)
        sharpe = service.calculate_sharpe_ratio(self.returns_eu)
        
        self.assertGreater(vol, 0)
        self.assertGreater(var, 0)
        self.assertIsNotNone(sharpe)
    
    def test_sg_market_risk_calculation(self):
        """测试新加坡市场风险计算"""
        config = self.config_manager.generate_config_template('SG')
        service = RiskMetricsService(config)
        
        # 计算风险指标
        vol = service.calculate_volatility(self.returns_sg)
        var = service.calculate_value_at_risk(self.returns_sg, 0.95)
        sharpe = service.calculate_sharpe_ratio(self.returns_sg)
        
        self.assertGreater(vol, 0)
        self.assertGreater(var, 0)
        self.assertIsNotNone(sharpe)
    
    def test_all_markets_config_completeness(self):
        """测试所有6个市场配置完整性"""
        markets = ['CN', 'US', 'HK', 'JP', 'EU', 'SG']
        
        for market in markets:
            config = self.config_manager.generate_config_template(market)
            market_config = config['market_configs'][market]
            
            # 验证必需参数存在
            required_params = [
                'var_method_priority',
                'covariance_lookback',
                'jump_adjustment_coef',
                'evt_threshold',
                'min_required_returns',
                'volatility_persistence',
                'liquidity_risk_weight',
                'political_risk_premium'
            ]
            
            for param in required_params:
                self.assertIn(param, market_config, 
                             f"{market}市场缺少参数: {param}")
                self.assertIsNotNone(market_config[param], 
                                   f"{market}市场参数{param}为None")
    
    def test_six_markets_cross_comparison(self):
        """测试6个市场风险对比"""
        config = self.config_manager.generate_config_template('CN')
        service = RiskMetricsService(config)
        
        # 准备6市场数据
        returns_map = {
            'CN': self.returns_cn,
            'US': self.returns_us,
            'HK': self.returns_hk,
            'JP': self.returns_jp,
            'EU': self.returns_eu,
            'SG': self.returns_sg
        }
        
        # 执行跨市场对比
        comparison = service.calculate_cross_market_risk_comparison(returns_map)
        
        self.assertEqual(len(comparison['markets_analyzed']), 6)
        for market in ['CN', 'US', 'HK', 'JP', 'EU', 'SG']:
            self.assertIn(market, comparison['risk_metrics'])


if __name__ == '__main__':
    unittest.main()
