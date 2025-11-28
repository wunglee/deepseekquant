"""
测试蒙特卡洛VaR迁移到服务层

P0任务测试：验证RiskMetricsService.calculate_var_monte_carlo正确实现，
RiskCalculator委托调用正常，且deprecated警告正确触发
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any

from core_bak_refactored.core.risk.risk_metrics_service import RiskMetricsService
from core_bak_refactored.core.risk.risk_calculator import RiskCalculator


class MockPortfolioState:
    """模拟组合状态"""
    def __init__(self, allocations: Dict[str, Dict[str, float]]):
        self.allocations = allocations


@pytest.fixture
def basic_config():
    """基础配置"""
    return {
        'market_type': 'CN',
        'trading_days_per_year': 252,
        'monte_carlo_sims': 1000,
        'monte_carlo_seed': 42,
        'min_data_points': 63,
        'market_configs': {
            'CN': {
                'trading_days': 252,
                'risk_free_rate': 0.03,
                'base_currency': 'CNY'
            }
        }
    }


@pytest.fixture
def sample_market_data():
    """生成样本市场数据"""
    np.random.seed(42)
    n_days = 100
    
    # 生成价格序列
    prices_aapl = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n_days))
    prices_msft = 150 * np.cumprod(1 + np.random.normal(0.0006, 0.018, n_days))
    
    return {
        'prices': {
            'AAPL': {
                'close': prices_aapl.tolist(),
                'currency': 'USD'
            },
            'MSFT': {
                'close': prices_msft.tolist(),
                'currency': 'USD'
            }
        },
        'risk_free_rate': 0.03
    }


@pytest.fixture
def sample_portfolio_state():
    """样本组合状态"""
    return MockPortfolioState({
        'AAPL': {'weight': 0.6},
        'MSFT': {'weight': 0.4}
    })


class TestMonteCarloMigration:
    """蒙特卡洛VaR迁移测试"""
    
    def test_service_layer_monte_carlo_basic(self, basic_config, sample_market_data, sample_portfolio_state):
        """测试服务层蒙特卡洛VaR基础功能"""
        service = RiskMetricsService(basic_config)
        
        var = service.calculate_var_monte_carlo(
            portfolio_state=sample_portfolio_state,
            market_data=sample_market_data,
            confidence_level=0.95
        )
        
        # 验证返回值有效
        assert not np.isnan(var), "VaR不应为NaN"
        assert var >= 0, "VaR应为正数（表示损失）"
        assert var < 1.0, "VaR应在合理范围内（<100%）"
    
    def test_service_layer_monte_carlo_custom_params(self, basic_config, sample_market_data, sample_portfolio_state):
        """测试服务层蒙特卡洛VaR自定义参数"""
        service = RiskMetricsService(basic_config)
        
        var = service.calculate_var_monte_carlo(
            portfolio_state=sample_portfolio_state,
            market_data=sample_market_data,
            confidence_level=0.99,
            n_simulations=5000,
            random_seed=123
        )
        
        assert not np.isnan(var)
        assert var >= 0
        
        # 验证可重现性
        var2 = service.calculate_var_monte_carlo(
            portfolio_state=sample_portfolio_state,
            market_data=sample_market_data,
            confidence_level=0.99,
            n_simulations=5000,
            random_seed=123
        )
        
        assert abs(var - var2) < 1e-10, "相同随机种子应产生相同结果"
    
    def test_service_layer_monte_carlo_insufficient_data(self, basic_config):
        """测试服务层蒙特卡洛VaR数据不足情况"""
        service = RiskMetricsService(basic_config)
        
        # 数据点不足
        insufficient_data = {
            'prices': {
                'AAPL': {
                    'close': [100, 101, 102],  # 只有3个点
                    'currency': 'USD'
                }
            }
        }
        
        portfolio = MockPortfolioState({'AAPL': {'weight': 1.0}})
        
        var = service.calculate_var_monte_carlo(
            portfolio_state=portfolio,
            market_data=insufficient_data,
            confidence_level=0.95
        )
        
        assert np.isnan(var), "数据不足时应返回NaN"
    
    def test_calculator_delegation_with_deprecation_warning(self, basic_config, sample_market_data, sample_portfolio_state):
        """测试RiskCalculator委托调用并验证deprecation警告"""
        calculator = RiskCalculator(basic_config)
        
        # 捕获警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            var = calculator.calculate_var_monte_carlo(
                portfolio_state=sample_portfolio_state,
                market_data=sample_market_data,
                confidence_level=0.95
            )
            
            # 验证警告触发
            assert len(w) >= 1, "应触发deprecation警告"
            assert issubclass(w[0].category, DeprecationWarning), "应为DeprecationWarning"
            assert "迁移至 RiskMetricsService" in str(w[0].message), "警告消息应提示迁移"
        
        # 验证结果有效
        assert not np.isnan(var)
        assert var >= 0
    
    def test_calculator_delegation_result_consistency(self, basic_config, sample_market_data, sample_portfolio_state):
        """测试RiskCalculator委托结果与服务层一致"""
        calculator = RiskCalculator(basic_config)
        service = RiskMetricsService(basic_config)
        
        # 抑制警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            var_calc = calculator.calculate_var_monte_carlo(
                portfolio_state=sample_portfolio_state,
                market_data=sample_market_data,
                confidence_level=0.95
            )
        
        var_service = service.calculate_var_monte_carlo(
            portfolio_state=sample_portfolio_state,
            market_data=sample_market_data,
            confidence_level=0.95
        )
        
        # 由于使用相同的随机种子（从config），结果应相同
        assert abs(var_calc - var_service) < 1e-10, "委托调用应与直接调用服务层结果一致"
    
    def test_monte_carlo_confidence_level_impact(self, basic_config, sample_market_data, sample_portfolio_state):
        """测试不同置信水平对VaR的影响"""
        service = RiskMetricsService(basic_config)
        
        var_95 = service.calculate_var_monte_carlo(
            portfolio_state=sample_portfolio_state,
            market_data=sample_market_data,
            confidence_level=0.95
        )
        
        var_99 = service.calculate_var_monte_carlo(
            portfolio_state=sample_portfolio_state,
            market_data=sample_market_data,
            confidence_level=0.99
        )
        
        # 99%置信度的VaR应大于95%
        assert var_99 >= var_95, "更高置信度应有更高VaR"
    
    def test_monte_carlo_single_asset_portfolio(self, basic_config, sample_market_data):
        """测试单资产组合蒙特卡洛VaR"""
        service = RiskMetricsService(basic_config)
        
        single_asset_portfolio = MockPortfolioState({
            'AAPL': {'weight': 1.0}
        })
        
        var = service.calculate_var_monte_carlo(
            portfolio_state=single_asset_portfolio,
            market_data=sample_market_data,
            confidence_level=0.95
        )
        
        assert not np.isnan(var)
        assert var >= 0
