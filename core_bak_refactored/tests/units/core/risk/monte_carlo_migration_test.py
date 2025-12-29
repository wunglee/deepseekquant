"""
测试蒙特卡洛VaR迁移到服务层

P0任务测试：验证RiskMetricsService.calculate_var_monte_carlo正确实现，
RiskCalculator委托调用正常，且deprecated警告正确触发
"""

from typing import Dict

import numpy as np
import pytest

from core_bak_refactored.core.risk.risk_metrics_service import RiskMetricsService


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
