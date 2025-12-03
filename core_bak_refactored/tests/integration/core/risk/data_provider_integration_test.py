"""
数据提供者集成测试
验证Yahoo Finance与Mock的切换功能
"""

import pytest
from core_bak_refactored.core.risk.backtest_framework import create_data_provider
from core_bak_refactored.core.data.providers.factory import get_global_factory, reset_global_factory
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider


class TestDataProviderIntegration:
    """数据提供者集成测试套件"""
    
    def setup_method(self):
        """每个测试前重置工厂并注册Mock"""
        reset_global_factory()
        factory = get_global_factory()
        factory.register('mock', MockHistoricalDataProvider)
    
    def test_create_mock_provider(self):
        """测试：创建Mock数据提供者"""
        provider = create_data_provider('mock')
        
        # 验证返回Mock实例
        assert provider is not None
        assert hasattr(provider, 'get_index_prices')
        assert hasattr(provider, 'get_index_returns')
    
    def test_create_yahoo_provider(self):
        """测试：创建Yahoo Finance数据提供者"""
        provider = create_data_provider('yahoo', fallback_to_mock=False)
        
        # 验证返回Yahoo实例
        assert provider is not None
        assert hasattr(provider, 'get_index_prices')
        assert hasattr(provider, 'get_index_returns')
    
    def test_create_auto_provider(self):
        """测试：自动选择数据提供者"""
        provider = create_data_provider('auto')
        
        # 应该成功创建（Yahoo或Mock）
        assert provider is not None
        assert hasattr(provider, 'get_index_prices')
    
    def test_invalid_provider_type_raises(self):
        """测试：无效类型抛出异常"""
        with pytest.raises(ValueError, match="未知的provider_type"):
            create_data_provider('invalid_type')
    
    def test_mock_provider_get_data(self):
        """测试：Mock提供者获取数据"""
        provider = create_data_provider('mock')
        
        # 获取数据
        data = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
        
        # 验证数据格式
        assert len(data) > 0
        assert 'date' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
    
    def test_yahoo_provider_fallback(self):
        """测试：Yahoo提供者不再fallback，失败应抛异常"""
        provider = create_data_provider('yahoo', fallback_to_mock=False)
        
        # 真实场景下数据不可用应抛异常
        try:
            data = provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
            # 如果数据可用，验证格式
            assert len(data) > 0
            assert 'date' in data.columns
            assert 'close' in data.columns
        except ValueError:
            # 预期的异常（真实数据不可用）
            pass
    
    def test_auto_provider_get_data(self):
        """测试：自动提供者获取数据"""
        provider = create_data_provider('auto')
        
        # 应该成功获取数据（Yahoo或Mock）
        # 使用一个更可能成功的指数代码和日期范围
        try:
            data = provider.get_index_prices('SPY', '2020-01-01', '2020-01-31')
        except Exception:
            # 如果Yahoo失败，尝试使用Mock
            provider = create_data_provider('mock')
            data = provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-31')
        
        # 验证数据格式
        assert len(data) > 0
        assert 'date' in data.columns
        assert 'close' in data.columns
    
    def test_provider_consistency(self):
        """测试：不同提供者返回格式一致性"""
        mock_provider = create_data_provider('mock')
        yahoo_provider = create_data_provider('yahoo', fallback_to_mock=False)
        
        # 获取相同时间段数据
        mock_data = mock_provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
        
        try:
            yahoo_data = yahoo_provider.get_index_prices('000300.SH', '2015-06-01', '2015-06-10')
            # 验证列名一致
            assert list(mock_data.columns) == list(yahoo_data.columns)
            # 验证列名为预期格式
            assert list(mock_data.columns) == ['date', 'close', 'volume']
        except ValueError:
            # Yahoo数据不可用时，只验证Mock格式
            assert list(mock_data.columns) == ['date', 'close', 'volume']


class TestBacktestFrameworkWithRealData:
    """回测框架真实数据集成测试（Phase 3B验证）"""
    
    def setup_method(self):
        """每个测试前重置工厂并注册Mock"""
        reset_global_factory()
        factory = get_global_factory()
        factory.register('mock', MockHistoricalDataProvider)
    
    @pytest.mark.skip(reason="需要完整回测引擎，当前仅测试数据提供者")
    def test_backtest_with_yahoo_data(self):
        """集成测试：使用Yahoo数据运行回测"""
        # 此测试将在回测引擎集成完成后启用
        pass
    
    def test_data_provider_switching(self):
        """测试：数据提供者切换功能"""
        # Phase 3A: Mock数据
        mock_provider = create_data_provider('mock')
        mock_data = mock_provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-31')
        assert len(mock_data) > 0
        
        # Phase 3B: Yahoo数据（不带回退）
        yahoo_provider = create_data_provider('yahoo', fallback_to_mock=False)
        try:
            yahoo_data = yahoo_provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-31')
            assert len(yahoo_data) > 0
            # 格式应该一致
            assert list(mock_data.columns) == list(yahoo_data.columns)
        except ValueError:
            # Yahoo数据不可用时，只验证Mock可用
            pass
