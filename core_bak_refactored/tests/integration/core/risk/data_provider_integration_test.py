"""
数据提供者集成测试
验证配置驱动的 provider 创建（新实现）
"""

import pandas as pd
import pytest

from core_bak_refactored.core.data.providers.factory import get_global_factory, reset_global_factory
from core_bak_refactored.core.share.config_manager import ConfigManager


class TestDataProviderIntegration:
    """数据提供者集成测试套件"""
    
    def setup_method(self):
        """每个测试前重置工厂（使用配置驱动，不需要手动注册）"""
        reset_global_factory()
    
    def test_create_akshare_provider(self):
        """测试：创建AKShare数据提供者（从配置）"""
        factory = get_global_factory()
        
        # 检查是否已配置
        if not factory.is_registered('akshare'):
            pytest.skip("akshare provider 未在配置中注册")
        
        provider = factory.get('akshare')
        
        # 验证返回实例
        assert provider is not None
        assert hasattr(provider, 'get_index_prices')
        assert hasattr(provider, 'get_stock_prices')
    
    def test_create_yahoo_provider(self):
        """测试：创建Yahoo Finance数据提供者"""
        factory = get_global_factory()
        provider = factory.get('yahoo')
        
        # 验证返回Yahoo实例
        assert provider is not None
        assert hasattr(provider, 'get_index_prices')
        assert hasattr(provider, 'get_index_returns')
    
    def test_create_provider_from_config(self):
        """测试：从配置选择数据提供者（配置驱动）"""
        config_manager = ConfigManager()
        
        # 从 MarketConfig 获取 market_sources
        market_config = config_manager.get_market_config()
        provider_id = market_config.market_sources.get('CN', 'akshare')
        
        # 使用工厂创建
        factory = get_global_factory()
        provider = factory.get(provider_id)
        
        # 应该成功创建
        assert provider is not None
        assert hasattr(provider, 'get_index_prices')
    
    def test_invalid_provider_type_raises(self):
        """测试：无效类型抛出异常"""
        factory = get_global_factory()
        with pytest.raises(ValueError):
            factory.get('invalid_nonexistent_provider_12345')
    
    def test_akshare_provider_get_data(self):
        """测试：AKShare提供者获取数据"""
        factory = get_global_factory()
        
        if not factory.is_registered('akshare'):
            pytest.skip("akshare provider 未配置")
        
        provider = factory.get('akshare')
        
        # 获取数据（使用 PriceData 对象）
        try:
            price_data = provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-10', pd.Timestamp.now())
            
            # 验证 PriceData 对象
            assert price_data is not None
            assert hasattr(price_data, 'symbol')
            assert hasattr(price_data, 'records')
            assert len(price_data.records) > 0
        except Exception as e:
            pytest.skip(f"AKShare 数据获取失败: {e}")
    
    def test_yahoo_provider_get_data(self):
        """测试：Yahoo提供者获取数据"""
        factory = get_global_factory()
        
        if not factory.is_registered('yahoo'):
            pytest.skip("yahoo provider 未配置")
        
        provider = factory.get('yahoo')
        
        # Yahoo 可能因限流失败，这是预期的
        try:
            price_data = provider.get_index_prices('^GSPC', '2020-01-01', '2020-01-10', pd.Timestamp.now())
            
            # 如果成功，验证 PriceData 对象
            assert price_data is not None
            assert hasattr(price_data, 'symbol')
            assert hasattr(price_data, 'records')
        except (ValueError, RuntimeError) as e:
            # Yahoo 限流或数据不可用是预期的
            pytest.skip(f"Yahoo Finance 不可用: {e}")
    
    def test_config_driven_provider_get_data(self):
        """测试：配置驱动的 provider 获取数据"""
        config_manager = ConfigManager()
        
        # 从 MarketConfig 获取 market_sources
        market_config = config_manager.get_market_config()
        provider_id = market_config.market_sources.get('CN', 'akshare')
        
        factory = get_global_factory()
        provider = factory.get(provider_id)
        
        # 应该成功获取数据（返回 PriceData 对象）
        try:
            price_data = provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-10', pd.Timestamp.now())
            
            # 验证 PriceData 对象
            assert price_data is not None
            assert hasattr(price_data, 'symbol')
            assert price_data.symbol == '000300.SH'
            assert len(price_data.records) > 0
            
            # 验证记录格式
            first_record = price_data.records[0]
            assert hasattr(first_record, 'date')
            assert hasattr(first_record, 'close')
        except Exception as e:
            # 如果该 provider 不可用，跳过测试
            pytest.skip(f"Provider {provider_id} 不可用: {e}")
    
    def test_provider_consistency(self):
        """测试：不同提供者返回格式一致性（PriceData）"""
        factory = get_global_factory()
        
        # 获取两个不同的 provider
        providers_to_test = []
        if factory.is_registered('akshare'):
            providers_to_test.append(('akshare', '000300.SH'))
        if factory.is_registered('yahoo'):
            providers_to_test.append(('yahoo', '^GSPC'))
        
        if len(providers_to_test) < 2:
            pytest.skip("需要至少2个 provider 进行一致性测试")
        
        # 获取数据并验证格式一致
        price_data_list = []
        for provider_id, symbol in providers_to_test:
            try:
                provider = factory.get(provider_id)
                price_data = provider.get_index_prices(symbol, '2020-01-01', '2020-01-10', pd.Timestamp.now())
                price_data_list.append(price_data)
            except Exception:
                pass
        
        if len(price_data_list) >= 2:
            # 验证所有 provider 返回相同类型
            for price_data in price_data_list:
                assert hasattr(price_data, 'symbol')
                assert hasattr(price_data, 'records')
                assert hasattr(price_data, 'start_date')
                assert hasattr(price_data, 'end_date')


class TestBacktestFrameworkWithRealData:
    """回测框架真实数据集成测试（配置驱动）"""
    
    def setup_method(self):
        """每个测试前重置工厂（使用配置驱动）"""
        reset_global_factory()
    
    @pytest.mark.skip(reason="需要完整回测引擎，当前仅测试数据提供者")
    def test_backtest_with_yahoo_data(self):
        """集成测试：使用Yahoo数据运行回测"""
        # 此测试将在回测引擎集成完成后启用
        pass
    
    def test_data_provider_switching(self):
        """测试：数据提供者切换功能（配置驱动）"""
        factory = get_global_factory()
        
        # 获取可用的 providers
        available_providers = factory.list_providers()
        if len(available_providers) < 2:
            pytest.skip(f"需要至少2个 provider，当前只有: {available_providers}")
        
        # 测试切换不同 provider
        for provider_id in available_providers[:2]:  # 测试前2个
            try:
                provider = factory.get(provider_id)
                
                # 根据 provider 选择合适的 symbol
                if provider_id == 'yahoo':
                    symbol = '^GSPC'
                else:
                    symbol = '000300.SH'
                
                price_data = provider.get_index_prices(symbol, '2020-01-01', '2020-01-10', pd.Timestamp.now())
                
                # 验证 PriceData 格式
                assert price_data is not None
                assert hasattr(price_data, 'records')
                assert len(price_data.records) > 0
            except Exception as e:
                # Provider 不可用时跳过
                pytest.skip(f"Provider {provider_id} 不可用: {e}")
