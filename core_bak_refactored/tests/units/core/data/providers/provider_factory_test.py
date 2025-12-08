"""
DataProviderFactory 测试 - 验证工厂模式和依赖注入
"""
import unittest
import pandas as pd
from datetime import datetime

from core_bak_refactored.core.data.providers.factory import (
    DataProviderFactory,
    get_global_factory,
    reset_global_factory
)


class CustomMockProvider:
    """自定义Mock Provider用于测试依赖注入"""
    
    def __init__(self, custom_value=100):
        self.custom_value = custom_value
    
    def get_index_prices(self, index_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """返回自定义数据"""
        dates = pd.date_range(start_date, end_date, freq='B')
        prices = [self.custom_value * (1 + i * 0.01) for i in range(len(dates))]
        return pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': [1000000] * len(dates)
        })
    
    def get_index_returns(self, index_id: str, start_date: str, end_date: str) -> pd.Series:
        """返回收益率"""
        df = self.get_index_prices(index_id, start_date, end_date)
        returns = df['close'].pct_change().fillna(0)
        returns.index = df['date']
        return returns


class DataProviderFactoryTest(unittest.TestCase):
    """DataProviderFactory 测试"""
    
    def setUp(self):
        """每个测试前重置全局工厂并注册Mock provider"""
        reset_global_factory()
        # 在测试中手动注册Mock provider
        from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
        self.factory = DataProviderFactory()
        self.factory.register('mock', MockHistoricalDataProvider)
    
    def test_factory_creates_builtin_providers(self):
        """测试工厂能创建所有内置providers"""
        
        # 验证内置providers已注册（不包拮mock，mock已在setUp中手动注册）
        providers = self.factory.list_providers()
        self.assertIn('yahoo', providers)
        self.assertIn('tushare', providers)
        self.assertIn('mock', providers)  # 在setUp中注册
        self.assertIn('real', providers)
        self.assertIn('akshare', providers)
    
    def test_factory_registers_new_providers(self):
        """测试工厂能注册新的商业数据源providers"""
        
        # 验证新添加的providers已注册
        providers = self.factory.list_providers()
        self.assertIn('alpha_vantage', providers)
        self.assertIn('polygon', providers)
        self.assertIn('iex_cloud', providers)
        self.assertIn('finnhub', providers)
        self.assertIn('twelve_data', providers)
    
    def test_create_yahoo_provider(self):
        """测试创建Yahoo provider"""
        provider = self.factory.create('yahoo')
        
        # 验证provider有正确的方法
        self.assertTrue(hasattr(provider, 'get_index_prices'))
        self.assertTrue(hasattr(provider, 'get_index_returns'))
    
    def test_create_akshare_provider(self):
        """测试创建AKShare provider"""
        provider = self.factory.create('akshare')
        
        # 验证provider有正确的方法
        self.assertTrue(hasattr(provider, 'get_index_prices'))
        self.assertTrue(hasattr(provider, 'get_index_returns'))
    
    def test_create_alpha_vantage_provider(self):
        """测试创建Alpha Vantage provider"""
        # Alpha Vantage需要API密钥，这里只是测试能否创建实例
        try:
            provider = self.factory.create('alpha_vantage')
            # 验证provider有正确的方法
            self.assertTrue(hasattr(provider, '_fetch_fn'))
        except Exception as e:
            # 如果没有安装依赖或没有API密钥，应该抛出相应的错误
            self.assertIn('not installed', str(e).lower()) or self.assertIn('api key', str(e).lower())
    
    def test_create_mock_provider(self):
        """测试创建Mock provider"""
        provider = self.factory.create('mock')
        
        # 获取数据验证
        data = provider.get_index_prices('000300.SH', '2020-01-01', '2020-01-10')
        self.assertFalse(data.empty)
        self.assertIn('date', data.columns)
        self.assertIn('close', data.columns)
    
    def test_register_custom_provider(self):
        """测试注册自定义provider（依赖注入）"""
        
        # 注册自定义provider
        self.factory.register('custom', CustomMockProvider)
        
        # 验证已注册
        self.assertTrue(self.factory.is_registered('custom'))
        self.assertIn('custom', self.factory.list_providers())
        
        # 创建并使用
        provider = self.factory.create('custom', custom_value=200)
        data = provider.get_index_prices('TEST', '2020-01-01', '2020-01-05')
        
        # 验证使用了自定义值
        self.assertEqual(data['close'].iloc[0], 200.0)
    
    def test_create_unknown_provider_raises_error(self):
        """测试创建未知provider抛出错误"""
        
        with self.assertRaises(ValueError) as ctx:
            self.factory.create('unknown_provider')
        
        # 验证错误信息包含可用providers
        error_msg = str(ctx.exception)
        self.assertIn('unknown_provider', error_msg)
        self.assertIn('yahoo', error_msg)
    
    def test_global_factory_singleton(self):
        """测试全局工厂单例"""
        factory1 = get_global_factory()
        factory2 = get_global_factory()
        
        # 验证是同一个实例
        self.assertIs(factory1, factory2)
        
        # 在一个工厂注册，另一个能看到
        factory1.register('test', CustomMockProvider)
        self.assertTrue(factory2.is_registered('test'))
    
    def test_reset_global_factory(self):
        """测试重置全局工厂"""
        factory1 = get_global_factory()
        factory1.register('test', CustomMockProvider)
        
        # 重置
        reset_global_factory()
        factory2 = get_global_factory()
        
        # 验证是新实例，且没有之前的注册
        self.assertIsNot(factory1, factory2)
        self.assertFalse(factory2.is_registered('test'))
    
    def test_unregister_provider(self):
        """测试移除provider"""
        self.factory.register('temp', CustomMockProvider)
        
        self.assertTrue(self.factory.is_registered('temp'))
        
        self.factory.unregister('temp')
        
        self.assertFalse(self.factory.is_registered('temp'))
    
    def test_override_builtin_provider(self):
        """测试覆盖内置provider"""
        
        # 覆盖mock provider
        self.factory.register('mock', CustomMockProvider)
        
        # 创建应该使用自定义实现
        provider = self.factory.create('mock', custom_value=300)
        data = provider.get_index_prices('TEST', '2020-01-01', '2020-01-05')
        
        # 验证使用了自定义值
        self.assertEqual(data['close'].iloc[0], 300.0)


if __name__ == '__main__':
    unittest.main()