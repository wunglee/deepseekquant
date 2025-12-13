"""
Yahoo Finance数据提供者测试套件
测试YahooFinanceDataProvider的各种功能和边界情况
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# 导入被测试的类
from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider
from core_bak_refactored.core.data.providers.protocols import PriceData


class TestYahooFinanceDataProvider(unittest.TestCase):
    """Yahoo Finance数据提供者测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = YahooFinanceDataProvider()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.provider, YahooFinanceDataProvider)
        self.assertIsNotNone(self.provider.yf)
        
    def test_get_test_symbol(self):
        """测试获取测试符号"""
        symbol = self.provider.get_test_symbol()
        self.assertEqual(symbol, '^GSPC')
        
    @patch('yfinance.download')
    def test_get_index_prices_success(self, mock_download):
        """测试成功获取指数价格数据"""
        # 创建模拟数据
        mock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_download.return_value = mock_data
        
        # 测试获取指数价格数据
        result = self.provider.get_index_prices('^GSPC', '2023-01-01', '2023-01-03')
        
        # 验证结果
        self.assertIsInstance(result, PriceData)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.symbol, '^GSPC')
        
        # 验证第一条记录
        first_record = result.records[0]
        self.assertEqual(first_record.open, 100.0)
        self.assertEqual(first_record.high, 105.0)
        self.assertEqual(first_record.low, 99.0)
        self.assertEqual(first_record.close, 104.0)
        self.assertEqual(first_record.volume, 1000)
        
    @patch('yfinance.download')
    def test_get_index_prices_empty_data(self, mock_download):
        """测试获取空数据时的处理"""
        # 模拟返回空数据
        mock_data = pd.DataFrame()
        mock_download.return_value = mock_data
        
        # 应该抛出ValueError异常
        with self.assertRaises(ValueError) as context:
            self.provider.get_index_prices('^GSPC', '2023-01-01', '2023-01-03')
            
        # 错误消息被包装在"Failed to fetch data"中
        self.assertIn("Failed to fetch data for ^GSPC", str(context.exception))
        
    @patch('yfinance.download')
    def test_get_index_prices_none_data(self, mock_download):
        """测试获取None数据时的处理"""
        # 模拟返回None
        mock_download.return_value = None
        
        # 应该抛出ValueError异常
        with self.assertRaises(ValueError) as context:
            self.provider.get_index_prices('^GSPC', '2023-01-01', '2023-01-03')
            
        # 错误消息被包装在"Failed to fetch data"中
        self.assertIn("Failed to fetch data for ^GSPC", str(context.exception))
        
    @patch('yfinance.download')
    def test_get_stock_prices_success(self, mock_download):
        """测试成功获取个股价格数据"""
        # 创建模拟数据
        mock_data = pd.DataFrame({
            'Open': [50.0, 51.0, 52.0],
            'High': [55.0, 56.0, 57.0],
            'Low': [49.0, 50.0, 51.0],
            'Close': [54.0, 55.0, 56.0],
            'Volume': [2000, 2100, 2200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_download.return_value = mock_data
        
        # 测试获取个股价格数据
        result = self.provider.get_stock_prices('AAPL', '2023-01-01', '2023-01-03')
        
        # 验证结果
        self.assertIsInstance(result, PriceData)
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.symbol, 'AAPL')
        
    @patch('yfinance.download')
    def test_get_stock_prices_empty_data(self, mock_download):
        """测试获取个股空数据时的处理"""
        # 模拟返回空数据
        mock_data = pd.DataFrame()
        mock_download.return_value = mock_data
        
        # 应该抛出ValueError异常
        with self.assertRaises(ValueError) as context:
            self.provider.get_stock_prices('AAPL', '2023-01-01', '2023-01-03')
            
        # 错误消息被包装在"Failed to fetch data"中
        self.assertIn("Failed to fetch data for AAPL", str(context.exception))
        
    @patch('yfinance.download')
    def test_get_index_prices_with_datetime_objects(self, mock_download):
        """测试使用datetime对象获取指数价格数据"""
        # 创建模拟数据
        mock_data = pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Low': [99.0],
            'Close': [104.0],
            'Volume': [1000]
        }, index=[pd.Timestamp('2023-01-01')])
        
        mock_download.return_value = mock_data
        
        # 使用datetime对象
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 1)
        
        result = self.provider.get_index_prices('^GSPC', start_date, end_date)
        
        self.assertIsInstance(result, PriceData)
        self.assertEqual(len(result.records), 1)
    
    # 注意：_map_index_to_yahoo 方法已被移除，相关测试已删除
        
    @patch('yfinance.download')
    def test_standardize_format_multiindex_columns(self, mock_download):
        """测试MultiIndex列结构的标准化"""
        # 创建具有MultiIndex列的模拟数据
        columns = pd.MultiIndex.from_tuples([
            ('Open', '^GSPC'), ('High', '^GSPC'), ('Low', '^GSPC'), 
            ('Close', '^GSPC'), ('Volume', '^GSPC')
        ])
        mock_data = pd.DataFrame([
            [100.0, 105.0, 99.0, 104.0, 1000],
            [101.0, 106.0, 100.0, 105.0, 1100]
        ], columns=columns, index=pd.date_range('2023-01-01', periods=2))
        
        mock_download.return_value = mock_data
        
        result = self.provider.get_index_prices('^GSPC', '2023-01-01', '2023-01-02')
        
        self.assertIsInstance(result, PriceData)
        self.assertEqual(len(result.records), 2)
        
    @patch('yfinance.download')
    def test_fetch_with_retry_success_first_attempt(self, mock_download):
        """测试第一次尝试就成功的重试机制"""
        # 创建模拟数据
        mock_data = pd.DataFrame({
            'Open': [100.0],
            'Close': [104.0],
            'Volume': [1000]
        }, index=[pd.Timestamp('2023-01-01')])
        
        mock_download.return_value = mock_data
        
        # 调用内部方法测试
        result = self.provider._fetch_with_retry('^GSPC', '2023-01-01', '2023-01-01')
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        mock_download.assert_called_once()
        
    @patch('yfinance.download')
    def test_fetch_with_retry_rate_limit_then_success(self, mock_download):
        """测试速率限制后重试成功"""
        # 创建模拟数据
        mock_data = pd.DataFrame({
            'Open': [100.0],
            'Close': [104.0],
            'Volume': [1000]
        }, index=[pd.Timestamp('2023-01-01')])
        
        # 第一次调用抛出速率限制错误，第二次调用返回数据
        mock_download.side_effect = [
            Exception("Too Many Requests"),
            mock_data
        ]
        
        # 调用内部方法测试
        result = self.provider._fetch_with_retry('^GSPC', '2023-01-01', '2023-01-01', max_retries=3)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(mock_download.call_count, 2)
        
    @patch('yfinance.download')
    def test_fetch_with_retry_max_attempts_exceeded(self, mock_download):
        """测试超过最大重试次数"""
        # 模拟始终抛出速率限制错误
        mock_download.side_effect = Exception("Too Many Requests")
        
        # 应该抛出Exception异常（因为_fetch_with_retry会直接raise）
        with self.assertRaises(Exception) as context:
            self.provider._fetch_with_retry('^GSPC', '2023-01-01', '2023-01-01', max_retries=2)
            
        self.assertIn("Too Many Requests", str(context.exception))
        self.assertEqual(mock_download.call_count, 3)  # 初始尝试 + 2次重试
        
    @patch('yfinance.download')
    def test_fetch_with_retry_non_rate_limit_error(self, mock_download):
        """测试非速率限制错误"""
        # 模拟抛出其他类型的错误
        mock_download.side_effect = Exception("Network error")
        
        # 应该直接抛出原始异常
        with self.assertRaises(Exception) as context:
            self.provider._fetch_with_retry('^GSPC', '2023-01-01', '2023-01-01')
            
        self.assertIn("Network error", str(context.exception))
        # 当前实现会重试所有异常，所以call_count应该4（初始 + 3次重试）
        self.assertEqual(mock_download.call_count, 4)
    
    @patch('core_bak_refactored.core.data.providers.yahoo_provider.YahooFinanceDataProvider.get_index_prices')
    @patch('core_bak_refactored.core.data.providers.yahoo_provider.YahooFinanceDataProvider.get_stock_prices')
    def test_fetch_from_external_api_index(self, mock_get_stock, mock_get_index):
        """
        测试 _fetch_from_external_api 方法 - 指数
        
        🐞 Bug Fix: 验证抽象方法已实现，不会报 "Can't instantiate abstract class" 错误
        """
        # 模拟指数数据
        mock_price_data = PriceData(
            symbol='^GSPC',
            records=[],
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2023-01-03'),
            count=0
        )
        mock_get_index.return_value = mock_price_data
        
        # 调用 _fetch_from_external_api（指数以 ^ 开头）
        result = self.provider._fetch_from_external_api('^GSPC', '2023-01-01', '2023-01-03')
        
        # 验证调用了 get_index_prices
        mock_get_index.assert_called_once_with('^GSPC', '2023-01-01', '2023-01-03')
        mock_get_stock.assert_not_called()
        self.assertEqual(result, mock_price_data)
    
    @patch('core_bak_refactored.core.data.providers.yahoo_provider.YahooFinanceDataProvider.get_index_prices')
    @patch('core_bak_refactored.core.data.providers.yahoo_provider.YahooFinanceDataProvider.get_stock_prices')
    def test_fetch_from_external_api_stock(self, mock_get_stock, mock_get_index):
        """
        测试 _fetch_from_external_api 方法 - 个股
        
        🐞 Bug Fix: 验证抽象方法已实现，不会报 "Can't instantiate abstract class" 错误
        """
        # 模拟个股数据
        mock_price_data = PriceData(
            symbol='AAPL',
            records=[],
            start_date=pd.Timestamp('2023-01-01'),
            end_date=pd.Timestamp('2023-01-03'),
            count=0
        )
        mock_get_stock.return_value = mock_price_data
        
        # 调用 _fetch_from_external_api（个股不以 ^ 开头）
        result = self.provider._fetch_from_external_api('AAPL', '2023-01-01', '2023-01-03')
        
        # 验证调用了 get_stock_prices
        mock_get_stock.assert_called_once_with('AAPL', '2023-01-01', '2023-01-03')
        mock_get_index.assert_not_called()
        self.assertEqual(result, mock_price_data)
    
    def test_provider_can_be_instantiated(self):
        """
        测试 YahooFinanceDataProvider 可以被实例化
        
        🐞 Bug Fix: 验证不会报 "Can't instantiate abstract class YahooFinanceDataProvider 
                   with abstract method _fetch_from_external_api" 错误
        """
        # 这个测试如果通过，说明所有抽象方法都已实现
        provider = YahooFinanceDataProvider()
        self.assertIsInstance(provider, YahooFinanceDataProvider)
        
        # 验证 _fetch_from_external_api 方法存在
        self.assertTrue(hasattr(provider, '_fetch_from_external_api'))
        self.assertTrue(callable(getattr(provider, '_fetch_from_external_api')))


if __name__ == '__main__':
    unittest.main()