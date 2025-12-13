"""
AKShare数据提供者测试套件
测试AKShareDataProvider的各种功能和边界情况
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# 导入被测试的类
from core_bak_refactored.core.data.providers.akshare_provider import AKShareDataProvider
from core_bak_refactored.core.data.providers.protocols import PriceData, IntradayData


class TestAKShareDataProvider(unittest.TestCase):
    """AKShare数据提供者测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = AKShareDataProvider()
        
    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.provider, AKShareDataProvider)
        self.assertTrue(self.provider.available)
        self.assertIsNotNone(self.provider.ak)
        
    def test_get_test_symbol(self):
        """测试获取测试符号"""
        symbol = self.provider.get_test_symbol()
        self.assertEqual(symbol, '000300.SH')
        

class TestAKShareIntradayData(unittest.TestCase):
    """AKShare分时数据获取测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = AKShareDataProvider()
        
    @patch.object(AKShareDataProvider, '_fetch_real_intraday_from_akshare')
    def test_get_intraday_data_from_api_success(self, mock_fetch_real):
        """测试从真实API成功获取分时数据"""
        # 模拟真实API返回
        mock_intraday = IntradayData(
            symbol='000300.SH',
            name='沪深300',
            current_price=3500.0,
            yesterday_close=3480.0,
            change=20.0,
            change_percent=0.57,
            ticks=[],
            order_book_bids=[],
            order_book_asks=[],
            tickers=[],
            trade_date='2023-01-03'  # 周二，工作日
        )
        mock_fetch_real.return_value = mock_intraday
        
        # 调用方法（使用工作日）
        result = self.provider.get_intraday_data('000300.SH', '2023-01-03')
        
        # 验证结果
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(result.symbol, '000300.SH')
        self.assertEqual(result.current_price, 3500.0)
        mock_fetch_real.assert_called_once_with('000300.SH', '2023-01-03')
        
    @patch.object(AKShareDataProvider, '_fetch_real_intraday_from_akshare')
    def test_get_intraday_data_memory_cache_hit(self, mock_fetch_real):
        """测试内存缓存命中"""
        # 第一次调用（会缓存）
        mock_intraday = IntradayData(
            symbol='000300.SH',
            name='沪深300',
            current_price=3500.0,
            yesterday_close=3480.0,
            change=20.0,
            change_percent=0.57,
            ticks=[],
            order_book_bids=[],
            order_book_asks=[],
            tickers=[],
            trade_date='2023-01-01'
        )
        mock_fetch_real.return_value = mock_intraday
        
        result1 = self.provider.get_intraday_data('000300.SH', '2023-01-01')
        
        # 第二次调用（应从缓存读取）
        result2 = self.provider.get_intraday_data('000300.SH', '2023-01-01')
        
        # 验证
        self.assertEqual(result1.current_price, result2.current_price)
        # 只调用一次真实API
        mock_fetch_real.assert_called_once()
        
    @patch.object(AKShareDataProvider, '_fetch_real_intraday_from_akshare')
    @patch.object(AKShareDataProvider, '_get_previous_trading_day')
    def test_get_intraday_data_fallback_to_previous_day(self, mock_prev_day, mock_fetch_real):
        """测试fallback到前一交易日缓存"""
        # 当天API调用失败
        mock_fetch_real.side_effect = Exception("API失败")
        mock_prev_day.return_value = '2022-12-31'
        
        # 预先写入前一天的缓存
        prev_intraday = IntradayData(
            symbol='000300.SH',
            name='沪深300',
            current_price=3480.0,
            yesterday_close=3460.0,
            change=20.0,
            change_percent=0.58,
            ticks=[],
            order_book_bids=[],
            order_book_asks=[],
            tickers=[],
            trade_date='2022-12-31'
        )
        cache_key = "intraday_000300.SH_2022-12-31"
        self.provider._set_to_memory_cache_obj(cache_key, prev_intraday)
        
        # 调用当天数据（会fallback）
        result = self.provider.get_intraday_data('000300.SH', '2023-01-01')
        
        # 验证使用了前一天的数据
        self.assertEqual(result.current_price, 3480.0)
        
    @patch.object(AKShareDataProvider, '_fetch_real_intraday_from_akshare')
    def test_get_intraday_data_fallback_to_mock(self, mock_fetch_real):
        """测试fallback到模拟数据"""
        # API调用失败
        mock_fetch_real.side_effect = Exception("API失败")
        
        # 调用方法（会生成模拟数据），使用工作日
        result = self.provider.get_intraday_data('000300.SH', '2023-01-03')
        
        # 验证返回了模拟数据
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(result.symbol, '000300.SH')
        # 模拟数据应该有tick，但如果当前时间在交易时间前可能为0
        # 所以我们只检查其他字段
        self.assertIsNotNone(result.order_book_bids)
        self.assertIsNotNone(result.order_book_asks)
        

class TestAKShareIntradayHelperMethods(unittest.TestCase):
    """AKShare分时数据辅助方法测试"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = AKShareDataProvider()
        
    @patch('akshare.stock_zh_a_hist_min_em')
    def test_fetch_real_intraday_from_akshare_success(self, mock_ak_api):
        """测试真实API调用成功"""
        # 模拟AKShare API返回
        mock_df = pd.DataFrame({
            '时间': ['2023-01-01 09:30:00', '2023-01-01 09:31:00', '2023-01-01 09:32:00'],
            '收盘': [3500.0, 3501.0, 3502.0],
            '开盘': [3498.0, 3500.0, 3501.0],
            '最高': [3502.0, 3503.0, 3504.0],
            '最低': [3497.0, 3499.0, 3500.0],
            '成交量': [10000, 11000, 12000],
            '成交额': [35000000, 35100000, 35200000],
            '涨跌额': [20.0, 21.0, 22.0]
        })
        mock_ak_api.return_value = mock_df
        self.provider.ak.stock_zh_a_hist_min_em = mock_ak_api
        
        # 调用方法
        result = self.provider._fetch_real_intraday_from_akshare('000300.SH', '2023-01-01')
        
        # 验证结果
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(result.symbol, '000300.SH')
        self.assertGreater(len(result.ticks), 0)
        
    @patch('akshare.stock_zh_a_hist_min_em')
    def test_fetch_real_intraday_from_akshare_empty_data(self, mock_ak_api):
        """测试API返回空数据"""
        mock_ak_api.return_value = pd.DataFrame()
        self.provider.ak.stock_zh_a_hist_min_em = mock_ak_api
        
        # 调用方法
        result = self.provider._fetch_real_intraday_from_akshare('000300.SH', '2023-01-01')
        
        # 应返回None
        self.assertIsNone(result)
        
    def test_get_previous_trading_day(self):
        """测试获取前一交易日"""
        # 测试工作日
        result = self.provider._get_previous_trading_day('2023-01-05')  # 星期四
        self.assertEqual(result, '2023-01-04')  # 星期三
        
        # 测试周一（应跳过周末）
        result = self.provider._get_previous_trading_day('2023-01-02')  # 星期一
        self.assertEqual(result, '2022-12-30')  # 上周五
    
    def test_get_latest_trading_day(self):
        """测试获取最近交易日"""
        # 测试工作日（周三）
        result = self.provider._get_latest_trading_day('2023-01-04')
        self.assertEqual(result, '2023-01-04')  # 返回当天
        
        # 测试周六
        result = self.provider._get_latest_trading_day('2023-01-07')  # 周六
        self.assertEqual(result, '2023-01-06')  # 周五
        
        # 测试周日
        result = self.provider._get_latest_trading_day('2023-01-08')  # 周日
        self.assertEqual(result, '2023-01-06')  # 周五
    
    @patch('core_bak_refactored.core.data.providers.akshare_provider.AKShareDataProvider._get_latest_trading_day')
    def test_get_intraday_data_weekend_fallback(self, mock_latest_day):
        """测试周末调用时自动获取最近交易日"""
        # 模拟最近交易日为周五
        mock_latest_day.return_value = '2023-12-08'  # 周五
        
        # 周末调用（不指定trade_date）
        result = self.provider.get_intraday_data('000300.SH')
        
        # 验证调用了_get_latest_trading_day
        mock_latest_day.assert_called_once()
        
        # 验证返回了数据
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(result.symbol, '000300.SH')
        
    def test_generate_mock_intraday_data(self):
        """测试生成模拟分时数据"""
        result = self.provider._generate_mock_intraday_data('000300.SH', '2023-01-03')  # 工作日
        
        # 验证基本字段
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(result.symbol, '000300.SH')
        self.assertEqual(result.trade_date, '2023-01-03')
        
        # 验证tick数据（只检查是否为列表，不检查数量，因为取决于当前时间）
        self.assertIsInstance(result.ticks, list)
        
        # 🔧 关键修改：验证盘口和成交明细为空（仅实时数据提供）
        self.assertEqual(len(result.order_book_bids), 0)
        self.assertEqual(len(result.order_book_asks), 0)
        self.assertEqual(len(result.tickers), 0)
        
    def test_generate_mock_order_book(self):
        """测试生成模拟盘口"""
        bids, asks = self.provider._generate_mock_order_book(3500.0)
        
        # 验证数量
        self.assertEqual(len(bids), 10)
        self.assertEqual(len(asks), 10)
        
        # 验证买盘价格递减
        self.assertGreater(bids[0].price, bids[1].price)
        
        # 验证卖盘价格递增
        self.assertLess(asks[0].price, asks[1].price)
        
    def test_generate_mock_tickers(self):
        """测试生成模拟成交明细"""
        tickers = self.provider._generate_mock_tickers(3500.0)
        
        # 验证数量
        self.assertEqual(len(tickers), 20)
        
        # 验证字段
        for ticker in tickers:
            self.assertIsNotNone(ticker.time)
            self.assertIsNotNone(ticker.price)
            self.assertIsNotNone(ticker.volume)
            self.assertIn(ticker.direction, ['buy', 'sell'])


class TestAKShareIntradayConversion(unittest.TestCase):
    """AKShare数据转换测试"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = AKShareDataProvider()
        
    def test_convert_akshare_df_to_intraday(self):
        """测试DataFrame转IntradayData"""
        # 构造测试数据
        df = pd.DataFrame({
            '时间': ['2023-01-01 09:30:00', '2023-01-01 09:31:00'],
            '收盘': [3500.0, 3501.0],
            '开盘': [3498.0, 3500.0],
            '最高': [3502.0, 3503.0],
            '最低': [3497.0, 3499.0],
            '成交量': [10000, 11000],
            '涨跌额': [20.0, 21.0]
        })
        
        # 调用转换方法
        result = self.provider._convert_akshare_df_to_intraday(df, '000300.SH', '2023-01-01')
        
        # 验证结果
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(result.symbol, '000300.SH')
        self.assertEqual(len(result.ticks), 2)
        
        # 验证第一个tick
        first_tick = result.ticks[0]
        self.assertEqual(first_tick.time, '09:30')
        self.assertEqual(first_tick.price, 3500.0)
        

if __name__ == '__main__':
    unittest.main()
