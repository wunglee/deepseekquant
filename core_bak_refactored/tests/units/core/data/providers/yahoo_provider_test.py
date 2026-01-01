"""
Yahoo Finance 数据提供者单元测试

测试范围：
- get_intraday_data 方法的基本功能
- 不同交易时段的处理
- 错误处理和边界情况
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider
from core_bak_refactored.core.data.providers.protocols import IntradayData
from core_bak_refactored.core.share.market.market_enums import TradingPhase, MarketCode


class YahooFinanceDataProviderIntradayTest(unittest.TestCase):
    """测试 YahooFinanceDataProvider 的 get_intraday_data 方法"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = YahooFinanceDataProvider()
        self.test_symbol = "AAPL"
        
    def test_get_intraday_data_before_open_returns_empty(self):
        """测试盘前时段返回空数据"""
        # 模拟盘前时间（美股盘前 8:00）
        mock_time = pd.Timestamp("2025-01-01 08:00:00")
        
        with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.determine_trading_phase',
                   return_value=TradingPhase.BEFORE_OPEN):
            with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.infer_market_from_symbol',
                       return_value=MarketCode.US):
                result = self.provider.get_intraday_data(
                    symbol=self.test_symbol,
                    current_time=mock_time
                )
        
        # 验证返回空数据
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(len(result.ticks), 0)
        self.assertTrue(result.should_poll)  # 盘前应该轮询
        
    def test_get_intraday_data_after_close_returns_empty_on_no_data(self):
        """测试盘后无数据时返回空数据"""
        mock_time = pd.Timestamp("2025-01-01 17:00:00")
        
        with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.determine_trading_phase',
                   return_value=TradingPhase.AFTER_CLOSE):
            with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.infer_market_from_symbol',
                       return_value=MarketCode.US):
                # 模拟 yfinance 返回空数据
                with patch.object(self.provider, '_throttle_request'):
                    mock_ticker = MagicMock()
                    mock_ticker.history.return_value = pd.DataFrame()  # 空DataFrame
                    
                    with patch.object(self.provider.yf, 'Ticker', return_value=mock_ticker):
                        result = self.provider.get_intraday_data(
                            symbol=self.test_symbol,
                            current_time=mock_time
                        )
        
        # 验证返回空数据
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(len(result.ticks), 0)
        self.assertFalse(result.should_poll)  # 盘后不应轮询
        
    def test_get_intraday_data_rate_limit_returns_empty(self):
        """测试速率限制时返回空数据而不抛出异常"""
        mock_time = pd.Timestamp("2025-01-01 10:00:00")
        
        with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.determine_trading_phase',
                   return_value=TradingPhase.TRADING):
            with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.infer_market_from_symbol',
                       return_value=MarketCode.US):
                # 模拟速率限制异常
                with patch.object(self.provider, '_throttle_request'):
                    mock_ticker = MagicMock()
                    from yfinance.exceptions import YFRateLimitError
                    mock_ticker.history.side_effect = YFRateLimitError()
                    
                    with patch.object(self.provider.yf, 'Ticker', return_value=mock_ticker):
                        result = self.provider.get_intraday_data(
                            symbol=self.test_symbol,
                            current_time=mock_time
                        )
        
        # 验证返回空数据而不是抛出异常
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(len(result.ticks), 0)
        
    def test_get_intraday_data_success_with_valid_data(self):
        """测试成功获取有效数据"""
        # 🔧 使用美东时间，确保时区一致性
        import pytz
        us_eastern = pytz.timezone('America/New_York')
        mock_time = us_eastern.localize(pd.Timestamp("2024-12-31 10:30:00"))  # 美东时间盘中
        
        # 创建模拟的分时数据（从 09:30 开始）
        mock_df = pd.DataFrame({
            'Open': [150.0, 150.5],
            'High': [151.0, 151.5],
            'Low': [149.5, 150.0],
            'Close': [150.5, 151.0],
            'Volume': [1000, 1500]
        }, index=pd.DatetimeIndex([
            pd.Timestamp("2024-12-31 09:35:00"),  # 开盘后 5 分钟
            pd.Timestamp("2024-12-31 09:40:00")   # 开盘后 10 分钟
        ]))
        
        with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.determine_trading_phase',
                   return_value=TradingPhase.TRADING):
            with patch('core_bak_refactored.core.share.market.market_utils.MarketUtils.infer_market_from_symbol',
                       return_value=MarketCode.US):
                with patch.object(self.provider, '_throttle_request'):
                    mock_ticker = MagicMock()
                    mock_ticker.history.return_value = mock_df
                    
                    with patch.object(self.provider.yf, 'Ticker', return_value=mock_ticker):
                        result = self.provider.get_intraday_data(
                            symbol=self.test_symbol,
                            current_time=mock_time
                        )
        
        # 验证返回有效数据
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(len(result.ticks), 2)
        self.assertTrue(result.should_poll)  # 盘中应该轮询
        self.assertEqual(result.symbol, self.test_symbol)
        
    def test_generate_empty_intraday_data(self):
        """测试生成空分时数据对象"""
        trade_date = "2025-01-01"
        result = self.provider._generate_empty_intraday_data(
            symbol=self.test_symbol,
            trade_date=trade_date,
            should_poll=True
        )
        
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(result.symbol, self.test_symbol)
        self.assertEqual(len(result.ticks), 0)
        self.assertEqual(len(result.order_book_bids), 0)
        self.assertEqual(len(result.order_book_asks), 0)
        self.assertTrue(result.should_poll)
        # 验证提示信息（更新后的文案）
        self.assertIn("实时盘口数据", result.order_book_message)
        self.assertIn("逐笔成交", result.trade_records_message)
        
    def test_convert_yahoo_df_to_intraday(self):
        """测试将 Yahoo DataFrame 转换为 IntradayData"""
        trade_date = "2025-01-01"
        
        # 创建测试数据
        mock_df = pd.DataFrame({
            'Open': [150.0, 150.5, 151.0],
            'High': [151.0, 151.5, 152.0],
            'Low': [149.5, 150.0, 150.5],
            'Close': [150.5, 151.0, 151.5],
            'Volume': [1000, 1500, 2000]
        }, index=pd.DatetimeIndex([
            pd.Timestamp("2025-01-01 09:30:00"),
            pd.Timestamp("2025-01-01 09:35:00"),
            pd.Timestamp("2025-01-01 09:40:00")
        ]))
        
        result = self.provider._convert_yahoo_df_to_intraday(
            df=mock_df,
            symbol=self.test_symbol,
            trade_date=trade_date
        )
        
        self.assertIsInstance(result, IntradayData)
        self.assertEqual(len(result.ticks), 3)
        self.assertEqual(result.symbol, self.test_symbol)
        
        # 验证第一个tick
        first_tick = result.ticks[0]
        self.assertEqual(first_tick.time, "09:30:00")
        self.assertEqual(first_tick.price, 150.5)
        self.assertEqual(first_tick.volume, 1000)
        
        # 验证价格计算
        self.assertGreater(result.current_price, 0)
        self.assertGreater(result.yesterday_close, 0)


if __name__ == '__main__':
    unittest.main()
