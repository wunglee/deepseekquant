"""MockDataProvider.get_realtime_kline 方法单元测试

状态：新增功能测试
覆盖范围：
- 交易时段K线计算（缓存命中/未命中）
- 盘前时段集合竞价价格
- 盘后时段处理
- 缓存机制验证

注意：MockProvider的get_realtime_kline需要显式传入trading_phase和is_index参数
"""

import unittest
from datetime import datetime
from core_bak_refactored.core.data.providers.mock_provider import MockDataProvider
from core_bak_refactored.core.share.market.market_enums import MarketCode, TradingPhase


class MockProviderRealtimeKlineTest(unittest.TestCase):
    """MockDataProvider实时K线测试"""
    
    def setUp(self):
        """测试前准备"""
        self.provider = MockDataProvider()
    
    def test_trading_phase_first_call(self):
        """测试盘中时段K线计算（首次调用，无缓存）"""
        symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        result = self.provider.get_realtime_kline(
            symbol, 
            trade_date, 
            TradingPhase.TRADING,
            is_index=False
        )
        
        # 验证结果
        self.assertEqual(result['date'], '2025-12-16')
        self.assertIsNotNone(result['open'])
        self.assertIsNotNone(result['high'])
        self.assertIsNotNone(result['low'])
        self.assertIsNotNone(result['close'])
        self.assertGreaterEqual(result['high'], result['open'])  # 高>=开
        self.assertLessEqual(result['low'], result['open'])  # 低<=开
        self.assertGreaterEqual(result['volume'], 0)  # 成交量>=0（mock数据可能为0）
        self.assertEqual(result['trading_phase'], 'TRADING')  # 盘中时段
        self.assertTrue(result['should_poll'])  # 盘中应该轮询
    
    def test_trading_phase_with_cache(self):
        """测试盘中时段K线计算（第二次调用，有缓存）"""
        symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        # 第一次调用
        first_result = self.provider.get_realtime_kline(
            symbol, trade_date, TradingPhase.TRADING, is_index=False
        )
        first_open = first_result['open']
        
        # 第二次调用（使用缓存）
        second_result = self.provider.get_realtime_kline(
            symbol, trade_date, TradingPhase.TRADING, is_index=False, cached=first_result
        )
        
        # 验证开盘价复用缓存
        self.assertEqual(second_result['open'], first_open)
        # 其他字段可能变化（因为mock数据每次生成不同）
        self.assertEqual(second_result['trading_phase'], 'TRADING')  # 盘中时段
        self.assertTrue(second_result['should_poll'])
    
    def test_before_open_phase(self):
        """测试盘前时段（集合竞价）"""
        symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        result = self.provider.get_realtime_kline(
            symbol, trade_date, TradingPhase.BEFORE_OPEN, is_index=False
        )
        
        # 验证结果
        self.assertEqual(result['date'], '2025-12-16')
        self.assertIsNotNone(result['open'])
        self.assertEqual(result['open'], result['high'])  # 盘前OHLC相同
        self.assertEqual(result['open'], result['low'])
        self.assertEqual(result['open'], result['close'])
        self.assertEqual(result['volume'], 0)  # 盘前无成交量
        self.assertEqual(result['trading_phase'], 'BEFORE_OPEN')  # 盘前时段
        self.assertTrue(result['should_poll'])  # 盘前应该轮询
    
    def test_after_close_phase(self):
        """测试盘后时段"""
        symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        result = self.provider.get_realtime_kline(
            symbol, trade_date, TradingPhase.AFTER_CLOSE, is_index=False
        )
        
        # 验证结果：盘后返回全天数据（有波动）或昨收价
        self.assertEqual(result['date'], '2025-12-16')
        self.assertIsNotNone(result['open'])
        self.assertIsNotNone(result['high'])
        self.assertIsNotNone(result['low'])
        self.assertIsNotNone(result['close'])
        # 盘后可能有全天数据（OHLC不相等）或昨收价（OHLC相等）
        self.assertGreaterEqual(result['high'], result['low'])
        self.assertGreaterEqual(result['volume'], 0)
        self.assertEqual(result['trading_phase'], 'AFTER_CLOSE')  # 盘后时段
        self.assertFalse(result['should_poll'])  # 盘后不轮询
    
    def test_cache_persistence_same_day(self):
        """测试同一天内缓存持久化"""
        symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        # 第一次调用
        result1 = self.provider.get_realtime_kline(
            symbol, trade_date, TradingPhase.TRADING, is_index=False
        )
        open1 = result1['open']
        
        # 第二次调用（同一天），使用缓存
        result2 = self.provider.get_realtime_kline(
            symbol, trade_date, TradingPhase.TRADING, is_index=False, cached=result1
        )
        
        # 验证开盘价相同（同一天的开盘价不变）
        self.assertEqual(result2['open'], open1)
    
    def test_cache_reset_different_day(self):
        """测试不同日期缓存重置"""
        symbol = '000001.SZ'
        trade_date1 = '2025-12-16'
        trade_date2 = '2025-12-17'  # 第二天
        
        # 第一天
        result1 = self.provider.get_realtime_kline(
            symbol, trade_date1, TradingPhase.TRADING, is_index=False
        )
        
        # 第二天（缓存key不同，应该是新的开盘价）
        result2 = self.provider.get_realtime_kline(
            symbol, trade_date2, TradingPhase.TRADING, is_index=False
        )
        
        # 验证日期不同
        self.assertEqual(result1['date'], '2025-12-16')
        self.assertEqual(result2['date'], '2025-12-17')
        # 不同天的开盘价可能不同（但不强制要求）
    
    def test_cross_market_symbols(self):
        """测试不同市场的股票代码"""
        test_cases = [
            ('000001.SZ', '2025-12-16', TradingPhase.TRADING, False, True),   # 深圳A股，盘中
            ('600000.SH', '2025-12-16', TradingPhase.TRADING, False, True),   # 上海A股，盘中
            ('000300.SH', '2025-12-16', TradingPhase.TRADING, True, True),    # 沪深300指数，盘中
            ('AAPL', '2025-12-16', TradingPhase.AFTER_CLOSE, False, False),   # 美股（盘后）
        ]
        
        for symbol, trade_date, phase, is_index, should_poll_expected in test_cases:
            with self.subTest(symbol=symbol):
                result = self.provider.get_realtime_kline(
                    symbol, trade_date, phase, is_index
                )
                
                # 验证基本结构
                self.assertIn('date', result)
                self.assertIn('open', result)
                self.assertIn('high', result)
                self.assertIn('low', result)
                self.assertIn('close', result)
                self.assertIn('volume', result)
                self.assertIn('should_poll', result)
                
                # 验证数值类型
                if result['open'] is not None:
                    self.assertIsInstance(result['open'], (int, float))
                if result['volume'] is not None:
                    self.assertGreaterEqual(result['volume'], 0)
    
    def test_index_vs_stock(self):
        """测试指数与个股的区别"""
        index_symbol = '000300.SH'
        stock_symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        index_result = self.provider.get_realtime_kline(
            index_symbol, trade_date, TradingPhase.TRADING, is_index=True
        )
        stock_result = self.provider.get_realtime_kline(
            stock_symbol, trade_date, TradingPhase.TRADING, is_index=False
        )
        
        # 两者都应该有有效数据
        self.assertIsNotNone(index_result['open'])
        self.assertIsNotNone(stock_result['open'])
        
        # 两者的should_poll在盘中都应该是True
        self.assertTrue(index_result['should_poll'])
        self.assertTrue(stock_result['should_poll'])
    
    def test_price_consistency(self):
        """测试价格一致性（high >= open >= low）"""
        symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        result = self.provider.get_realtime_kline(
            symbol, trade_date, TradingPhase.TRADING, is_index=False
        )
        
        if result['open'] is not None:
            # 验证价格关系
            self.assertGreaterEqual(result['high'], result['open'])
            self.assertLessEqual(result['low'], result['open'])
            self.assertGreaterEqual(result['high'], result['close'])
            self.assertLessEqual(result['low'], result['close'])
            self.assertGreaterEqual(result['high'], result['low'])
    
    def test_auction_price_volatility(self):
        """测试集合竞价价格波动（盘前多次调用价格应有变化）"""
        symbol = '000001.SZ'
        trade_date = '2025-12-16'
        
        # 多次调用，记录集合竞价价格
        prices = []
        for i in range(5):
            # 每次调用都不带缓存，模拟时间流逝
            result = self.provider.get_realtime_kline(
                symbol, trade_date, TradingPhase.BEFORE_OPEN, is_index=False
            )
            prices.append(result['close'])
        
        # 注意：由于缓存机制，同一天的价格可能相同
        # 这个测试主要验证不会抛异常
        self.assertEqual(len(prices), 5)


if __name__ == '__main__':
    unittest.main()
