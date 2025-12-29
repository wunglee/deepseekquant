"""
测试 MarketUtils 工具类
"""

import unittest


import pandas as pd
from unittest.mock import patch, MagicMock

from core_bak_refactored.core.share.market.market_utils import MarketUtils
from core_bak_refactored.core.share.market.market_enums import MarketCode, TradingPhase


class TestMarketUtils(unittest.TestCase):
    """测试 MarketUtils 工具类"""

    def test_is_index(self):
        """测试证券类型判断"""
        # 测试上海指数
        self.assertTrue(MarketUtils.is_index('000001.SH'))  # 上证指数
        self.assertTrue(MarketUtils.is_index('000300.SH'))  # 沪深300
        
        # 测试深圳指数
        self.assertTrue(MarketUtils.is_index('399001.SZ'))  # 深证成指
        self.assertTrue(MarketUtils.is_index('399006.SZ'))  # 创业板指
        
        # 测试个股
        self.assertFalse(MarketUtils.is_index('600000.SH'))  # 浦发银行
        self.assertFalse(MarketUtils.is_index('000001.SZ'))  # 平安银行
        self.assertFalse(MarketUtils.is_index('300001.SZ'))  # 特锐德
        
        # 测试美股和其它
        self.assertFalse(MarketUtils.is_index('^GSPC'))  # 标普500（美股）
        self.assertFalse(MarketUtils.is_index('AAPL'))    # 苹果（美股）
        
        # 测试边界情况
        self.assertFalse(MarketUtils.is_index(''))
        self.assertFalse(MarketUtils.is_index(None))
    
    def test_standardize_format_valid_data(self):
        """测试有效数据标准化"""
        # 创建测试数据（A股格式）
        df = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-02', '2023-01-01'],  # 乱序
            'open': [100.0, 99.0, 98.0],
            'high': [102.0, 101.0, 100.0],
            'low': [98.0, 97.0, 96.0],
            'close': [101.0, 100.0, 99.0],
            'volume': [1000, 1100, 1200]
        })
        
        result = MarketUtils.standardize_format(df)
        
        # 验证数据已排序（按时间升序）
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['close'], 99.0)  # 2023-01-01
        self.assertEqual(result.iloc[1]['close'], 100.0)  # 2023-01-02
        self.assertEqual(result.iloc[2]['close'], 101.0)  # 2023-01-03
        
        # 验证列名标准化
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(result.columns), expected_columns)
    
    def test_standardize_format_chinese_columns(self):
        """测试中文列名数据标准化"""
        # 创建测试数据（港股/美股格式）
        df = pd.DataFrame({
            '日期': ['2023-01-03', '2023-01-02', '2023-01-01'],
            '开盘': [100.0, 99.0, 98.0],
            '最高': [102.0, 101.0, 100.0],
            '最低': [98.0, 97.0, 96.0],
            '收盘': [101.0, 100.0, 99.0],
            '成交量': [1000, 1100, 1200]
        })
        
        result = MarketUtils.standardize_format(df)
        
        # 验证数据已排序（按时间升序）
        self.assertEqual(len(result), 3)
        self.assertEqual(result.iloc[0]['close'], 99.0)  # 2023-01-01
        self.assertEqual(result.iloc[1]['close'], 100.0)  # 2023-01-02
        self.assertEqual(result.iloc[2]['close'], 101.0)  # 2023-01-03
        
        # 验证列名标准化
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(result.columns), expected_columns)
    
    def test_standardize_format_missing_columns(self):
        """测试缺少列的数据标准化"""
        # 创建测试数据（缺少open/high/low列）
        df = pd.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'close': [99.0, 100.0, 101.0],
            'volume': [1000, 1100, 1200]
        })
        
        result = MarketUtils.standardize_format(df)
        
        # 验证缺失列使用close填充
        self.assertEqual(result.iloc[0]['open'], 99.0)
        self.assertEqual(result.iloc[0]['high'], 99.0)
        self.assertEqual(result.iloc[0]['low'], 99.0)
        
        # 验证列名标准化
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(result.columns), expected_columns)

    def test_standardize_format_missing_date_close_columns(self):
        """测试缺少日期或收盘价列的数据标准化"""
        # 创建测试数据（缺少date列）
        df = pd.DataFrame({
            'open': [100.0, 99.0, 98.0],
            'high': [102.0, 101.0, 100.0],
            'low': [98.0, 97.0, 96.0],
            'close': [101.0, 100.0, 99.0],
            'volume': [1000, 1100, 1200]
        })
        
        # 应该抛出ValueError
        with self.assertRaises(ValueError) as context:
            MarketUtils.standardize_format(df)
        
        self.assertIn("Cannot find date or close columns", str(context.exception))
    
    def test_standardize_format_to_price_data_valid_data(self):
        """测试有效数据标准化为PriceData"""
        # 创建测试数据（A股格式）
        df = pd.DataFrame({
            'date': ['2023-01-03', '2023-01-02', '2023-01-01'],  # 乱序
            'open': [100.0, 99.0, 98.0],
            'high': [102.0, 101.0, 100.0],
            'low': [98.0, 97.0, 96.0],
            'close': [101.0, 100.0, 99.0],
            'volume': [1000, 1100, 1200]
        })
        
        result = MarketUtils.standardize_format_to_price_data(df, "TEST")
        
        # 验证返回类型
        from core_bak_refactored.core.data.providers.protocols import PriceData
        self.assertIsInstance(result, PriceData)
        
        # 验证数据已排序（按时间升序）
        self.assertEqual(len(result.records), 3)
        self.assertEqual(result.records[0].close, 99.0)  # 2023-01-01
        self.assertEqual(result.records[1].close, 100.0)  # 2023-01-02
        self.assertEqual(result.records[2].close, 101.0)  # 2023-01-03
        
        # 验证元数据
        self.assertEqual(result.symbol, "TEST")
        self.assertEqual(result.count, 3)
    
    def test_standardize_format_to_price_data_multiindex_columns(self):
        """测试MultiIndex列名数据标准化为PriceData"""
        # 创建测试数据（Yahoo Finance格式）
        columns = pd.MultiIndex.from_tuples([
            ('Open', 'AAPL'), ('High', 'AAPL'), ('Low', 'AAPL'), 
            ('Close', 'AAPL'), ('Volume', 'AAPL')
        ])
        df = pd.DataFrame([
            [100.0, 105.0, 99.0, 104.0, 1000],
            [101.0, 106.0, 100.0, 105.0, 1100]
        ], columns=columns, index=pd.date_range('2023-01-01', periods=2))
        
        result = MarketUtils.standardize_format_to_price_data(df, "AAPL")
        
        # 验证返回类型
        from core_bak_refactored.core.data.providers.protocols import PriceData
        self.assertIsInstance(result, PriceData)
        
        # 验证数据
        self.assertEqual(len(result.records), 2)
        self.assertEqual(result.records[0].close, 104.0)
        self.assertEqual(result.records[1].close, 105.0)
        
        # 验证元数据
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.count, 2)
    
    def test_standardize_format_to_price_data_empty_data(self):
        """测试空数据标准化为PriceData"""
        # 创建空的DataFrame
        df = pd.DataFrame()
        
        result = MarketUtils.standardize_format_to_price_data(df, "EMPTY")
        
        # 验证返回类型
        from core_bak_refactored.core.data.providers.protocols import PriceData
        self.assertIsInstance(result, PriceData)
        
        # 验证空数据
        self.assertEqual(len(result.records), 0)
        self.assertEqual(result.symbol, "EMPTY")
        self.assertEqual(result.count, 0)
    
    @patch('core_bak_refactored.core.share.config_manager.ConfigManager')
    def test_get_last_trade_date_weekend(self, mock_config_manager):
        """测试周末获取最后交易日"""
        # 模拟配置管理器
        mock_config = MagicMock()
        mock_config.get_trading_hours.return_value = {
            'open': '09:30',
            'close': '15:00'
        }
        mock_config_manager.return_value = mock_config
        
        # 测试周六
        result = MarketUtils.get_last_trade_date(MarketCode.CN, '2023-01-07')  # 周六
        self.assertEqual(result, '2023-01-06')  # 应该返回周五
        
        # 测试周日
        result = MarketUtils.get_last_trade_date(MarketCode.CN, '2023-01-08')  # 周日
        self.assertEqual(result, '2023-01-06')  # 应该返回周五

    @patch('core_bak_refactored.core.share.config_manager.ConfigManager')
    def test_get_last_trade_date_workday_before_open(self, mock_config_manager):
        """测试工作日盘前获取最后交易日"""
        # 模拟配置管理器
        mock_config = MagicMock()
        mock_config.get_trading_hours.return_value = {
            'open': '09:30',
            'close': '15:00'
        }
        mock_config_manager.return_value = mock_config
        
        # 模拟时间在盘前
        mock_now = MagicMock()
        mock_now.time.return_value = dt_time(8, 0)  # 08:00
        
        # 测试普通工作日盘前（传入current_time参数）
        result = MarketUtils.get_last_trade_date(MarketCode.CN, '2023-01-05', current_time=mock_now)  # 星期四盘前
        self.assertEqual(result, '2023-01-04')  # 应该返回昨天（星期三）
        
        # 测试周一盘前
        result = MarketUtils.get_last_trade_date(MarketCode.CN, '2023-01-02', current_time=mock_now)  # 星期一盘前
        self.assertEqual(result, '2022-12-30')  # 应该返回上周五

    @patch('core_bak_refactored.core.share.config_manager.ConfigManager')
    def test_get_last_trade_date_workday_after_open(self, mock_config_manager):
        """测试工作日盘中/盘后获取最后交易日"""
        # 模拟配置管理器
        mock_config = MagicMock()
        mock_config.get_trading_hours.return_value = {
            'open': '09:30',
            'close': '15:00'
        }
        mock_config_manager.return_value = mock_config
        
        # 模拟时间在盘中
        mock_now = MagicMock()
        mock_now.time.return_value = dt_time(10, 30)  # 10:30
        
        # 测试工作日盘中（传入current_time参数）
        result = MarketUtils.get_last_trade_date(MarketCode.CN, '2023-01-05', current_time=mock_now)  # 星期四盘中
        self.assertEqual(result, '2023-01-05')  # 应该返回当天
        
        # 模拟时间在盘后
        mock_now.time.return_value = dt_time(16, 0)  # 16:00
        
        # 测试工作日盘后（传入current_time参数）
        result = MarketUtils.get_last_trade_date(MarketCode.CN, '2023-01-05', current_time=mock_now)  # 星期四盘后
        self.assertEqual(result, '2023-01-05')  # 应该返回当天

    def test_standardize_format_with_symbol_parameter(self):
        """测试standardize_format方法的symbol参数"""
        # 创建测试数据（Yahoo Finance格式）
        columns = pd.MultiIndex.from_tuples([
            ('Open', 'AAPL'), ('High', 'AAPL'), ('Low', 'AAPL'), 
            ('Close', 'AAPL'), ('Volume', 'AAPL')
        ])
        df = pd.DataFrame([
            [100.0, 105.0, 99.0, 104.0, 1000],
            [101.0, 106.0, 100.0, 105.0, 1100]
        ], columns=columns, index=pd.date_range('2023-01-01', periods=2))
        
        result = MarketUtils.standardize_format(df, "AAPL")
        
        # 验证数据
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['close'], 104.0)
        self.assertEqual(result.iloc[1]['close'], 105.0)
        
        # 验证列名标准化
        expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(result.columns), expected_columns)
    
    @patch('core_bak_refactored.core.share.config_manager.ConfigManager')
    def test_determine_trading_phase_weekend(self, mock_config_manager):
        """测试周末交易时段判断"""
        # 模拟配置管理器
        mock_config = MagicMock()
        mock_config.get_trading_hours.return_value = {
            'open': '09:30',
            'close': '15:00'
        }
        mock_config_manager.return_value = mock_config
        
        # 测试周六
        saturday = datetime(2023, 1, 7, 10, 0)  # 周六 10:00
        result = MarketUtils.determine_trading_phase(MarketCode.CN, saturday)
        self.assertEqual(result, TradingPhase.AFTER_CLOSE)
        
        # 测试周日
        sunday = datetime(2023, 1, 8, 15, 0)  # 周日 15:00
        result = MarketUtils.determine_trading_phase(MarketCode.CN, sunday)
        self.assertEqual(result, TradingPhase.AFTER_CLOSE)

    @patch('core_bak_refactored.core.share.config_manager.ConfigManager')
    def test_determine_trading_phase_weekday(self, mock_config_manager):
        """测试工作日交易时段判断"""
        # 模拟配置管理器
        mock_config = MagicMock()
        mock_config.get_trading_hours.return_value = {
            'open': '09:30',
            'close': '15:00'
        }
        mock_config_manager.return_value = mock_config
        
        # 测试集合竞价时段 (09:00-09:30)
        call_auction_time = datetime(2023, 1, 4, 9, 15)  # 周三 09:15
        result = MarketUtils.determine_trading_phase(MarketCode.CN, call_auction_time)
        self.assertEqual(result, TradingPhase.BEFORE_OPEN)
        
        # 测试交易时段 (09:30-15:00)
        trading_time = datetime(2023, 1, 4, 10, 30)  # 周三 10:30
        result = MarketUtils.determine_trading_phase(MarketCode.CN, trading_time)
        self.assertEqual(result, TradingPhase.TRADING)
        
        # 测试盘后时段 (15:00之后)
        after_close_time = datetime(2023, 1, 4, 16, 0)  # 周三 16:00
        result = MarketUtils.determine_trading_phase(MarketCode.CN, after_close_time)
        self.assertEqual(result, TradingPhase.AFTER_CLOSE)

    @patch('core_bak_refactored.core.share.config_manager.ConfigManager')
    def test_determine_trading_phase_with_custom_hours(self, mock_config_manager):
        """测试使用自定义交易时间的交易时段判断"""
        # 模拟配置管理器返回自定义交易时间
        mock_config = MagicMock()
        mock_config.get_trading_hours.return_value = {
            'open': '09:00',
            'close': '16:00'
        }
        mock_config_manager.return_value = mock_config
        
        # 测试自定义开盘时间前的集合竞价时段
        call_auction_time = datetime(2023, 1, 4, 8, 45)  # 08:45 (开盘前30分钟)
        result = MarketUtils.determine_trading_phase(MarketCode.CN, call_auction_time)
        self.assertEqual(result, TradingPhase.BEFORE_OPEN)
        
        # 测试自定义交易时段
        trading_time = datetime(2023, 1, 4, 10, 0)  # 10:00
        result = MarketUtils.determine_trading_phase(MarketCode.CN, trading_time)
        self.assertEqual(result, TradingPhase.TRADING)
        
        # 测试自定义收盘时间后
        after_close_time = datetime(2023, 1, 4, 17, 0)  # 17:00
        result = MarketUtils.determine_trading_phase(MarketCode.CN, after_close_time)
        self.assertEqual(result, TradingPhase.AFTER_CLOSE)


class TestGetLastTradeDateFix(unittest.TestCase):
    """测试 get_last_trade_date 方法的修复"""
    
    def test_monday_before_open(self):
        """测试周一凌晨盘前（应返回上周五）"""
        now = datetime(2025, 12, 15, 4, 49)  # 周一凌晨04:49
        trade_date = '2025-12-15'
        result = MarketUtils.get_last_trade_date(MarketCode.CN, trade_date, now)
        
        # 周一盘前应该返回上周五
        self.assertEqual(result, '2025-12-12', 
                        f"周一凌晨04:49盘前应返回上周五2025-12-12，实际返回{result}")
    
    def test_monday_during_trading(self):
        """测试周一盘中（应返回今天）"""
        now = datetime(2025, 12, 15, 10, 0)  # 周一10:00盘中
        trade_date = '2025-12-15'
        result = MarketUtils.get_last_trade_date(MarketCode.CN, trade_date, now)
        
        # 周一盘中应该返回今天
        self.assertEqual(result, '2025-12-15',
                        f"周一10:00盘中应返回今天2025-12-15，实际返回{result}")
    
    def test_monday_after_close(self):
        """测试周一盘后（应返回今天）"""
        now = datetime(2025, 12, 15, 16, 0)  # 周一16:00盘后
        trade_date = '2025-12-15'
        result = MarketUtils.get_last_trade_date(MarketCode.CN, trade_date, now)
        
        # 周一盘后应该返回今天
        self.assertEqual(result, '2025-12-15',
                        f"周一16:00盘后应返回今天2025-12-15，实际返回{result}")
    
    def test_saturday(self):
        """测试周六（应返回周五）"""
        now = datetime(2025, 12, 13, 20, 0)  # 周六20:00
        trade_date = '2025-12-13'
        result = MarketUtils.get_last_trade_date(MarketCode.CN, trade_date, now)
        
        # 周六应该返回上个交易日（周五）
        self.assertEqual(result, '2025-12-12',
                        f"周六应返回周五2025-12-12，实际返回{result}")
    
    def test_tuesday_before_open(self):
        """测试周二盘前（应返回昨天/周一）"""
        now = datetime(2025, 12, 16, 8, 0)  # 周二08:00盘前
        trade_date = '2025-12-16'
        result = MarketUtils.get_last_trade_date(MarketCode.CN, trade_date, now)
        
        # 周二盘前应该返回昨天（周一）
        self.assertEqual(result, '2025-12-15',
                        f"周二08:00盘前应返回昨天2025-12-15，实际返回{result}")


if __name__ == '__main__':
    unittest.main()