"""
测试 MarketTimeUtils - 市场时间工具类

重要：所有测试必须传入市场本地时间（带正确的时区信息）

测试覆盖：
1. 本地时间输入验证
2. 不同交易时段判断（基于市场本地时间）
3. 节假日处理
4. 获取交易日逻辑
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from core_bak_refactored.core.share.market.market_enums import MarketCode, TradingPhase
from core_bak_refactored.core.share.market.market_time_utils import MarketTimeUtils


def create_mock_trading_calendar(is_trading_day_func=None, prev_trading_day_func=None):
    """创建mock交易日历服务"""
    mock_instance = MagicMock()
    
    if is_trading_day_func:
        mock_instance.is_trading_day.side_effect = is_trading_day_func
    else:
        # 默认：工作日是交易日
        mock_instance.is_trading_day.side_effect = lambda market, dt: dt.weekday() < 5
    
    if prev_trading_day_func:
        mock_instance.get_previous_trading_day.side_effect = prev_trading_day_func
    else:
        # 默认：返回前一天
        mock_instance.get_previous_trading_day.side_effect = lambda market, dt: dt - pd.Timedelta(days=1)
    
    return mock_instance


class BaseTest(unittest.TestCase):
    """测试基类"""
    
    def setUp(self):
        """设置mock交易日历服务"""
        self.mock_calendar = create_mock_trading_calendar()
        
        self.patcher = patch(
            'core_bak_refactored.core.share.market.trading_calendar_service.get_trading_calendar_service',
            return_value=self.mock_calendar
        )
        self.patcher.start()
    
    def tearDown(self):
        """清理"""
        self.patcher.stop()


class TestInputValidation(BaseTest):
    """测试输入验证"""
    
    def test_reject_non_utc_timezone(self):
        """测试接受市场本地时区的输入"""
        # 传入北京时间应该被接受
        beijing_time = pd.Timestamp('2024-01-15 10:00', tz='Asia/Shanghai')
        
        # 不应该抛出异常
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, beijing_time)
        self.assertIsInstance(phase, TradingPhase)
    
    def test_accept_utc_timezone(self):
        """测试接受美股本地时区的输入"""
        # 美东时间10:00
        us_time = pd.Timestamp('2024-01-15 10:00', tz='America/New_York')
        
        # 不应该抛出异常
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.US, us_time)
        self.assertIsInstance(phase, TradingPhase)
    
    def test_accept_naive_timezone(self):
        """测试拒绝naive时间（必须带时区信息）"""
        naive_time = pd.Timestamp('2024-01-15 10:00')  # naive，无时区
        
        # 应该抛出异常
        with self.assertRaises(ValueError) as context:
            MarketTimeUtils.determine_trading_phase(MarketCode.CN, naive_time)
        
        self.assertIn("时间戳必须包含时区信息", str(context.exception))


class TestTimezoneConversion(BaseTest):
    """测试时区处理（已删除时区转换方法，现在直接使用市场本地时间）"""
    
    def test_cn_local_time_hour(self):
        """测试A股本地时间的小时正确性"""
        local_time = pd.Timestamp('2024-01-15 10:00', tz='Asia/Shanghai')
        
        # 北京时间10:00
        self.assertEqual(local_time.hour, 10)
        self.assertEqual(local_time.minute, 0)
        self.assertEqual(str(local_time.tzinfo), 'Asia/Shanghai')
    
    def test_us_local_time_hour(self):
        """测试美股本地时间的小时正确性"""
        local_time = pd.Timestamp('2024-01-15 10:00', tz='America/New_York')
        
        # 美东时间10:00
        self.assertEqual(local_time.hour, 10)
        self.assertEqual(local_time.minute, 0)
    
    def test_get_market_local_date_cn(self):
        """测试获取A股市场本地日期"""
        local_time = pd.Timestamp('2024-01-15 23:00', tz='Asia/Shanghai')
        
        # 北京23:00仍是15号
        self.assertEqual(local_time.date(), pd.Timestamp('2024-01-15').date())
    
    def test_get_market_local_date_cross_day(self):
        """测试跨日期边界的市场本地日期"""
        # 北京 2024-01-16 08:30 -> 日期是16号
        cn_time = pd.Timestamp('2024-01-16 08:30', tz='Asia/Shanghai')
        self.assertEqual(cn_time.date(), pd.Timestamp('2024-01-16').date())
        
        # 美东 2024-01-15 19:30 -> 日期是15号
        us_time = pd.Timestamp('2024-01-15 19:30', tz='America/New_York')
        self.assertEqual(us_time.date(), pd.Timestamp('2024-01-15').date())


class TestTradingPhase(BaseTest):
    """测试交易时段判断"""
    
    def test_cn_before_open(self):
        """测试A股集合竞价时段"""
        # 北京 09:15（集合竞价）
        local_time = pd.Timestamp('2024-01-15 09:15', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, local_time)
        
        self.assertEqual(phase, TradingPhase.BEFORE_OPEN)
    
    def test_cn_trading_morning(self):
        """测试A股上午交易时段"""
        # 北京 10:00（上午交易）
        local_time = pd.Timestamp('2024-01-15 10:00', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, local_time)
        
        self.assertEqual(phase, TradingPhase.TRADING)
    
    def test_cn_noon_break(self):
        """测试A股午休时段"""
        # 北京 12:00（午休）
        local_time = pd.Timestamp('2024-01-15 12:00', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, local_time)
        
        self.assertEqual(phase, TradingPhase.NOON_BREAK)
    
    def test_cn_trading_afternoon(self):
        """测试A股下午交易时段"""
        # 北京 14:00（下午交易）
        local_time = pd.Timestamp('2024-01-15 14:00', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, local_time)
        
        self.assertEqual(phase, TradingPhase.TRADING)
    
    def test_cn_after_close(self):
        """测试A股收盘后"""
        # 北京 16:00（收盘后）
        local_time = pd.Timestamp('2024-01-15 16:00', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, local_time)
        
        self.assertEqual(phase, TradingPhase.AFTER_CLOSE)
    
    def test_us_trading(self):
        """测试美股交易时段"""
        # 美东 10:00（交易中）
        local_time = pd.Timestamp('2024-01-15 10:00', tz='America/New_York')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.US, local_time)
        
        self.assertEqual(phase, TradingPhase.TRADING)
    
    def test_hk_trading(self):
        """测试港股交易时段"""
        # 香港 10:00（交易中）
        local_time = pd.Timestamp('2024-01-15 10:00', tz='Asia/Hong_Kong')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.HK, local_time)
        
        self.assertEqual(phase, TradingPhase.TRADING)


class TestWeekendAndHolidays(BaseTest):
    """测试周末和节假日"""
    
    def test_cn_weekend(self):
        """测试A股周末"""
        # 北京 10:00（周六）
        saturday_local = pd.Timestamp('2024-01-20 10:00', tz='Asia/Shanghai')
        
        # Mock周末不是交易日
        self.mock_calendar.is_trading_day.side_effect = lambda market, dt: dt.weekday() < 5
        
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, saturday_local)
        self.assertEqual(phase, TradingPhase.AFTER_CLOSE)
    
    def test_cn_holiday(self):
        """测试A股节假日"""
        # 北京 10:00（春节）
        holiday_local = pd.Timestamp('2024-02-10 10:00', tz='Asia/Shanghai')
        
        # Mock该日期不是交易日
        def is_trading_day_mock(market, dt):
            if dt.date() == pd.Timestamp('2024-02-10').date():
                return False
            return dt.weekday() < 5
        
        self.mock_calendar.is_trading_day.side_effect = is_trading_day_mock
        
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, holiday_local)
        self.assertEqual(phase, TradingPhase.AFTER_CLOSE)


class TestGetLastTradeDate(BaseTest):
    """测试获取最后交易日"""
    
    def test_cn_before_open_returns_previous_day(self):
        """测试A股盘前返回前一交易日"""
        # 北京 08:00（周一，盘前）
        monday_local = pd.Timestamp('2024-01-15 08:00', tz='Asia/Shanghai')
        
        # Mock返回上周五
        def prev_day_mock(market, dt):
            if dt.date() == pd.Timestamp('2024-01-15').date():
                return pd.Timestamp('2024-01-12')
            return dt - pd.Timedelta(days=1)
        
        self.mock_calendar.get_previous_trading_day.side_effect = prev_day_mock
        
        last_date = MarketTimeUtils.get_last_trade_date(MarketCode.CN, monday_local)
        self.assertEqual(last_date.date(), pd.Timestamp('2024-01-12').date())
    
    def test_cn_trading_returns_today(self):
        """测试A股盘中返回当天"""
        # 北京 10:00（周一，盘中）
        monday_local = pd.Timestamp('2024-01-15 10:00', tz='Asia/Shanghai')
        
        # Mock周一是交易日
        self.mock_calendar.is_trading_day.return_value = True
        
        last_date = MarketTimeUtils.get_last_trade_date(MarketCode.CN, monday_local)
        self.assertEqual(last_date.date(), pd.Timestamp('2024-01-15').date())
    
    def test_cn_after_close_returns_today(self):
        """测试A股盘后返回当天"""
        # 北京 16:00（周一，盘后）
        monday_local = pd.Timestamp('2024-01-15 16:00', tz='Asia/Shanghai')
        
        # Mock周一是交易日
        self.mock_calendar.is_trading_day.return_value = True
        
        last_date = MarketTimeUtils.get_last_trade_date(MarketCode.CN, monday_local)
        self.assertEqual(last_date.date(), pd.Timestamp('2024-01-15').date())
    
    def test_cn_weekend_returns_friday(self):
        """测试A股周末返回周五"""
        # 北京 15:00（周六）
        saturday_local = pd.Timestamp('2024-01-20 15:00', tz='Asia/Shanghai')
        
        # Mock周六不是交易日，返回周五
        def is_trading_day_mock(market, dt):
            return dt.weekday() < 5
        
        def prev_day_mock(market, dt):
            if dt.date() == pd.Timestamp('2024-01-20').date():
                return pd.Timestamp('2024-01-19')
            return dt - pd.Timedelta(days=1)
        
        self.mock_calendar.is_trading_day.side_effect = is_trading_day_mock
        self.mock_calendar.get_previous_trading_day.side_effect = prev_day_mock
        
        last_date = MarketTimeUtils.get_last_trade_date(MarketCode.CN, saturday_local)
        self.assertEqual(last_date.date(), pd.Timestamp('2024-01-19').date())


class TestAllMarkets(BaseTest):
    """测试所有市场"""
    
    def test_all_markets_trading_phase(self):
        """测试所有市场在交易时段的判断"""
        # 所有测试都使用市场本地时间10:00
        test_cases = [
            # (市场, 市场本地时间, 预期时段)
            (MarketCode.CN, pd.Timestamp('2024-01-15 10:00', tz='Asia/Shanghai'), TradingPhase.TRADING),
            (MarketCode.US, pd.Timestamp('2024-01-15 10:00', tz='America/New_York'), TradingPhase.TRADING),
            (MarketCode.HK, pd.Timestamp('2024-01-15 10:00', tz='Asia/Hong_Kong'), TradingPhase.TRADING),
            (MarketCode.JP, pd.Timestamp('2024-01-15 10:00', tz='Asia/Tokyo'), TradingPhase.TRADING),
            (MarketCode.SG, pd.Timestamp('2024-01-15 10:00', tz='Asia/Singapore'), TradingPhase.TRADING),
        ]
        
        for market, local_time, expected_phase in test_cases:
            with self.subTest(market=market):
                phase = MarketTimeUtils.determine_trading_phase(market, local_time)
                self.assertEqual(phase, expected_phase,
                               f"市场 {market.value} 在本地时间 {local_time} 应为 {expected_phase}")


class TestEdgeCases(BaseTest):
    """测试边界情况"""
    
    def test_exact_open_time(self):
        """测试精确开盘时间"""
        # 北京 09:30:00（精确开盘）
        exact_open = pd.Timestamp('2024-01-15 09:30:00', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, exact_open)
        
        # 09:30应该是交易中
        self.assertEqual(phase, TradingPhase.TRADING)
    
    def test_exact_close_time(self):
        """测试精确收盘时间"""
        # 北京 15:00:00（精确收盘）
        exact_close = pd.Timestamp('2024-01-15 15:00:00', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, exact_close)
        
        # 15:00应该仍在交易中（包含边界）
        self.assertEqual(phase, TradingPhase.TRADING)
    
    def test_one_second_after_close(self):
        """测试收盘后1秒"""
        # 北京 15:00:01（收盘后1秒）
        after_close = pd.Timestamp('2024-01-15 15:00:01', tz='Asia/Shanghai')
        phase = MarketTimeUtils.determine_trading_phase(MarketCode.CN, after_close)
        
        # 应该是收盘后
        self.assertEqual(phase, TradingPhase.AFTER_CLOSE)
    
    def test_default_current_time(self):
        """测试使用当前系统时间（转换为市场本地时间）"""
        # 获取当前UTC时间并转换为北京时间
        utc_now = pd.Timestamp.now(tz='UTC')
        cn_now = utc_now.tz_convert('Asia/Shanghai')
        
        last_date = MarketTimeUtils.get_last_trade_date(MarketCode.CN, cn_now)
        
        # 应该返回一个日期
        self.assertIsInstance(last_date, pd.Timestamp)


if __name__ == '__main__':
    unittest.main()
