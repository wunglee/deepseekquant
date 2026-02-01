"""
事件分析模块测试

测试 EventConfig 和 EventAnalyzer 的功能
"""
import unittest
from unittest.mock import Mock

import pandas as pd

from core_bak_refactored.core.backtest.event_analysis import EventConfig, EventAnalyzer
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider


class EventAnalysisTest(unittest.TestCase):
    """事件分析测试"""
    
    def test_event_config_creation(self):
        """测试事件配置创建"""
        event = EventConfig(
            event_id='test_event',
            symbol='000300.SH',
            event_date='2020-02-20',
            event_type='market_crash',
            expected_decline=-0.20,
            market_id='CN'
        )
        
        self.assertEqual(event.event_id, 'test_event')
        self.assertEqual(event.symbol, '000300.SH')
        self.assertEqual(event.event_date, '2020-02-20')
        self.assertEqual(event.event_type, 'market_crash')
        self.assertEqual(event.expected_decline, -0.20)
        self.assertEqual(event.market_id, 'CN')
    
    def test_calculate_actual_return(self):
        """测试实际收益率计算"""
        # 准备测试数据
        df = pd.DataFrame({
            'close': [100, 110, 121, 115, 120],
            'date': pd.date_range('2020-01-01', periods=5, freq='D')
        })
        
        # 计算收益率
        actual_return = EventAnalyzer.calculate_actual_return(df)
        
        # 验证结果：从100到120，涨幅20%
        self.assertAlmostEqual(actual_return, 0.20, places=6)
    
    def test_calculate_actual_return_empty_data(self):
        """测试空数据的收益率计算"""
        df = pd.DataFrame()
        actual_return = EventAnalyzer.calculate_actual_return(df)
        self.assertEqual(actual_return, 0.0)
    
    def test_calculate_actual_return_insufficient_data(self):
        """测试数据不足的收益率计算"""
        df = pd.DataFrame({'close': [100]})
        actual_return = EventAnalyzer.calculate_actual_return(df)
        self.assertEqual(actual_return, 0.0)
    
    def test_calculate_prediction_error(self):
        """测试预测误差计算"""
        # 场景1：预测准确
        error1 = EventAnalyzer.calculate_prediction_error(-0.20, -0.20)
        self.assertAlmostEqual(error1, 0.0, places=6)
        
        # 场景2：小误差
        error2 = EventAnalyzer.calculate_prediction_error(-0.25, -0.20)
        self.assertAlmostEqual(error2, 0.05, places=6)
        
        # 场景3：大误差（裁剪到15%上限）
        error3 = EventAnalyzer.calculate_prediction_error(-0.40, -0.20)
        self.assertEqual(error3, 0.15)  # 裁剪到15%
    
    def test_safe_get_event_data_success(self):
        """测试安全获取事件数据成功"""
        provider = MockHistoricalDataProvider()
        event = EventConfig(
            event_id='test_event',
            symbol='000300.SH',
            event_date='2020-02-20',
            event_type='market_crash',
            expected_decline=-0.20,
            market_id='CN'
        )
        
        data, success = EventAnalyzer.safe_get_event_data(
            provider, event, window_days=10, baseline_days=30
        )
        
        self.assertTrue(success)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertIn('close', data.columns)
        self.assertGreater(len(data), 0)
    
    def test_safe_get_event_data_failure(self):
        """测试安全获取事件数据失败"""
        failing_provider = Mock()
        failing_provider.get_event_window_data.side_effect = ValueError("数据源连接失败")
        
        event = EventConfig(
            event_id='test_event',
            symbol='000300.SH',
            event_date='2020-02-20',
            event_type='market_crash',
            expected_decline=-0.20,
            market_id='CN'
        )
        
        data, success = EventAnalyzer.safe_get_event_data(
            failing_provider, event
        )
        
        self.assertFalse(success)
        self.assertTrue(data.empty)
    
    def test_safe_get_event_data_unexpected_format(self):
        """测试非预期格式数据"""
        unexpected_provider = Mock()
        unexpected_provider.get_event_window_data.return_value = "invalid_format"
        
        event = EventConfig(
            event_id='test_event',
            symbol='000300.SH',
            event_date='2020-02-20',
            event_type='market_crash',
            expected_decline=-0.20,
            market_id='CN'
        )
        
        data, success = EventAnalyzer.safe_get_event_data(
            unexpected_provider, event
        )
        
        self.assertFalse(success)
        self.assertTrue(data.empty)
    
    def test_analyze_event_full_workflow(self):
        """测试完整的事件分析工作流"""
        provider = MockHistoricalDataProvider()
        event = EventConfig(
            event_id='covid_19_pandemic',
            symbol='000300.SH',
            event_date='2020-02-20',
            event_type='pandemic',
            expected_decline=-0.20,
            market_id='CN'
        )
        
        # 1. 获取事件数据
        data, success = EventAnalyzer.safe_get_event_data(
            provider, event, window_days=30
        )
        self.assertTrue(success)
        
        # 2. 计算实际收益率
        actual_return = EventAnalyzer.calculate_actual_return(data)
        self.assertIsInstance(actual_return, float)
        
        # 3. 计算预测误差
        error = EventAnalyzer.calculate_prediction_error(
            actual_return, event.expected_decline
        )
        self.assertIsInstance(error, float)
        self.assertLessEqual(error, 0.15)  # 误差不超过15%上限


if __name__ == '__main__':
    unittest.main()
