"""ChartDataAssembler 单元测试

符合规范：
- 测试文件命名: chart_data_test.py (对应 chart_data.py)
- 目录镜像: tests/units/app/quality_monitoring/api/ 对应 app/quality_monitoring/api/
"""

import unittest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np

from pandas import DataFrame

from core_bak_refactored.app.quality_monitoring.api.chart_data import ChartDataAssembler
from core_bak_refactored.core.data.providers.protocols import PriceData


class ChartDataAssemblerBasicTest(unittest.TestCase):
    """ChartDataAssembler 基础功能测试"""
    
    def setUp(self):
        """测试前准备"""
        # Mock数据提供者
        self.mock_provider = Mock()
        
        # Mock技术指标服务
        self.mock_indicator = Mock()
        
        # 创建组装器实例
        self.assembler = ChartDataAssembler(
            data_provider=self.mock_provider,
            indicator_service=self.mock_indicator
        )
    
    def test_safe_float_normal_value(self):
        """测试安全浮点数转换 - 正常值"""
        result = self.assembler._safe_float(123.45)
        self.assertEqual(result, 123.45)
    
    def test_safe_float_nan_value(self):
        """测试安全浮点数转换 - NaN值"""
        result = self.assembler._safe_float(np.nan)
        self.assertIsNone(result)
    
    def test_safe_float_none_value(self):
        """测试安全浮点数转换 - None值"""
        result = self.assembler._safe_float(None)
        self.assertIsNone(result)
    
    def test_safe_float_string_value(self):
        """测试安全浮点数转换 - 字符串数值"""
        result = self.assembler._safe_float("123.45")
        self.assertEqual(result, 123.45)
    
    def test_safe_float_invalid_value(self):
        """测试安全浮点数转换 - 无效值"""
        result = self.assembler._safe_float("invalid")
        self.assertIsNone(result)


class ChartDataAssemblerPeriodConversionTest(unittest.TestCase):
    """周期转换功能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_provider = Mock()
        self.mock_indicator = Mock()
        self.assembler = ChartDataAssembler(
            data_provider=self.mock_provider,
            indicator_service=self.mock_indicator
        )
    
    def test_convert_period_daily(self):
        """测试日线转换 - 无需转换"""
        # 准备测试数据
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        price_data = PriceData.from_dataframe(DataFrame({
            'date': dates,
            'open': np.random.rand(10) * 100,
            'high': np.random.rand(10) * 100,
            'low': np.random.rand(10) * 100,
            'close': np.random.rand(10) * 100,
            'volume': np.random.rand(10) * 1000000
        }))
        
        result = self.assembler._convert_period(price_data, 'daily', 10)
        
        # 验证结果
        pd_result=result.to_dataframe()
        self.assertEqual(len(pd_result), 10)
        self.assertIn('open', pd_result.columns)
        self.assertIn('date', pd_result.columns)
    
    def test_convert_period_weekly(self):
        """测试日线转周线"""
        # 准备30天的日线数据
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.arange(30, dtype=float) + 100,
            'high': np.arange(30, dtype=float) + 110,
            'low': np.arange(30, dtype=float) + 90,
            'close': np.arange(30, dtype=float) + 105,
            'volume': np.ones(30) * 1000000
        })
        
        result = self.assembler._convert_period(PriceData.from_dataframe(df), 'weekly', 4)
        pd_result=result.to_dataframe()
        # 验证结果（30天约4-5周）
        self.assertLessEqual(len(pd_result), 5)
        self.assertIn('open', pd_result.columns)
    
    def test_convert_period_monthly(self):
        """测试日线转月线"""
        # 准备90天的日线数据
        dates = pd.date_range('2024-01-01', periods=90, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.arange(90, dtype=float) + 100,
            'high': np.arange(90, dtype=float) + 110,
            'low': np.arange(90, dtype=float) + 90,
            'close': np.arange(90, dtype=float) + 105,
            'volume': np.ones(90) * 1000000
        })
        
        result = self.assembler._convert_period(PriceData.from_dataframe(df), 'monthly', 3)
        pd_result=result.to_dataframe()
        # 验证结果（90天=3个月）
        self.assertEqual(len(pd_result), 3)
        self.assertIn('open', pd_result.columns)


class ChartDataAssemblerEventDetectionTest(unittest.TestCase):
    """市场事件检测测试"""
    
    def setUp(self):
        """测试前准备"""
        self.mock_provider = Mock()
        self.mock_indicator = Mock()
        self.assembler = ChartDataAssembler(
            data_provider=self.mock_provider,
            indicator_service=self.mock_indicator
        )
    
    def test_detect_events_crash(self):
        """测试暴跌事件检测"""
        # 构造包含暴跌的数据
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100.0, 100.0, 93.0, 93.0, 93.0]  # 第3天暴跌7%
        })
        
        events = self.assembler._detect_events(df)
        
        # 验证事件检测
        crash_events = [e for e in events if e['type'] == 'market_crash']
        self.assertGreater(len(crash_events), 0)
        
        # 验证事件属性
        event = crash_events[0]
        self.assertEqual(event['impact'], 'negative')
        self.assertIn('severity', event)
        self.assertLess(event['decline_pct'], 0)
    
    def test_detect_events_rally(self):
        """测试暴涨事件检测"""
        # 构造包含暴涨的数据
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100.0, 100.0, 106.0, 106.0, 106.0]  # 第3天暴涨6%
        })
        
        events = self.assembler._detect_events(df)
        
        # 验证事件检测
        rally_events = [e for e in events if e['type'] == 'rally']
        self.assertGreater(len(rally_events), 0)
        
        # 验证事件属性
        event = rally_events[0]
        self.assertEqual(event['impact'], 'positive')
        self.assertGreater(event['rise_pct'], 0)
    
    def test_detect_events_no_extreme(self):
        """测试无极端事件"""
        # 构造正常波动数据
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'close': [100.0, 101.0, 102.0, 101.5, 102.5]  # 正常波动
        })
        
        events = self.assembler._detect_events(df)
        
        # 验证无事件
        self.assertEqual(len(events), 0)


class ChartDataAssemblerIntegrationTest(unittest.TestCase):
    """集成测试 - 完整数据组装流程"""
    
    def setUp(self):
        """测试前准备"""
        # Mock数据提供者返回模拟数据
        self.mock_provider = Mock()
        self.mock_provider.get_index_prices = Mock(return_value=self._create_mock_data())
        
        # Mock技术指标服务
        self.mock_indicator = Mock()
        self._setup_indicator_mocks()
        
        self.assembler = ChartDataAssembler(
            data_provider=self.mock_provider,
            indicator_service=self.mock_indicator
        )
    
    def _create_mock_data(self):
        """创建模拟K线数据"""
        dates = pd.date_range('2024-01-01', periods=120, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(120) * 5,
            'high': 105 + np.random.randn(120) * 5,
            'low': 95 + np.random.randn(120) * 5,
            'close': 100 + np.random.randn(120) * 5,
            'volume': 1000000 + np.random.randn(120) * 100000
        })
        return df
    
    def _setup_indicator_mocks(self):
        """设置指标服务的Mock返回值"""
        # MACD
        self.mock_indicator.calculate_macd = Mock(return_value=(
            pd.Series([0.5] * 120),
            pd.Series([0.3] * 120),
            pd.Series([0.2] * 120)
        ))
        
        # RSI
        self.mock_indicator.calculate_rsi = Mock(return_value=pd.Series([60.0] * 120))
        
        # KDJ
        self.mock_indicator.calculate_kdj = Mock(return_value=(
            pd.Series([70.0] * 120),
            pd.Series([65.0] * 120)
        ))
        
        # OBV
        self.mock_indicator.calculate_obv = Mock(return_value=pd.Series([5000000] * 120))
    
    def test_assemble_chart_data_success(self):
        """测试完整数据组装 - 成功场景"""
        result = self.assembler.assemble_chart_data(
            index_id='000001.SH',
            period='daily',
            count=120,
            before=None,
            indicators='all'
        )
        
        # 验证返回结构
        self.assertIn('kline', result)
        self.assertIn('indicators', result)
        self.assertIn('events', result)
        
        # 验证K线数据
        self.assertGreater(len(result['kline']), 0)
        first_kline = result['kline'][0]
        self.assertIn('date', first_kline)
        self.assertIn('open', first_kline)
        self.assertIn('ma5', first_kline)
        
        # 验证指标数据
        self.assertIn('vol', result['indicators'])
        self.assertIn('macd', result['indicators'])
        self.assertIn('rsi', result['indicators'])
        self.assertIn('kdj', result['indicators'])
        self.assertIn('obv', result['indicators'])
    
    def test_assemble_chart_data_partial_indicators(self):
        """测试部分指标组装"""
        result = self.assembler.assemble_chart_data(
            index_id='000001.SH',
            period='daily',
            count=120,
            indicators='macd,rsi'
        )
        
        # 验证仅包含请求的指标
        self.assertIn('macd', result['indicators'])
        self.assertIn('rsi', result['indicators'])
        self.assertNotIn('kdj', result['indicators'])
        self.assertNotIn('obv', result['indicators'])


if __name__ == '__main__':
    unittest.main()
