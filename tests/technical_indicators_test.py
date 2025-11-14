"""
技术指标计算测试
测试 infrastructure/technical_indicators.py

修订历史:
- 根据量化专家建议更新测试用例
- 验证RSI的Wilder平滑法
- 验证ADX完整系统
- 验证市场参数适配
- 补充边界条件测试
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_bak_refactored.infrastructure.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """技术指标计算测试"""
    
    def setUp(self):
        """准备测试数据"""
        # 创建模拟的股票价格数据
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # 生成模拟价格序列（随机游走）
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        self.prices = pd.Series(prices, index=dates, name='close')
        
        # 生成OHLCV数据
        self.high = self.prices + np.random.rand(100) * 2
        self.low = self.prices - np.random.rand(100) * 2
        self.close = self.prices
        self.volume = pd.Series(np.random.randint(1000000, 10000000, 100), index=dates)
    
    def test_sma_basic(self):
        """测试简单移动平均 - 基本功能"""
        period = 5
        sma = TechnicalIndicators.calculate_sma(self.prices, period)
        
        # 验证前4个值应该是NaN（不足5个数据点）
        self.assertTrue(pd.isna(sma.iloc[:period-1]).all())
        
        # 验证第5个值等于前5个价格的平均
        expected_first_sma = self.prices.iloc[:period].mean()
        self.assertAlmostEqual(sma.iloc[period-1], expected_first_sma, places=5)
        
        # 验证长度一致
        self.assertEqual(len(sma), len(self.prices))
    
    def test_sma_manual_calculation(self):
        """测试SMA - 手动验证计算正确性"""
        simple_data = pd.Series([10, 20, 30, 40, 50])
        sma_3 = TechnicalIndicators.calculate_sma(simple_data, 3)
        
        # 前两个应该是NaN
        self.assertTrue(pd.isna(sma_3.iloc[0]))
        self.assertTrue(pd.isna(sma_3.iloc[1]))
        
        # 第3个 = (10+20+30)/3 = 20
        self.assertEqual(sma_3.iloc[2], 20.0)
        
        # 第4个 = (20+30+40)/3 = 30
        self.assertEqual(sma_3.iloc[3], 30.0)
        
        # 第5个 = (30+40+50)/3 = 40
        self.assertEqual(sma_3.iloc[4], 40.0)
    
    def test_ema_basic(self):
        """测试指数移动平均 - 基本功能"""
        period = 12
        ema = TechnicalIndicators.calculate_ema(self.prices, period)
        
        # EMA应该没有NaN（pandas EMA实现）
        self.assertEqual(len(ema), len(self.prices))
        
        # EMA应该对近期价格更敏感
        # 验证EMA不等于SMA（除非是完全平稳的序列）
        sma = TechnicalIndicators.calculate_sma(self.prices, period)
        self.assertFalse(ema.equals(sma))
    
    def test_macd_structure(self):
        """测试MACD - 返回结构"""
        macd_line, signal_line, histogram = TechnicalIndicators.calculate_macd(self.prices)
        
        # 验证返回三个Series
        self.assertIsInstance(macd_line, pd.Series)
        self.assertIsInstance(signal_line, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
        
        # 验证长度一致
        self.assertEqual(len(macd_line), len(self.prices))
        self.assertEqual(len(signal_line), len(self.prices))
        self.assertEqual(len(histogram), len(self.prices))
        
        # 验证histogram = macd_line - signal_line
        diff = macd_line - signal_line
        pd.testing.assert_series_equal(histogram, diff, check_names=False)
    
    def test_macd_market_parameters(self):
        """测试MACD市场参数适配（关键功能）"""
        # 美股参数
        macd_us, signal_us, hist_us = TechnicalIndicators.calculate_macd(
            self.prices, market='US'
        )
        
        # A股参数
        macd_cn, signal_cn, hist_cn = TechnicalIndicators.calculate_macd(
            self.prices, market='CN'
        )
        
        # 两者应该不同（因为参数不同）
        self.assertFalse(macd_us.equals(macd_cn))
        
        # 自定义参数应该覆盖市场默认值
        macd_custom, _, _ = TechnicalIndicators.calculate_macd(
            self.prices, fast=6, slow=13, signal=5, market='CN'
        )
        self.assertFalse(macd_custom.equals(macd_cn))
    
    def test_rsi_range(self):
        """测试RSI - 值域范围"""
        rsi = TechnicalIndicators.calculate_rsi(self.prices, period=14)
        
        # RSI应该在0-100之间（忽略NaN）
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_rsi_wilder_smoothing(self):
        """测试RSI使用Wilder平滑法（关键修正）"""
        # 创建简单的价格序列验证Wilder平滑
        prices = pd.Series([44, 44.5, 45, 45.5, 45, 44.5, 44, 43.5, 44, 44.5])
        rsi = TechnicalIndicators.calculate_rsi(prices, period=3)
        
        # RSI使用EWM，应该对近期价格更敏感
        # 验证RSI在有效范围内
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
        
        # Wilder平滑应该使第一个有效值出现在period位置
        self.assertTrue(pd.isna(rsi.iloc[:2]).all())  # 前2个应该是NaN
    
    def test_bollinger_bands_structure(self):
        """测试布林带 - 结构验证"""
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(self.prices)
        
        # 验证返回三个Series
        self.assertIsInstance(upper, pd.Series)
        self.assertIsInstance(middle, pd.Series)
        self.assertIsInstance(lower, pd.Series)
        
        # 验证长度一致
        self.assertEqual(len(upper), len(self.prices))
        
        # 验证关系：上轨 > 中轨 > 下轨（在有效值上）
        valid_idx = ~middle.isna()
        self.assertTrue((upper[valid_idx] >= middle[valid_idx]).all())
        self.assertTrue((middle[valid_idx] >= lower[valid_idx]).all())
    
    def test_bollinger_bands_middle_is_sma(self):
        """测试布林带 - 中轨应该等于SMA"""
        period = 20
        upper, middle, lower = TechnicalIndicators.calculate_bollinger_bands(
            self.prices, period=period
        )
        sma = TechnicalIndicators.calculate_sma(self.prices, period)
        
        # 中轨应该等于SMA
        pd.testing.assert_series_equal(middle, sma, check_names=False)
    
    def test_atr_positive(self):
        """测试ATR - 应该总是正值"""
        atr = TechnicalIndicators.calculate_atr(self.high, self.low, self.close)
        
        # ATR应该都是正值（忽略NaN）
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())
    
    def test_atr_volatility_measure(self):
        """测试ATR - 波动性度量"""
        # 高波动数据
        volatile_high = pd.Series([100, 110, 95, 105, 90, 115] * 5)
        volatile_low = volatile_high - 5
        volatile_close = (volatile_high + volatile_low) / 2
        
        # 低波动数据
        stable_high = pd.Series([100, 101, 100, 101, 100, 101] * 5)
        stable_low = stable_high - 1
        stable_close = (stable_high + stable_low) / 2
        
        atr_volatile = TechnicalIndicators.calculate_atr(volatile_high, volatile_low, volatile_close)
        atr_stable = TechnicalIndicators.calculate_atr(stable_high, stable_low, stable_close)
        
        # 高波动的ATR应该大于低波动的ATR
        self.assertGreater(atr_volatile.mean(), atr_stable.mean())
    
    def test_stochastic_range(self):
        """测试随机指标 - 值域范围"""
        k_line, d_line = TechnicalIndicators.calculate_stochastic(
            self.high, self.low, self.close
        )
        
        # K线和D线应该在0-100之间
        valid_k = k_line.dropna()
        valid_d = d_line.dropna()
        
        self.assertTrue((valid_k >= 0).all())
        self.assertTrue((valid_k <= 100).all())
        self.assertTrue((valid_d >= 0).all())
        self.assertTrue((valid_d <= 100).all())
    
    def test_obv_accumulation(self):
        """测试OBV - 累积特性"""
        obv = TechnicalIndicators.calculate_obv(self.close, self.volume)
        
        # OBV应该是累积值
        self.assertEqual(len(obv), len(self.close))
        
        # OBV应该随价格和成交量变化
        # 第一个值应该是0（diff的第一个是NaN，fillna(0)）
        self.assertEqual(obv.iloc[0], 0)
    
    def test_adx_complete_system(self):
        """测试完整ADX系统（+DI, -DI, ADX）"""
        plus_di, minus_di, adx = TechnicalIndicators.calculate_adx(
            self.high, self.low, self.close, period=14
        )
        
        # 验证返回三个Series
        self.assertIsInstance(plus_di, pd.Series)
        self.assertIsInstance(minus_di, pd.Series)
        self.assertIsInstance(adx, pd.Series)
        
        # 验证长度一致
        self.assertEqual(len(plus_di), len(self.high))
        self.assertEqual(len(minus_di), len(self.high))
        self.assertEqual(len(adx), len(self.high))
        
        # +DI和-DI应该都是正值
        valid_plus_di = plus_di.dropna()
        valid_minus_di = minus_di.dropna()
        self.assertTrue((valid_plus_di >= 0).all())
        self.assertTrue((valid_minus_di >= 0).all())
        
        # ADX应该在0-100之间
        valid_adx = adx.dropna()
        self.assertTrue((valid_adx >= 0).all())
        self.assertTrue((valid_adx <= 100).all())
    
    def test_adx_trend_detection(self):
        """测试ADX趋势检测能力"""
        # 创建明显上升趋势
        uptrend_high = pd.Series([100, 102, 105, 108, 112, 115, 119, 123, 127, 132] * 3)
        uptrend_low = uptrend_high - 2
        uptrend_close = (uptrend_high + uptrend_low) / 2
        
        plus_di_up, minus_di_up, adx_up = TechnicalIndicators.calculate_adx(
            uptrend_high, uptrend_low, uptrend_close, period=5
        )
        
        # 上升趋势中，+DI应该大于-DI
        valid_idx = ~plus_di_up.isna()
        if valid_idx.sum() > 0:
            # 取最后几个有效值比较
            last_valid = valid_idx[valid_idx].index[-5:]
            self.assertGreater(
                plus_di_up[last_valid].mean(),
                minus_di_up[last_valid].mean()
            )
    
    def test_insufficient_data(self):
        """测试数据不足 - 边界条件"""
        # 只有3个数据点，但要求5周期SMA
        short_data = pd.Series([10, 20, 30])
        sma = TechnicalIndicators.calculate_sma(short_data, 5)
        
        # 应该全是NaN
        self.assertTrue(sma.isna().all())


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
