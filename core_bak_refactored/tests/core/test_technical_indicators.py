"""
业务层测试 - 技术指标
测试 core/signal/technical_indicators.py (TechnicalIndicators)

测试范围：业务概念、市场参数、金融术语
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signal.technical_indicators import TechnicalIndicators, MARKET_PARAMS


class TestTechnicalIndicators(unittest.TestCase):
    """技术指标业务层测试"""
    
    def setUp(self):
        """准备测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        prices = 100 + np.cumsum(np.random.randn(100) * 2)
        self.prices = pd.Series(prices, index=dates)
        
        self.high = self.prices + np.random.rand(100) * 2
        self.low = self.prices - np.random.rand(100) * 2
        self.close = self.prices
        self.volume = pd.Series(np.random.randint(1000000, 10000000, 100), index=dates)
        
        # 创建OHLCV DataFrame
        self.data = pd.DataFrame({
            'open': self.prices,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        })
    
    # ==================== 市场参数测试 ====================
    
    def test_market_params_cn(self):
        """测试A股市场参数"""
        indicator = TechnicalIndicators(market='CN')
        
        self.assertEqual(indicator.market, 'CN')
        self.assertEqual(indicator.params['macd']['fast'], 8)
        self.assertEqual(indicator.params['macd']['slow'], 21)
        self.assertEqual(indicator.params['macd']['signal'], 8)
    
    def test_market_params_us(self):
        """测试美股市场参数"""
        indicator = TechnicalIndicators(market='US')
        
        self.assertEqual(indicator.market, 'US')
        self.assertEqual(indicator.params['macd']['fast'], 12)
        self.assertEqual(indicator.params['macd']['slow'], 26)
        self.assertEqual(indicator.params['macd']['signal'], 9)
    
    # ==================== MACD业务测试 ====================
    
    def test_macd_cn_market(self):
        """测试MACD - A股市场默认参数"""
        indicator = TechnicalIndicators(market='CN')
        macd, signal, hist = indicator.calculate_macd(self.prices)
        
        # 应该使用A股参数(8, 21, 8)
        self.assertIsInstance(macd, pd.Series)
        self.assertEqual(len(macd), len(self.prices))
    
    def test_macd_us_market(self):
        """测试MACD - 美股市场默认参数"""
        indicator = TechnicalIndicators(market='US')
        macd, signal, hist = indicator.calculate_macd(self.prices)
        
        # 应该使用美股参数(12, 26, 9)
        self.assertIsInstance(macd, pd.Series)
    
    def test_macd_custom_params(self):
        """测试MACD - 自定义参数覆盖市场默认值"""
        indicator = TechnicalIndicators(market='CN')
        macd_default, _, _ = indicator.calculate_macd(self.prices)
        macd_custom, _, _ = indicator.calculate_macd(self.prices, fast=6, slow=13, signal=5)
        
        # 自定义参数应该产生不同结果
        self.assertFalse(macd_default.equals(macd_custom))
    
    # ==================== RSI业务测试 ====================
    
    def test_rsi_range(self):
        """测试RSI - 值域范围（0-100）"""
        indicator = TechnicalIndicators(market='CN')
        rsi = indicator.calculate_rsi(self.prices)
        
        valid = rsi.dropna()
        self.assertTrue((valid >= 0).all())
        self.assertTrue((valid <= 100).all())
    
    def test_rsi_overbought_oversold(self):
        """测试RSI - 超买超卖区域"""
        indicator = TechnicalIndicators(market='CN')
        
        # 持续上涨 -> RSI > 70 (超买)
        up_trend = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] * 3)
        rsi_up = indicator.calculate_rsi(up_trend, period=5)
        self.assertGreater(rsi_up.iloc[-1], 70)
        
        # 持续下跌 -> RSI < 30 (超卖)
        down_trend = pd.Series([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10] * 3)
        rsi_down = indicator.calculate_rsi(down_trend, period=5)
        self.assertLess(rsi_down.iloc[-1], 30)
    
    # ==================== 布林带业务测试 ====================
    
    def test_bollinger_bands_structure(self):
        """测试布林带 - 三轨结构"""
        indicator = TechnicalIndicators(market='CN')
        upper, middle, lower = indicator.calculate_bollinger_bands(self.prices)
        
        # 上轨 >= 中轨 >= 下轨
        valid_idx = ~middle.isna()
        self.assertTrue((upper[valid_idx] >= middle[valid_idx]).all())
        self.assertTrue((middle[valid_idx] >= lower[valid_idx]).all())
    
    def test_bollinger_bands_market_params(self):
        """测试布林带 - 市场参数"""
        indicator_cn = TechnicalIndicators(market='CN')
        indicator_us = TechnicalIndicators(market='US')
        
        # A股和美股参数相同（都是period=20, std=2.0）
        upper_cn, middle_cn, lower_cn = indicator_cn.calculate_bollinger_bands(self.prices)
        upper_us, middle_us, lower_us = indicator_us.calculate_bollinger_bands(self.prices)
        
        pd.testing.assert_series_equal(middle_cn, middle_us, check_names=False)
    
    # ==================== ATR业务测试 ====================
    
    def test_atr_positive(self):
        """测试ATR - 恒为正值"""
        indicator = TechnicalIndicators(market='CN')
        atr = indicator.calculate_atr(self.high, self.low, self.close)
        
        valid = atr.dropna()
        self.assertTrue((valid > 0).all())
    
    def test_atr_for_stop_loss(self):
        """测试ATR - 用于止损计算的业务场景"""
        indicator = TechnicalIndicators(market='CN')
        atr = indicator.calculate_atr(self.high, self.low, self.close)
        
        # 模拟止损位计算：当前价格 - 2*ATR
        current_price = self.close.iloc[-1]
        current_atr = atr.iloc[-1]
        
        if not pd.isna(current_atr):
            stop_loss = current_price - 2 * current_atr
            self.assertLess(stop_loss, current_price)
    
    # ==================== KDJ业务测试 ====================
    
    def test_kdj_range(self):
        """测试KDJ - 值域范围（0-100）"""
        indicator = TechnicalIndicators(market='CN')
        k, d = indicator.calculate_kdj(self.high, self.low, self.close)
        
        valid_k = k.dropna()
        valid_d = d.dropna()
        
        self.assertTrue((valid_k >= 0).all())
        self.assertTrue((valid_k <= 100).all())
        self.assertTrue((valid_d >= 0).all())
        self.assertTrue((valid_d <= 100).all())
    
    # ==================== OBV业务测试 ====================
    
    def test_obv_trend_confirmation(self):
        """测试OBV - 趋势确认业务场景"""
        indicator = TechnicalIndicators(market='CN')
        obv = indicator.calculate_obv(self.close, self.volume)
        
        # OBV应该累积
        self.assertEqual(len(obv), len(self.close))
        self.assertEqual(obv.iloc[0], 0)
    
    # ==================== ADX业务测试 ====================
    
    def test_adx_trend_strength(self):
        """测试ADX - 趋势强度判断"""
        indicator = TechnicalIndicators(market='CN')
        plus_di, minus_di, adx = indicator.calculate_adx(
            self.high, self.low, self.close
        )
        
        # ADX > 25 表示强趋势
        valid_adx = adx.dropna()
        if len(valid_adx) > 0:
            # ADX应该在0-100之间
            self.assertTrue((valid_adx >= 0).all())
            self.assertTrue((valid_adx <= 100).all())
    
    def test_adx_direction_detection(self):
        """测试ADX - 趋势方向判断"""
        indicator = TechnicalIndicators(market='CN')
        
        # 上升趋势数据
        uptrend_high = pd.Series([100, 102, 105, 108, 112, 115, 119, 123, 127, 132] * 3)
        uptrend_low = uptrend_high - 2
        uptrend_close = (uptrend_high + uptrend_low) / 2
        
        plus_di, minus_di, adx = indicator.calculate_adx(
            uptrend_high, uptrend_low, uptrend_close, period=5
        )
        
        # 上升趋势中，+DI应该大于-DI
        valid_idx = ~plus_di.isna()
        if valid_idx.sum() > 0:
            last_valid = valid_idx[valid_idx].index[-5:]
            self.assertGreater(
                plus_di[last_valid].mean(),
                minus_di[last_valid].mean()
            )
    
    # ==================== 批量计算测试 ====================
    
    def test_calculate_all_indicators(self):
        """测试批量计算所有指标"""
        indicator = TechnicalIndicators(market='CN')
        results = indicator.calculate_all_indicators(self.data)
        
        # 验证返回的指标
        expected_keys = [
            'macd', 'macd_signal', 'macd_hist',
            'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr',
            'kdj_k', 'kdj_d',
            'obv',
            'adx_plus', 'adx_minus', 'adx'
        ]
        
        for key in expected_keys:
            self.assertIn(key, results)
            self.assertIsInstance(results[key], pd.Series)
    
    # ==================== 业务场景集成测试 ====================
    
    def test_trading_signal_generation(self):
        """测试交易信号生成场景"""
        indicator = TechnicalIndicators(market='CN')
        
        # 计算多个指标
        macd, signal, hist = indicator.calculate_macd(self.prices)
        rsi = indicator.calculate_rsi(self.prices)
        plus_di, minus_di, adx = indicator.calculate_adx(
            self.high, self.low, self.close
        )
        
        # 模拟多指标共振信号
        # 条件：MACD金叉 + RSI不超买 + ADX强趋势
        if not (macd.isna().all() or rsi.isna().all() or adx.isna().all()):
            last_idx = len(self.prices) - 1
            
            # 验证指标都有值
            if not pd.isna(macd.iloc[last_idx]):
                self.assertIsNotNone(macd.iloc[last_idx])


if __name__ == '__main__':
    unittest.main(verbosity=2)
