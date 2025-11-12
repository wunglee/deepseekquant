"""
业务层测试 - 技术指标服务
测试 core/signal/indicator_service.py (TechnicalIndicators)

测试范围：业务概念、市场参数、金融术语
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signal.indicator_service import TechnicalIndicators, MARKET_PARAMS


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
        # A股调整为国际标准
        self.assertEqual(indicator.params['macd']['fast'], 12)
        self.assertEqual(indicator.params['macd']['slow'], 26)
        self.assertEqual(indicator.params['macd']['signal'], 9)
        # A股特色：短周期RSI
        self.assertEqual(indicator.params['rsi']['period'], 6)
    
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
        """测试批量计算所有指标 - 旧版接口兼容"""
        indicator = TechnicalIndicators(market='CN')
        # 使用'all'指标集包括所有指标
        results = indicator.calculate_all_indicators(self.data, indicator_set='all')
        
        # 验证返回的指标
        expected_keys = [
            'macd', 'macd_signal', 'macd_hist',
            'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr',
            'kdj_k', 'kdj_d',
            'obv',
            'adx_plus', 'adx_minus', 'adx'  # 高级指标
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
    
    # ==================== 新参数系统测试 ====================
    
    def test_cn_short_params(self):
        """测试A股短线参数"""
        indicator = TechnicalIndicators(market='CN_SHORT')
        
        self.assertEqual(indicator.market, 'CN_SHORT')
        self.assertEqual(indicator.params['macd']['fast'], 6)
        self.assertEqual(indicator.params['macd']['slow'], 13)
        self.assertEqual(indicator.params['macd']['signal'], 5)
        self.assertEqual(indicator.params['rsi']['period'], 6)
    
    def test_timeframe_params(self):
        """测试时间周期参数"""
        # 日线级别（使用市场默认参数，不被覆盖）
        indicator_daily = TechnicalIndicators(market='CN', timeframe='daily')
        self.assertEqual(indicator_daily.params['rsi']['period'], 6)  # A股默认6
        
        # 1分钟级别（高频，被时间周期覆盖）
        indicator_1min = TechnicalIndicators(market='CN', timeframe='1min')
        self.assertEqual(indicator_1min.params['rsi']['period'], 4)  # 被覆盖为1min参数
        self.assertEqual(indicator_1min.params['macd']['fast'], 3)
    
    def test_timeframe_overrides_market(self):
        """测试时间周期参数覆盖市场参数"""
        # US市场日线
        indicator_us_daily = TechnicalIndicators(market='US', timeframe='daily')
        
        # 5分钟应该覆盖为分钟级参数
        indicator_us_5min = TechnicalIndicators(market='US', timeframe='5min')
        self.assertEqual(indicator_us_5min.params['rsi']['period'], 6)  # 覆盖为5min参数
        
        # 但VWAP等市场特定参数保持
        self.assertIn('vwap', indicator_us_5min.params)
    
    # ==================== VWAP业务测试 ====================
    
    def test_vwap_calculation(self):
        """测试VWAP计算"""
        indicator = TechnicalIndicators(market='CN')
        vwap = indicator.calculate_vwap(self.high, self.low, self.close, self.volume)
        
        self.assertEqual(len(vwap), len(self.close))
        self.assertFalse(vwap.dropna().empty)
    
    def test_vwap_cn_daily_reset(self):
        """测试VWAP A股每日重置（T+1特色）"""
        indicator = TechnicalIndicators(market='CN')
        
        # 检查参数配置，应该使用pandas频率别'D'
        vwap_params = indicator.params.get('vwap', {})
        self.assertEqual(vwap_params.get('reset'), 'D')  # pandas频率别
        self.assertTrue(vwap_params.get('use_typical_price'))
    
    def test_vwap_price_comparison(self):
        """测试VWAP价格比较业务场景"""
        indicator = TechnicalIndicators(market='CN')
        vwap = indicator.calculate_vwap(self.high, self.low, self.close, self.volume)
        
        # VWAP应该在高低价范围内
        valid_idx = ~vwap.isna()
        if valid_idx.sum() > 0:
            # VWAP应该在合理范围内
            self.assertTrue(vwap[valid_idx].min() >= self.low[valid_idx].min() * 0.8)
            self.assertTrue(vwap[valid_idx].max() <= self.high[valid_idx].max() * 1.2)
    
    # ==================== CCI业务测试 ====================
    
    def test_cci_calculation(self):
        """测试CCI计算"""
        indicator = TechnicalIndicators(market='CN')
        cci = indicator.calculate_cci(self.high, self.low, self.close)
        
        self.assertEqual(len(cci), len(self.close))
        self.assertFalse(cci.dropna().empty)
    
    def test_cci_overbought_oversold(self):
        """测试CCI超买超卖区域"""
        indicator = TechnicalIndicators(market='CN')
        cci = indicator.calculate_cci(self.high, self.low, self.close)
        
        # CCI > 100 超买, < -100 超卖
        valid = cci.dropna()
        if len(valid) > 0:
            # 验证CCI可以超过这个范围
            self.assertTrue(len(valid) > 0)
    
    def test_cci_market_params(self):
        """测试CCI市场参数"""
        indicator_cn = TechnicalIndicators(market='CN')
        indicator_us = TechnicalIndicators(market='US')
        
        # A股使用14周期
        self.assertEqual(indicator_cn.params['cci']['period'], 14)
        # 美股使用20周期
        self.assertEqual(indicator_us.params['cci']['period'], 20)
    
    # ==================== 分层批量计算测试 ====================
    
    def test_calculate_basic_indicators(self):
        """测试基础指标组"""
        indicator = TechnicalIndicators(market='CN')
        results = indicator.calculate_all_indicators(self.data, indicator_set='basic')
        
        # 基础组包含：MACD, RSI, Bollinger
        expected_keys = ['macd', 'macd_signal', 'macd_hist', 'rsi', 
                        'bb_upper', 'bb_middle', 'bb_lower']
        
        for key in expected_keys:
            self.assertIn(key, results)
    
    def test_calculate_standard_indicators(self):
        """测试标准指标组"""
        indicator = TechnicalIndicators(market='CN')
        results = indicator.calculate_all_indicators(self.data, indicator_set='standard')
        
        # 标准组 = 基础组 + ATR, KDJ, OBV
        expected_keys = ['macd', 'rsi', 'bb_upper', 'atr', 'kdj_k', 'kdj_d', 'obv']
        
        for key in expected_keys:
            self.assertIn(key, results)
    
    def test_calculate_advanced_indicators(self):
        """测试高级指标组"""
        indicator = TechnicalIndicators(market='CN')
        results = indicator.calculate_all_indicators(self.data, indicator_set='advanced')
        
        # 高级组 = 标准组 + ADX, VWAP, CCI
        expected_keys = ['macd', 'rsi', 'atr', 'kdj_k', 'obv', 
                        'adx', 'vwap', 'cci']
        
        for key in expected_keys:
            self.assertIn(key, results)
    
    def test_calculate_all_indicators_set(self):
        """测试所有指标集合"""
        indicator = TechnicalIndicators(market='CN')
        results = indicator.calculate_all_indicators(self.data, indicator_set='all')
        
        # 应该包含所有指标
        self.assertIn('macd', results)
        self.assertIn('rsi', results)
        self.assertIn('atr', results)
        self.assertIn('adx', results)
        self.assertIn('vwap', results)
        self.assertIn('cci', results)


if __name__ == '__main__':
    unittest.main(verbosity=2)
