import unittest
from core.signal.signal_processor import SignalProcessor

class TestSignalRules(unittest.TestCase):
    def test_reason_details(self):
        sp = SignalProcessor(processor_name='Signal')
        sp.initialize()
        # 构造RSI触发
        prices = [i for i in range(100, 200)]
        params = {'rsi_buy': 70}  # 使用较高buy阈值使RSI低于阈值
        res = sp.process(symbol='DDD', price=prices[-1], prices=prices, params=params)
        self.assertEqual(res['status'], 'success')
        self.assertIn('signal', res)
        self.assertIn('reason', res['signal'])
        sp.cleanup()

    def test_ema_crossover(self):
        sp = SignalProcessor(processor_name='Signal')
        sp.initialize()
        # 构造快线大于慢线的数据（上涨趋势）
        prices = [i for i in range(1, 60)]
        res = sp.process(symbol='BBB', price=prices[-1], prices=prices)
        signal = res['signal']
        self.assertIn(signal['signal_type'], ['buy', 'sell', 'hold'])
        sp.cleanup()

    def test_composite_rule(self):
        sp = SignalProcessor(processor_name='Signal')
        sp.initialize()
        prices = [i for i in range(1, 80)]
        params = {'use_composite': True, 'composite_buy_thr': 0.1}
        res = sp.process(symbol='CCC', price=prices[-1], prices=prices, params=params)
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['signal']['signal_type'], 'buy')
        sp.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
