import unittest
from core.signal.signal_processor import SignalProcessor

class TestSignalRules(unittest.TestCase):
    def test_rsi_rule(self):
        sp = SignalProcessor(processor_name='Signal')
        sp.initialize()
        prices = [100 + i for i in range(40)]
        res = sp.process(symbol='AAA', price=prices[-1], prices=prices)
        self.assertEqual(res['status'], 'success')
        self.assertIn('signal', res)
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

if __name__ == '__main__':
    unittest.main(verbosity=2)
