import unittest
from core.risk.risk_processor import RiskProcessor

class TestRiskVolConcentration(unittest.TestCase):
    def test_volatility_threshold(self):
        rp = RiskProcessor(processor_name='Risk')
        rp.initialize()
        signal = {'price': 100.0, 'quantity': 10}
        prices = [i for i in range(1, 100)]
        limits = {'volatility_threshold': 0.05}
        res = rp.process(signal=signal, limits=limits, prices=prices)
        self.assertEqual(res['status'], 'success')
        rp.cleanup()

    def test_concentration_threshold(self):
        rp = RiskProcessor(processor_name='Risk')
        rp.initialize()
        res = rp.process(signal={'price': 100.0, 'quantity': 10}, limits={'max_position_value': 100000}, target_weight=0.8)
        self.assertEqual(res['status'], 'success')
        self.assertIn('assessment', res)
        rp.cleanup()

    def test_correlation_and_drawdown(self):
        rp = RiskProcessor(processor_name='Risk')
        rp.initialize()
        # 两条高度相关的序列
        a = [100 + i*0.5 for i in range(60)]
        b = [100 + i*0.5 + 0.1 for i in range(60)]
        limits = {'correlation_threshold': 0.7}
        res = rp.process(signal={'price': 0.0, 'quantity': 0.0}, limits=limits, histories={'A': a, 'B': b})
        self.assertEqual(res['status'], 'success')
        rp.cleanup()

        # 一个下跌序列触发回撤
        rp = RiskProcessor(processor_name='Risk')
        rp.initialize()
        c = [100 - i for i in range(60)]
        limits = {'max_drawdown_threshold': 0.2}
        res = rp.process(signal={'price': 0.0, 'quantity': 0.0}, limits=limits, histories={'C': c})
        self.assertEqual(res['status'], 'success')
        rp.cleanup()
