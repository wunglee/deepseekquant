import unittest

from core_bak_refactored.core.backtest._fragments.stress_test_result import StressTestResult, from_backtest_result


class _MockBacktestResult:
    def __init__(self):
        self.portfolio_id = 'P1'
        self.event_id = 'E1'
        self.actual_loss = -0.12
        self.predicted_loss = -0.10
        self.prediction_error = 0.02
        self.benchmark_index = '000300.SH'
        self.metadata = {'event_name': 'Event1', 'period': ('2020-01-01', '2020-02-01')}


class StressTestResultTest(unittest.TestCase):
    def test_from_backtest_result(self):
        r = _MockBacktestResult()
        s = from_backtest_result(r)
        self.assertIsInstance(s, StressTestResult)
        d = s.to_dict()
        self.assertEqual(d['portfolio_id'], 'P1')
        self.assertEqual(d['scenario_id'], 'E1')
        self.assertAlmostEqual(d['stress_loss_percentage'], -0.12, places=6)
        self.assertIn('metadata', d)
        self.assertEqual(d['metadata']['benchmark_index'], '000300.SH')


if __name__ == '__main__':
    unittest.main()
