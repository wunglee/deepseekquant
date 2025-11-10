import unittest
from core.backtest.backtest_engine import BacktestEngine, BacktestConfig, BacktestMetrics

class TestBacktestEngine(unittest.TestCase):
    def test_costs_applied(self):
        cfg = BacktestConfig(start_date='2020-01-01', end_date='2020-01-31', initial_capital=1000.0, commission=0.001, slippage=0.0005)
        engine = BacktestEngine(cfg)
        def strat(dp):
            return {'side': 'buy', 'quantity': 1}
        mkt = [{'timestamp': f'd{i}', 'price': 10.0 + i} for i in range(5)]
        metrics = engine.run(strat, mkt)
        notional_sum = sum(dp['price'] for dp in mkt)
        expected_capital = cfg.initial_capital - notional_sum * (1 + cfg.commission + cfg.slippage)
        self.assertAlmostEqual(engine.current_capital, expected_capital, places=6)

if __name__ == '__main__':
    unittest.main(verbosity=2)
