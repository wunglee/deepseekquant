import unittest
from core.backtest.backtest_engine import BacktestEngine, BacktestConfig, BacktestMetrics

class TestBacktestEngine(unittest.TestCase):
    def test_basic_backtest(self):
        config = BacktestConfig(start_date='2024-01-01', end_date='2024-12-31', initial_capital=100000)
        engine = BacktestEngine(config)
        
        # 模拟市场数据
        market_data = [
            {'timestamp': '2024-01-01', 'price': 100},
            {'timestamp': '2024-01-02', 'price': 105},
            {'timestamp': '2024-01-03', 'price': 110}
        ]
        
        # 简单策略：总是买入1股
        def strategy(data):
            return {'side': 'buy', 'quantity': 1}
        
        metrics = engine.run(strategy, market_data)
        self.assertIsInstance(metrics, BacktestMetrics)
        self.assertEqual(metrics.total_trades, 3)

if __name__ == '__main__':
    unittest.main(verbosity=2)
