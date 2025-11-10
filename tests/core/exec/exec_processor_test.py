import unittest
from core.exec.exec_processor import ExecProcessor, Order
from common import TradeDirection, OrderType, OrderStatus

class TestExecProcessor(unittest.TestCase):
    def test_initialization(self):
        p = ExecProcessor(processor_name='Exec')
        self.assertTrue(p.initialize())

    def test_order_execution(self):
        p = ExecProcessor(processor_name='Exec')
        p.initialize()
        order_data = {
            'symbol': 'AAPL',
            'quantity': 10,
            'side': TradeDirection.LONG,
            'order_type': OrderType.MARKET,
            'price': 150.0
        }
        result = p.process(order=order_data)
        self.assertEqual(result['status'], 'success')
        self.assertIn('order', result)
        order = result['order']
        self.assertEqual(order['status'], OrderStatus.FILLED.value)
        p.cleanup()

    def test_order_execution_with_costs(self):
        p = ExecProcessor(processor_name='Exec')
        p.initialize()
        order_data = {
            'symbol': 'MSFT',
            'quantity': 5,
            'side': TradeDirection.LONG,
            'order_type': OrderType.MARKET,
            'price': 200.0
        }
        result = p.process(order=order_data, commission=0.001, slippage=0.0005)
        self.assertEqual(result['status'], 'success')
        self.assertIn('execution_report', result)
        er = result['execution_report']
        self.assertIn('costs', er)
        self.assertGreaterEqual(er['costs'], 0.0)
        p.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
