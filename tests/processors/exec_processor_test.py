import unittest
from core.exec.exec_processor import ExecProcessor

class TestExecProcessor(unittest.TestCase):
    def test_basic_exec(self):
        p = ExecProcessor()
        p.initialize()
        r = p.process(order={'symbol': 'TEST', 'qty': 10})
        self.assertEqual(r['status'], 'success')
        self.assertEqual(r['order']['qty'], 10)
        p.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
