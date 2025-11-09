import unittest
from core.signal.signal_processor import SignalProcessor

class TestSignalProcessor(unittest.TestCase):
    def test_basic_signal(self):
        p = SignalProcessor()
        p.initialize()
        r = p.process(data='alpha')
        self.assertEqual(r['status'], 'success')
        self.assertIn('signal:alpha', r['data'])
        p.cleanup()

if __name__ == '__main__':
    unittest.main(verbosity=2)
