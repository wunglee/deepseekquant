import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import unittest

from core_bak_refactored.core.risk.stress_testing import StressTester


class StressTestingTest(unittest.TestCase):
    def test_engine_instantiation(self):
        engine = StressTester(config={})
        self.assertIsNotNone(engine)


if __name__ == '__main__':
    unittest.main()
