import unittest
import pandas as pd

from core_bak_refactored.core.exec.algorithms import ExecutionAlgorithms


class ExecutionAlgosTest(unittest.TestCase):
    def test_twap_schedule_basic(self):
        schedule = ExecutionAlgorithms.twap_schedule(
            total_quantity=1000,
            duration_minutes=60,
            interval_minutes=5
        )
        self.assertEqual(len(schedule), 12)
        self.assertAlmostEqual(schedule[0]['quantity'], 1000/12, places=2)
        
    def test_vwap_schedule_with_volume_profile(self):
        volume_profile = pd.Series([100, 200, 150, 50])
        schedule = ExecutionAlgorithms.vwap_schedule(
            total_quantity=1000,
            volume_profile=volume_profile
        )
        self.assertEqual(len(schedule), 4)
        total_qty = sum(s['quantity'] for s in schedule)
        self.assertAlmostEqual(total_qty, 1000, places=2)
        
    def test_iceberg_schedule(self):
        schedule = ExecutionAlgorithms.iceberg_schedule(
            total_quantity=1000,
            display_quantity=100
        )
        self.assertEqual(len(schedule), 10)
        self.assertTrue(all(s['is_hidden'] for s in schedule))


if __name__ == '__main__':
    unittest.main()
