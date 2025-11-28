import unittest
import numpy as np
import pandas as pd

from core_bak_refactored.infrastructure.data_preprocessor import RiskDataPreprocessor


class DataPreprocessorTest(unittest.TestCase):
    def test_extract_returns_from_dict_with_returns_key(self):
        data = {'returns': pd.Series([0.01, 0.02, -0.01, 0.03])}
        returns = RiskDataPreprocessor.extract_returns_from_dict(data)
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), 4)
        
    def test_extract_returns_from_dict_with_prices_key(self):
        data = {'prices': [100, 102, 101, 104]}
        returns = RiskDataPreprocessor.extract_returns_from_dict(data)
        self.assertIsInstance(returns, pd.Series)
        self.assertGreater(len(returns), 0)
        
    def test_validate_returns_data_valid(self):
        returns = pd.Series(np.random.randn(50))
        is_valid = RiskDataPreprocessor.validate_returns_data(returns, min_length=20)
        self.assertTrue(is_valid)
        
    def test_validate_returns_data_insufficient(self):
        returns = pd.Series([0.01, 0.02])
        is_valid = RiskDataPreprocessor.validate_returns_data(returns, min_length=20)
        self.assertFalse(is_valid)


if __name__ == '__main__':
    unittest.main()
