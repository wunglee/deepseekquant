import unittest
import pandas as pd

from core_bak_refactored.core.data.data_utils import DataUtils
from core_bak_refactored.core.backtest.event_analysis import EventConfig, EventAnalyzer
from core_bak_refactored.core.share.data_analysis_utils import DataAnalysisUtils
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider


class DataUtilsTest(unittest.TestCase):
    def test_calculate_return_and_actual_return(self):
        df = pd.DataFrame({'close': [100, 110, 121]})
        r = DataUtils.calculate_return(df)
        self.assertAlmostEqual(r, 0.21, places=6)
        ar = EventAnalyzer.calculate_actual_return(df)
        self.assertAlmostEqual(ar, 0.21, places=6)

    def test_validate_dataframe(self):
        df = pd.DataFrame({'close': [100, 110]})
        valid, msg = DataAnalysisUtils.validate_dataframe(df, ['close'], min_rows=2)
        self.assertTrue(valid)
        self.assertEqual(msg, '')
        invalid, msg2 = DataAnalysisUtils.validate_dataframe(df[['close']].iloc[:1], ['close'], min_rows=2)
        self.assertFalse(invalid)
        self.assertIn('数据行数不足', msg2)

    def test_compute_daily_and_cumulative_returns(self):
        prices = pd.Series([100.0, 110.0, 121.0])
        returns = DataUtils.compute_daily_returns(prices)
        self.assertTrue(pd.isna(returns.iloc[0]))
        self.assertAlmostEqual(returns.iloc[1], 0.1, places=6)
        cum = DataUtils.compute_cumulative_return(returns.fillna(0.0))
        self.assertAlmostEqual(cum.iloc[0], 1.0, places=6)
        self.assertAlmostEqual(cum.iloc[2], 1.21, places=6)

    def test_safe_get_event_data_with_mock_provider(self):
        provider = MockHistoricalDataProvider()
        event = EventConfig(
            event_id='unit_test_event',
            symbol='000300.SH',
            event_date='2015-06-15',
            event_type='market_crash',
            expected_decline=-0.3,
            market_id='CN'
        )
        df, success = EventAnalyzer.safe_get_event_data(provider, event, window_days=10, baseline_days=30)
        self.assertTrue(success)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('close', df.columns)


if __name__ == '__main__':
    unittest.main()
