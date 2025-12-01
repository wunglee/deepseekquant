import unittest

from core_bak_refactored.infrastructure.currency_converter import CurrencyConverter


class CurrencyConverterTest(unittest.TestCase):
    def test_basic_conversion(self):
        converter = CurrencyConverter()
        portfolio = {
            "allocations": {
                "AAPL": {"currency": "USD", "value": 1000.0},
                "600036.SH": {"currency": "CNY", "value": 7000.0},
            }
        }
        rates = {"CNY": {"USD": 0.14}, "USD": {"USD": 1.0}}
        result = converter.convert_portfolio_currency(portfolio, "USD", rates)
        self.assertEqual(result["target_currency"], "USD")
        self.assertGreater(result["total_converted_value"], 0.0)
        self.assertIn("AAPL", result["details"])

    def test_currency_exposure(self):
        converter = CurrencyConverter()
        portfolio = {
            "allocations": {
                "A": {"currency": "USD", "value": 100.0},
                "B": {"currency": "CNY", "value": 200.0},
                "C": {"currency": "CNY", "value": 300.0},
            }
        }
        exposure = converter.calculate_currency_exposure(portfolio)
        self.assertEqual(exposure["USD"], 100.0)
        self.assertEqual(exposure["CNY"], 500.0)


if __name__ == '__main__':
    unittest.main()
