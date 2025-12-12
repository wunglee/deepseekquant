import unittest

from core_bak_refactored.core.risk.backtest_framework import HistoricalDataProvider
from core_bak_refactored.core.data.providers.factory import get_global_factory
from core_bak_refactored.tests.fixtures.core.data.mock_historical_data_provider import MockHistoricalDataProvider
from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider
from core_bak_refactored.core.data.providers.akshare_provider import AKShareDataProvider


class BacktestFrameworkTest(unittest.TestCase):
    def test_create_mock_provider(self):
        provider: HistoricalDataProvider = MockHistoricalDataProvider()
        self.assertIsInstance(provider, MockHistoricalDataProvider)

    def test_create_yahoo_provider_with_factory(self):
        """测试使用工厂创建 Yahoo provider"""
        factory = get_global_factory()
        provider = factory.get('yahoo')
        self.assertTrue(isinstance(provider, YahooFinanceDataProvider))

    def test_factory_returns_configured_provider(self):
        """测试工厂返回配置的 provider（从配置文件读取）"""
        factory = get_global_factory()
        # 工厂会从 config/dev/data.yml 中加载配置的 providers
        # 验证 akshare 和 yahoo 都在配置中
        provider = factory.get('akshare')
        self.assertTrue(
            isinstance(provider, AKShareDataProvider),
            f"期望返回 AKShareDataProvider，但得到 {type(provider)}"
        )


if __name__ == '__main__':
    unittest.main()
