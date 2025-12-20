"""
测试 BaseProvider 基类
"""

import unittest
from unittest.mock import patch, MagicMock

from core_bak_refactored.core.data.providers.base_provider import BaseDataProvider
from core_bak_refactored.core.data.providers.protocols import IntradayData
from core_bak_refactored.core.share.market.market_enums import TradingPhase


class MockProvider(BaseDataProvider):
    """用于测试的模拟提供者"""
    
    def available(self) -> bool:
        return True
    
    def get_test_symbol(self) -> str:
        return "000300.SH"
    
    def _fetch_from_external_api(self, symbol: str, start_date: str, end_date: str, period: str = 'daily'):
        # 简单的模拟实现
        pass


class TestBaseProvider(unittest.TestCase):
    """测试 BaseProvider 基类"""

    def setUp(self):
        """测试前准备"""
        self.provider = MockProvider()


if __name__ == '__main__':
    unittest.main()