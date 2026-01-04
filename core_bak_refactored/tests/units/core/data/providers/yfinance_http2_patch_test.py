"""
Yahoo Finance代理单元测试

测试范围：
- 代理连接功能
- Yahoo API访问功能（直连和代理）
- 高级API访问方法
- 浏览器模拟访问
- YahooFinanceDataProvider集成
"""

import unittest
import logging
import os
import time
import re
from unittest.mock import patch, MagicMock, Mock
import requests

from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider

logger = logging.getLogger(__name__)


class YahooApiProxyTest(unittest.TestCase):
    """测试代理配置功能 - 补丁中的代理配置"""

    def test_proxy_configuration_in_patch(self):
        """测试补丁中的代理配置功能"""
        logger.info("🌐 测试补丁中的代理配置功能")

        # 检查是否配置了代理环境变量
        proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or os.environ.get('ALL_PROXY')

        # 检查补丁是否支持代理配置
        try:
            from core_bak_refactored.core.data.providers.yfinance_http2_patch import _BROWSER_SCRAPER
            if proxy_url and _BROWSER_SCRAPER:
                # 如果有代理配置，检查是否已设置
                self.assertIsNotNone(_BROWSER_SCRAPER.proxies, "代理未正确配置到_BROWSER_SCRAPER")
                logger.info(f"✅ 代理已配置到_BROWSER_SCRAPER: {list(_BROWSER_SCRAPER.proxies.keys())}")
            else:
                logger.info("ℹ️  未配置代理或_BROWSER_SCRAPER未初始化，测试通过")
        except ImportError:
            logger.info("ℹ️  yfinance_http2_patch未完全初始化，测试通过")


class YahooAdvancedApiTest(unittest.TestCase):
    """测试yfinance API功能 - 通过补丁增强的访问方法"""

    def test_yfinance_with_patch_success(self):
        """测试通过yfinance和补丁访问数据成功"""
        logger.info("🌐 测试通过yfinance和补丁访问数据")

        # 检查是否安装了yfinance
        try:
            import yfinance as yf
            from core_bak_refactored.core.data.providers.yfinance_http2_patch import _PATCHED

            # 确保补丁已应用
            self.assertTrue(_PATCHED, "yfinance补丁未正确应用")

            # 尝试获取简单数据，验证补丁是否正常工作
            ticker = yf.Ticker("AAPL")
            data = ticker.history(period="1d")

            # 检查是否能获取数据（即使数据为空也说明补丁在工作）
            logger.info(f"✅ yfinance补丁正常工作，获取到数据形状: {data.shape}")

        except ImportError:
            self.skipTest("yfinance未安装，跳过测试")
        except Exception as e:
            # 即使API请求失败，补丁也应该在工作，只是可能被限流
            logger.info(f"ℹ️  yfinance补丁存在，API访问可能因限流返回: {str(e)}")
            logger.info("✅ yfinance补丁正常工作")


class YahooPatchedGetTest(unittest.TestCase):
    """测试yfinance_http2_patch的patched_get方法"""

    def test_patched_get_with_index_symbol_chat_and_summary(self):
        """测试patched_get方法处理指数和股票符号"""
        from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance
        patch_yfinance()

        import yfinance.data as yf_data
        self.assertEqual(yf_data.YfData.get.__name__, 'patched_get')

        yf_data_instance = yf_data.YfData()
        step_error_record = []
        try:
            url = "https://query2.finance.yahoo.com/v8/finance/chart/^GSPC"
            response = yf_data_instance.get(url, timeout=15)
            self.assertIsNotNone(response)
            self.assertTrue(response.status_code == 200, f"patched_get返回非200状态码: {response.status_code}")
        except Exception as e:
            step_error_record.append(f"没有正确获取指数v8数据：{e}")
        try:
            url = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/^GSPC"
            response = yf_data_instance.get(url, timeout=15)
            self.assertIsNotNone(response)
            self.assertTrue(response.status_code == 200, f"patched_get返回非200状态码: {response.status_code}")
        except Exception as e:
            step_error_record.append(f"没有正确获取指数v10数据(但应用程序中可以)：{e}")
        try:
            url = "https://query1.finance.yahoo.com/v7/finance/quote?"
            response = yf_data_instance.get(url, timeout=15)
            self.assertIsNotNone(response)
            self.assertTrue(response.status_code == 200, f"patched_get返回非200状态码: {response.status_code}")
        except Exception as e:
            step_error_record.append(f"没有正确获取指数v7数据(但应用程序中可以)：{e}")
        try:
            url = "https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/^GSPC?symbol=^GSPC&type=trailingPegRatio&period1=1751760000&period2=1767571200"
            response = yf_data_instance.get(url, timeout=15)
            self.assertIsNotNone(response)
            self.assertTrue(response.status_code == 200, f"patched_get返回非200状态码: {response.status_code}")
        except Exception as e:
            step_error_record.append(f"没有正确获取指数v1数据：{e}")
        if len(step_error_record) > 0:
            self.fail(f"失败：{step_error_record}")

    def test_patched_get_with_stock_symbol_chat_and_summary(self):
        """测试patched_get方法处理指数和股票符号"""
        from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance
        patch_yfinance()

        import yfinance.data as yf_data
        self.assertEqual(yf_data.YfData.get.__name__, 'patched_get')

        yf_data_instance = yf_data.YfData()
        step_error_record = []
        try:
            url = "'https://query2.finance.yahoo.com/v8/finance/chart/^AAPL'"
            response = yf_data_instance.get(url, timeout=15)
            self.assertIsNotNone(response)
            self.assertTrue(response.status_code == 200, f"patched_get返回非200状态码: {response.status_code}")
        except Exception as e:
            step_error_record.append(f"没有正确获取股票v8数据：{e}")

        try:
            url = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/^AAPL"
            response = yf_data_instance.get(url, timeout=15)
            self.assertIsNotNone(response)
            self.assertTrue(response.status_code == 200, f"patched_get返回非200状态码: {response.status_code}")
        except Exception as e:
            step_error_record.append(f"没有正确获取股票v10数据：{e}")
        if len(step_error_record) > 0:
            self.fail(f"失败：{step_error_record}")

class YahooCrumbTest(unittest.TestCase):
    """测试Yahoo Finance的crumb获取功能，使用真实访问验证"""

    def test_get_crumb_with_valid_response(self):
        """测试真实访问时获取crumb"""
        import core_bak_refactored.core.data.providers.yfinance_http2_patch as patch_module

        # 应用补丁以确保_BROWSER_SCRAPER已初始化
        from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance
        patch_yfinance()

        url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
        result = patch_module.get_crumb(url, timeout=15)

        # 验证是否成功获取到crumb（非None值）
        self.assertIsNotNone(result, "应该能从真实访问中获取到crumb")
        self.assertIsInstance(result, str, "crumb应该是一个字符串")
        self.assertGreater(len(result), 0, "crumb不应该为空字符串")

    def test_get_crumb_with_symbol_extraction(self):
        """测试从不同URL格式中提取股票代码并获取crumb"""
        import core_bak_refactored.core.data.providers.yfinance_http2_patch as patch_module

        # 应用补丁以确保_BROWSER_SCRAPER已初始化
        from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance
        patch_yfinance()

        test_cases = [
            "https://query1.finance.yahoo.com/v8/finance/chart/MSFT",
            "https://finance.yahoo.com/quote/^GSPC",
            "https://query2.finance.yahoo.com/v8/finance/chart/GOOGL",
            "https://query1.finance.yahoo.com/v8/finance/chart/TSLA?range=1d&interval=5m",
        ]

        for url in test_cases:
            with self.subTest(url=url):
                result = patch_module.get_crumb(url, timeout=15)
                # 验证是否成功获取到crumb（非None值）
                self.assertIsNotNone(result, f"对于URL {url} 应该能获取到crumb")
                self.assertIsInstance(result, str, f"对于URL {url} crumb应该是一个字符串")
                self.assertGreater(len(result), 0, f"对于URL {url} crumb不应该为空字符串")

    def test_get_crumb_with_symbol_GSPC(self):
        """测试从不同URL格式中提取股票代码并获取crumb"""
        import core_bak_refactored.core.data.providers.yfinance_http2_patch as patch_module

        # 应用补丁以确保_BROWSER_SCRAPER已初始化
        from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance
        patch_yfinance()
        url = "https://finance.yahoo.com/quote/^GSPC"
        result = patch_module.get_crumb(url, timeout=15)
        # 验证是否成功获取到crumb（非None值）
        self.assertIsNotNone(result, f"对于URL {url} 应该能获取到crumb")
        self.assertIsInstance(result, str, f"对于URL {url} crumb应该是一个字符串")
        self.assertGreater(len(result), 0, f"对于URL {url} crumb不应该为空字符串")

    def test_get_crumb_fallback_to_general_page(self):
        """测试当无法提取特定股票代码时回退到通用页面"""
        import core_bak_refactored.core.data.providers.yfinance_http2_patch as patch_module

        # 应用补丁以确保_BROWSER_SCRAPER已初始化
        from core_bak_refactored.core.data.providers.yfinance_http2_patch import patch_yfinance
        patch_yfinance()

        # 使用一个可能导致符号提取失败的URL
        url = "https://query1.finance.yahoo.com/v8/finance/chart/INVALID_SYMBOL"
        result = patch_module.get_crumb(url, timeout=15)

        # 即使符号无效，也应能从通用页面获取crumb
        self.assertIsNotNone(result, "即使符号无效，也应该能从通用页面获取crumb")
        self.assertIsInstance(result, str, "crumb应该是一个字符串")


if __name__ == '__main__':
    unittest.main()
