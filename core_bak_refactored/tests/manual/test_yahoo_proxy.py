#!/usr/bin/env python3
"""
Yahoo Finance API 测试脚本

此脚本用于测试Yahoo Finance API的访问情况，包括：
1. 代理连接测试
2. 直连Yahoo API测试
3. 代理访问Yahoo API测试
4. YahooFinanceDataProvider测试

用于诊断429错误问题
"""

import requests
import time
import random
import logging
from typing import Optional
import json
import re
from urllib.parse import urlencode
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ssl
from requests.packages.urllib3.util.ssl_ import create_urllib3_context

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SSLAdapter(HTTPAdapter):
    """自定义SSL适配器，用于处理SSL/TLS连接问题"""

    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.set_ciphers('DEFAULT@SECLEVEL=1')  # 降低SSL安全级别以兼容性优先
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)


def test_proxy_connection(proxy_url: str):
    """测试代理连接"""
    try:
        # 测试代理连接
        proxies = {
            'http': proxy_url,
            'https': proxy_url
        }

        response = requests.get('http://httpbin.org/ip', proxies=proxies, timeout=10)
        if response.status_code == 200:
            logger.info(f"✅ 代理连接测试成功: {proxy_url}")
            logger.info(f"   代理IP: {response.json()}")
            assert response.status_code == 200, f"代理连接测试失败: {proxy_url}, 状态码: {response.status_code}"
        else:
            logger.error(f"❌ 代理连接测试失败: {proxy_url}, 状态码: {response.status_code}")
            assert response.status_code == 200, f"代理连接测试失败: {proxy_url}, 状态码: {response.status_code}"

    except Exception as e:
        logger.error(f"❌ 代理连接测试异常: {e}")
        raise AssertionError(f"代理连接测试异常: {e}")


def test_yahoo_api_direct():
    """测试直连Yahoo API"""
    logger.info("🔍 开始测试直连Yahoo API")

    # 尝试使用浏览器类似的请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://finance.yahoo.com/',
        'Origin': 'https://finance.yahoo.com'
    }

    # 测试URL - 使用Yahoo Finance的API端点
    url = 'https://query2.finance.yahoo.com/v8/finance/chart/^GSPC'

    # 添加查询参数以模拟浏览器请求
    params = {
        'period1': int(time.time()) - 365 * 24 * 60 * 60,  # 一年前
        'period2': int(time.time()),  # 当前时间
        'interval': '1d',
        'includePrePost': 'true',
        'events': 'div,splits,capitalGains'
    }

    try:
        # 创建会话并配置SSL
        session = requests.Session()
        session.mount('https://', SSLAdapter())

        # 设置重试策略
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,  # 减少退避因子
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 设置请求头
        session.headers.update(headers)

        # 先访问一个页面获取可能的cookies和crumb
        logger.info("🌐 访问Yahoo Finance主页以获取cookies和crumb...")
        home_response = session.get('https://finance.yahoo.com/quote/^GSPC', timeout=10)

        # 尝试从页面中提取crumb
        crumb = None
        if home_response.status_code == 200:
            # 查找crumb的常见模式
            crumb_match = re.search(r'"crumb":"([^"]+)"', home_response.text)
            if crumb_match:
                crumb = crumb_match.group(1)
                logger.info(f"🔑 找到crumb: {crumb[:10]}...")

        # 如果找到crumb，更新参数
        if crumb:
            params['crumb'] = crumb

        logger.info(f"🌐 发送请求到: {url}")
        logger.info(f"📊 参数: {params}")
        response = session.get(url, params=params, timeout=15)

        logger.info(f"📊 响应状态码: {response.status_code}")
        logger.info(f"📊 响应头: {dict(response.headers)}")

        assert response.status_code == 200, f"直连Yahoo API失败，状态码: {response.status_code}"

        logger.info("✅ 直连Yahoo API成功")
        try:
            data = response.json()
            logger.info(f"📊 返回数据大小: {len(str(data))} 字符")
            if 'chart' in data and 'result' in data['chart']:
                logger.info(f"📊 数据点数量: {len(data['chart']['result'])}")
                assert len(data['chart']['result']) > 0, "API返回空数据"
        except:
            logger.info("📊 响应不是JSON格式，但状态码为200")
            assert response.status_code == 200, f"API响应状态码异常: {response.status_code}"

    except requests.exceptions.ConnectionError as e:
        logger.error(f"❌ 连接错误: {e}")
        raise AssertionError(f"连接错误: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"❌ 请求超时: {e}")
        raise AssertionError(f"请求超时: {e}")
    except Exception as e:
        logger.error(f"❌ 请求异常: {e}")
        raise AssertionError(f"请求异常: {e}")


def test_yahoo_api_via_proxy(proxy_url: str):
    """通过代理访问Yahoo API"""
    logger.info(f"🔍 开始测试通过代理访问Yahoo API: {proxy_url}")

    # 使用更复杂的浏览器模拟请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Referer': 'https://finance.yahoo.com/',
        'Origin': 'https://finance.yahoo.com',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-site',
        'X-Requested-With': 'XMLHttpRequest'
    }

    url = 'https://query2.finance.yahoo.com/v8/finance/chart/^GSPC'

    params = {
        'period1': int(time.time()) - 365 * 24 * 60 * 60,
        'period2': int(time.time()),
        'interval': '1d',
        'includePrePost': 'true',
        'events': 'div,splits,capitalGains'
    }

    try:
        session = requests.Session()
        session.mount('https://', SSLAdapter())

        # 设置代理
        session.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }

        # 设置重试策略
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,  # 减少退避时间
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 设置请求头
        session.headers.update(headers)

        logger.info(f"🌐 通过代理发送请求到: {url}")
        response = session.get(url, params=params, timeout=15)

        logger.info(f"📊 响应状态码: {response.status_code}")

        assert response.status_code == 200, f"通过代理访问Yahoo API失败，状态码: {response.status_code}"

        logger.info("✅ 通过代理访问Yahoo API成功")
        try:
            data = response.json()
            if 'chart' in data and 'result' in data['chart']:
                logger.info(f"📊 数据点数量: {len(data['chart']['result'])}")
                assert len(data['chart']['result']) > 0, "代理API返回空数据"
        except:
            logger.info("📊 响应不是JSON格式，但状态码为200")
            assert response.status_code == 200, f"代理API响应状态码异常: {response.status_code}"

    except Exception as e:
        logger.error(f"❌ 通过代理访问异常: {e}")
        raise AssertionError(f"通过代理访问异常: {e}")


def test_yfinance_provider():
    """测试YahooFinanceDataProvider"""
    logger.info("🔍 开始测试YahooFinanceDataProvider")

    try:
        from core_bak_refactored.core.data.providers.yahoo_provider import YahooFinanceDataProvider
        from datetime import datetime, timedelta
        import pandas as pd

        # 创建数据提供者实例
        provider = YahooFinanceDataProvider()

        # 设置测试日期范围
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=30)  # 30天前

        logger.info(f"📊 获取^GSPC数据，时间范围: {start_date.date()} 到 {end_date.date()}")

        # 获取数据
        price_data = provider._inter_get_index_prices("^GSPC", start_date, end_date, 'daily')

        assert price_data is not None, "YahooFinanceDataProvider返回None"
        assert hasattr(price_data, 'records'), "YahooFinanceDataProvider返回对象缺少records属性"
        assert price_data.records, "YahooFinanceDataProvider返回空数据"

        logger.info(f"✅ YahooFinanceDataProvider测试成功")
        logger.info(f"📊 获取到 {len(price_data.records)} 条数据记录")
        if price_data.records:
            logger.info(f"📊 数据时间范围: {price_data.records[0].timestamp} 到 {price_data.records[-1].timestamp}")

    except ImportError as e:
        logger.error(f"❌ 导入YahooFinanceDataProvider失败: {e}")
        raise AssertionError(f"导入YahooFinanceDataProvider失败: {e}")
    except Exception as e:
        logger.error(f"❌ YahooFinanceDataProvider测试失败: {e}")
        raise AssertionError(f"YahooFinanceDataProvider测试失败: {e}")


def test_advanced_yahoo_api():
    """测试更复杂的Yahoo API访问方法"""
    logger.info("🔍 开始测试高级Yahoo API访问方法")

    # 尝试模拟浏览器完整流程
    session = requests.Session()
    session.mount('https://', SSLAdapter())

    # 浏览器样式请求头
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache'
    }

    session.headers.update(headers)

    try:
        # 1. 访问Yahoo Finance主页获取cookies和crumb等信息
        logger.info("🌐 访问Yahoo Finance主页...")
        home_response = session.get('https://finance.yahoo.com/quote/^GSPC', timeout=10)

        # 2. 尝试从页面中提取crumb（如果有的话）
        crumb = None
        if home_response.status_code == 200:
            # 查找crumb的常见模式
            crumb_match = re.search(r'"crumb":"([^"]+)"', home_response.text)
            if crumb_match:
                crumb = crumb_match.group(1)
                logger.info(f"🔑 找到crumb: {crumb[:10]}...")

        # 3. 使用可能的crumb访问API
        if crumb:
            api_url = f'https://query2.finance.yahoo.com/v8/finance/chart/^GSPC'
            params = {
                'crumb': crumb,
                'period1': int(time.time()) - 30 * 24 * 60 * 60,  # 30天前
                'period2': int(time.time()),
                'interval': '1d'
            }
        else:
            # 如果没有crumb，使用基本参数
            api_url = 'https://query2.finance.yahoo.com/v8/finance/chart/^GSPC'
            params = {
                'period1': int(time.time()) - 30 * 24 * 60 * 60,
                'period2': int(time.time()),
                'interval': '1d'
            }

        logger.info(f"🌐 访问API: {api_url}")
        logger.info(f"📊 参数: {params}")

        api_response = session.get(api_url, params=params, timeout=15)

        logger.info(f"📊 API响应状态码: {api_response.status_code}")

        assert api_response.status_code == 200, f"高级Yahoo API访问失败，状态码: {api_response.status_code}"

        logger.info("✅ 高级Yahoo API访问成功")
        try:
            data = api_response.json()
            if 'chart' in data and 'result' in data['chart']:
                logger.info(f"📊 数据点数量: {len(data['chart']['result'])}")
                assert len(data['chart']['result']) > 0, "高级API返回空数据"
        except:
            logger.info("📊 响应不是JSON格式，但状态码为200")
            assert api_response.status_code == 200, f"高级API响应状态码异常: {api_response.status_code}"

    except Exception as e:
        logger.error(f"❌ 高级Yahoo API访问异常: {e}")
        raise AssertionError(f"高级Yahoo API访问异常: {e}")


def test_browser_simulation():
    """使用更高级的浏览器模拟测试Yahoo API"""
    logger.info("🔍 开始测试浏览器模拟访问Yahoo API")

    try:
        import cloudscraper  # 尝试使用cloudscraper库来绕过保护

        # 创建cloudscraper会话
        scraper = cloudscraper.create_scraper()

        # 设置更真实的请求头
        scraper.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Referer': 'https://finance.yahoo.com/',
            'Origin': 'https://finance.yahoo.com',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
            'X-Requested-With': 'XMLHttpRequest'
        })

        # 访问API
        url = 'https://query2.finance.yahoo.com/v8/finance/chart/^GSPC'
        params = {
            'period1': int(time.time()) - 30 * 24 * 60 * 60,
            'period2': int(time.time()),
            'interval': '1d'
        }

        # 先访问主页获取可能的认证信息
        logger.info("🌐 使用cloudscraper访问Yahoo Finance主页...")
        home_response = scraper.get('https://finance.yahoo.com/quote/^GSPC', timeout=15)

        # 尝试从页面中提取crumb
        crumb = None
        if home_response.status_code == 200:
            crumb_match = re.search(r'"crumb":"([^"]+)"', home_response.text)
            if crumb_match:
                crumb = crumb_match.group(1)
                logger.info(f"🔑 从主页找到crumb: {crumb[:10]}...")

                # 使用找到的crumb更新参数
                params['crumb'] = crumb

        logger.info(f"🌐 使用cloudscraper访问API: {url}")
        response = scraper.get(url, params=params, timeout=15)

        logger.info(f"📊 响应状态码: {response.status_code}")

        assert response.status_code == 200, f"浏览器模拟访问Yahoo API失败，状态码: {response.status_code}"

        logger.info("✅ 浏览器模拟访问Yahoo API成功")
        try:
            data = response.json()
            if 'chart' in data and 'result' in data['chart']:
                logger.info(f"📊 数据点数量: {len(data['chart']['result'])}")
                assert len(data['chart']['result']) > 0, "浏览器模拟API返回空数据"
        except:
            logger.info("📊 响应不是JSON格式，但状态码为200")
            assert response.status_code == 200, f"浏览器模拟API响应状态码异常: {response.status_code}"

    except ImportError:
        logger.info("💡 安装命令: pip install cloudscraper")
        raise AssertionError(f"cloudscraper库未安装，请执行: pip install cloudscraper")
    except Exception as e:
        logger.error(f"❌ 浏览器模拟访问异常: {e}")
        raise AssertionError(f"浏览器模拟访问异常: {e}")


def main():
    """主测试函数"""
    logger.info("🚀 开始Yahoo Finance API测试")

    # 从环境变量获取代理配置
    proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY') or os.environ.get('ALL_PROXY')

    results = {}

    # 1. 测试代理连接（如果配置了代理）
    if proxy_url:
        logger.info(f"📡 检测到代理配置: {proxy_url}")
        try:
            test_proxy_connection(proxy_url)
            results['proxy_connection'] = True
            logger.info("✅ 代理连接测试通过")
        except AssertionError as e:
            logger.error(f"❌ 代理连接测试失败: {e}")
            results['proxy_connection'] = False
    else:
        logger.info("📡 未检测到代理配置")

    # 2. 测试直连Yahoo API
    try:
        test_yahoo_api_direct()
        results['direct_api'] = True
        logger.info("✅ 直连Yahoo API测试通过")
    except AssertionError as e:
        logger.error(f"❌ 直连Yahoo API测试失败: {e}")
        results['direct_api'] = False

    # 3. 如果配置了代理，测试通过代理访问
    if proxy_url:
        try:
            test_yahoo_api_via_proxy(proxy_url)
            results['proxy_api'] = True
            logger.info("✅ 通过代理访问Yahoo API测试通过")
        except AssertionError as e:
            logger.error(f"❌ 通过代理访问Yahoo API测试失败: {e}")
            results['proxy_api'] = False

    # 4. 测试高级API访问方法
    try:
        test_advanced_yahoo_api()
        results['advanced_api'] = True
        logger.info("✅ 高级Yahoo API访问测试通过")
    except AssertionError as e:
        logger.error(f"❌ 高级Yahoo API访问测试失败: {e}")
        results['advanced_api'] = False

    # 5. 测试浏览器模拟访问（使用cloudscraper）
    try:
        test_browser_simulation()
        results['browser_sim'] = True
        logger.info("✅ 浏览器模拟访问测试通过")
    except AssertionError as e:
        logger.error(f"❌ 浏览器模拟访问测试失败: {e}")
        results['browser_sim'] = False

    # 6. 测试YahooFinanceDataProvider
    try:
        test_yfinance_provider()
        results['provider'] = True
        logger.info("✅ YahooFinanceDataProvider测试通过")
    except AssertionError as e:
        logger.error(f"❌ YahooFinanceDataProvider测试失败: {e}")
        results['provider'] = False

    # 输出测试结果摘要
    logger.info("\n" + "=" * 60)
    logger.info("📋 测试结果摘要:")
    for test_name, result in results.items():
        status = "✅ 成功" if result else "❌ 失败"
        logger.info(f"   {test_name}: {status}")

    # 分析和建议
    logger.info("\n" + "=" * 60)
    logger.info("💡 问题分析和建议:")

    if not results.get('direct_api', True):
        logger.info("   1. 直连API失败，可能的原因:")
        logger.info("      - 请求头不够真实，需要更完整地模拟浏览器")
        logger.info("      - 缺少必要的Cookie或认证信息")
        logger.info("      - 需要实现更复杂的反爬虫绕过机制")
        logger.info("      - 可能需要使用selenium或requests-html模拟完整浏览器行为")

    if proxy_url and results.get('proxy_connection', False):
        if not results.get('proxy_api', True):
            logger.info("   2. 代理连接正常但代理访问API失败，可能的原因:")
            logger.info("      - 代理IP被Yahoo封禁")
            logger.info("      - 代理请求头仍需优化")
        else:
            logger.info("   2. 代理访问成功，建议在生产环境中使用代理")

    if results.get('advanced_api', False):
        logger.info("   3. 高级API访问成功，说明通过模拟浏览器流程可以绕过限制")
    else:
        logger.info("   3. 高级API访问失败，可能需要更复杂的浏览器模拟")

    if results.get('browser_sim', False):
        logger.info("   4. 浏览器模拟访问成功，cloudscraper可能是一个好的解决方案")

    if results.get('provider', False):
        logger.info("   4. YahooFinanceDataProvider工作正常")
    else:
        logger.info("   4. YahooFinanceDataProvider存在问题，需要检查实现")

    logger.info("\n" + "=" * 60)
    logger.info("🔧 针对429错误的改进建议:")
    logger.info("   • 实现更真实的浏览器请求头")
    logger.info("   • 添加Cookie管理和会话保持")
    logger.info("   • 考虑使用cloudscraper库绕过保护")
    logger.info("   • 使用多个User-Agent轮换")
    logger.info("   • 实现更智能的重试机制")
    logger.info("   • 考虑使用selenium或playwright进行浏览器自动化")


if __name__ == "__main__":
    main()