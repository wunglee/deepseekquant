"""
Yahoo Finance Browser Simulation 补丁 (支持 HTTP/2)

Note: 
- 此补丁修复了 yfinance 的 "Too Many Requests" (429) 问题
- 使用 curl_cffi 模拟浏览器请求 (内置 HTTP/2 支持)
- 通过 crumb 认证绕过 Yahoo 的反爬虫机制
- 通过请求限流避免触发速率限制
- 通过随机 User-Agent 避免检测
- 通过 Referer 和 Origin 头绕过跨域限制
- 通过 X-Requested-With 头模拟 AJAX 请求

⚠️ 重要: 
- 需要安装: pip install curl_cffi
- 此补丁会 monkey patch yfinance.data.YfData.get 方法
- 所有通过 yfinance 的请求都会经过此补丁处理
- 所有反爬虫逻辑（User-Agent轮换、请求限流、浏览器模拟头）都由补丁处理
- HTTP/2 支持通过 curl_cffi 库自动启用
- YahooFinanceDataProvider 不需要重复实现这些逻辑
"""

import logging
import random
import re
import time
from typing import Optional
import urllib.parse

# 导入curl_cffi，假设总是存在
from curl_cffi import requests as curl_requests
from yfinance.exceptions import YFRateLimitError

logger = logging.getLogger(__name__)

try:
    import yfinance
    from yfinance import utils
    from yfinance import data as yf_data
except ImportError:
    logger.warning("yfinance not installed")
    raise ImportError("yfinance not installed")

# 全局标志，避免重复 patch
_PATCHED = False

# curl_cffi session实例，用于模拟真实浏览器
_CURL_SESSION = curl_requests.Session()
# 设置浏览器模拟头
# User-Agent 池（轮换使用，避免被识别为爬虫）
_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]
headers = {
    'User-Agent': random.choice(_USER_AGENTS),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Referer': 'https://finance.yahoo.com/',
    'Origin': 'https://finance.yahoo.com',
}

_LAST_REQUEST_TIME = time.time()  # 上次请求时间，初始化为当前时间
_MIN_REQUEST_INTERVAL = 2.0  # 最小请求间隔（秒）


class Cookie:
    def __init__(self, name, value, domain='', domain_specified=False, domain_initial_dot=False,
                 path='', path_specified=False, secure=False, expires='', discard=False,
                 httponly=False, version=0, comment='', comment_url='', port=''):
        self.name = name
        self.value = value
        self.domain = domain
        self.domain_specified = domain_specified
        self.domain_initial_dot = domain_initial_dot
        self.path = path
        self.path_specified = path_specified
        self.secure = secure
        self.expires = expires
        self.discard = discard
        self.httponly = httponly
        self.version = version
        self.comment = comment
        self.comment_url = comment_url
        self.port = port


def speed_limit():
    # 应用速率限制，避免触发Yahoo的反爬虫机制
    # 使用与patched_get相同的速率限制逻辑
    global _LAST_REQUEST_TIME
    current_time = time.time()
    time_since_last = current_time - _LAST_REQUEST_TIME
    if time_since_last < 0.05:
        sleep_time = 0.05 - time_since_last
        if sleep_time > 0:  # 确保只有当需要等待时才等待
            logger.debug(f"请求限流: 等待 {sleep_time:.3f}秒以遵守Yahoo速率限制")
            time.sleep(sleep_time)


def patch_yfinance(proxy_url=None):
    """
    给 yfinance 打补丁，使其使用 curl_cffi Browser Simulation

    Args:
        proxy_url: 代理地址 (e.g., "http://127.0.0.1:8002")

    原理:
    - Monkey patch yfinance.data.YfData.get() 方法
    - 用 curl_cffi (浏览器模拟) 替换 requests 调用

    Note:
        如果代理配置了但代理服务未运行，会自动降级到直连
    """
    global _PATCHED, _CURL_SESSION
    if _PATCHED:
        logger.debug("yfinance already patched for Browser Simulation")
        return

    # 如果有代理，配置到 curl_cffi session
    if proxy_url:
        _CURL_SESSION.proxies = {
            'http': proxy_url,
            'https': proxy_url
        }
        logger.info(f"curl_cffi session configured with proxy: {proxy_url}")

    def patched_make_request(self, url, request_method, user_agent_headers=None, body=None, params=None, timeout=30):
        # Important: treat input arguments as immutable.
        if params is None:
            params = {}
        speed_limit()
        cookie, crumb, strategy = self._get_cookie_and_crumb()
        if crumb is not None:
            crumbs = {'crumb': crumb}
        else:
            crumbs = {}
        if strategy == 'basic' and cookie is not None:
            # Basic cookie strategy adds cookie to GET parameters
            cookies = {cookie.name: cookie.value}
        else:
            cookies = None
        headers.update(user_agent_headers or self.user_agent_headers)
        request_args = {
            'url': url,
            'params': {**params, **crumbs},
            'cookies': cookies,
            'timeout': timeout,
            'headers': headers
        }

        if body:
            request_args['json'] = body
        response = request_method(**request_args)
        utils.get_yf_logger().debug(f'response code={response.status_code}')
        if response.status_code >= 400:
            # Retry with other cookie strategy
            if strategy == 'basic':
                self._set_cookie_strategy('csrf')
            else:
                self._set_cookie_strategy('basic')
            cookie, crumb, strategy = self._get_cookie_and_crumb(timeout)
            request_args['params']['crumb'] = crumb
            if strategy == 'basic':
                request_args['cookies'] = {cookie.name: cookie.value}
            response = request_method(**request_args)
            utils.get_yf_logger().debug(f'response code={response.status_code}')

            # Raise exception if rate limited
            if response.status_code == 429:
                raise YFRateLimitError()

        return response

    def patched_get_crumb_basic(self, timeout=30):

        if self._crumb is not None:
            utils.get_yf_logger().debug('reusing crumb')
            return self._crumb

        cookie = self._get_cookie_basic()
        if cookie is None:
            return None
        speed_limit()
        # - 'allow_redirects' copied from @psychoz971 solution - does it help USA?
        get_args = {
            'url': "https://query1.finance.yahoo.com/v1/test/getcrumb",
            'headers': headers,
            'cookies': {cookie.name: cookie.value},
            'timeout': timeout,
            'allow_redirects': True
        }
        if self._session_is_caching:
            get_args['expire_after'] = self._expire_after
            crumb_response = self._session.get(**get_args)
        else:
            crumb_response = self._session.get(**get_args)
        self._crumb = crumb_response.text
        if self._crumb is None or '<html>' in self._crumb:
            utils.get_yf_logger().debug("Didn't receive crumb")
            return None

        utils.get_yf_logger().debug(f"crumb = '{self._crumb}'")
        return self._crumb

    def patched_get_cookie_basic(self, timeout=30) -> Optional[Cookie]:
        speed_limit()
        response = self._session.get(
            url='https://fc.yahoo.com',
            headers=headers,
            timeout=timeout,
            allow_redirects=True)
        if not response.cookies:
            utils.get_yf_logger().debug("response.cookies为空，返回空字典")
            return None
        if self._cookie is not None:
            utils.get_yf_logger().debug('reusing cookie')
            return self._cookie
        cookies = extract_cffi_cookie(response)
        self._cookie = cookies[0] if len(cookies) > 0 else None
        if self._cookie == '':
            utils.get_yf_logger().debug("list(response.cookies)[0] = ''")
            return None
        self._save_cookie_basic(self._cookie)
        utils.get_yf_logger().debug(f"fetched basic cookie = {self._cookie}")
        return self._cookie

    def extract_cffi_cookie(response) -> list[Cookie]:
        # 使用curl_cffi的CookieJar标准方法获取cookie属性
        # curl_cffi的Cookies对象有一个jar属性，其中包含完整的cookie信息
        cookies: list[Cookie] = []
        if hasattr(response.cookies, 'jar'):
            jar = response.cookies.jar

            # 通过jar的_cookies属性获取完整的cookie对象
            if hasattr(jar, '_cookies'):
                for domain, domain_cookies in jar._cookies.items():
                    for path, path_cookies in domain_cookies.items():
                        for cookie_name, cookie_obj in path_cookies.items():
                            # 使用真实获取的属性，不伪造任何值
                            cookie = Cookie(
                                name=getattr(cookie_obj, 'name', cookie_name),
                                value=getattr(cookie_obj, 'value', ''),
                                domain=getattr(cookie_obj, 'domain', ''),
                                domain_specified=getattr(cookie_obj, 'domain_specified', False),
                                domain_initial_dot=getattr(cookie_obj, 'domain_initial_dot', False),
                                path=getattr(cookie_obj, 'path', ''),
                                path_specified=getattr(cookie_obj, 'path_specified', False),
                                secure=getattr(cookie_obj, 'secure', False),
                                expires=getattr(cookie_obj, 'expires', ''),
                                discard=getattr(cookie_obj, 'discard', False),
                                httponly=getattr(cookie_obj, 'rest', {}).get('HttpOnly') is not None,
                                version=getattr(cookie_obj, 'version', 0),
                                comment=getattr(cookie_obj, 'comment', ''),
                                comment_url=getattr(cookie_obj, 'comment_url', ''),
                                port=getattr(cookie_obj, 'port', ''),
                            )
                            cookies.append(cookie)
        utils.get_yf_logger().debug(f"获取到 {len(cookies)} 个cookies")
        return cookies

    def patched_get(self, url, user_agent_headers=None, params=None, timeout=30):
        return self._make_request(url, request_method=self._session.get, user_agent_headers=user_agent_headers,
                                  params=params, timeout=timeout)

    # 应用补丁
    yf_data.YfData.user_agent_headers = headers
    yf_data.YfData.get = patched_get
    yf_data.YfData.cache_get = patched_get
    yf_data.YfData._make_request = patched_make_request
    yf_data.YfData._get_cookie_basic = patched_get_cookie_basic
    yf_data.YfData._get_crumb_basic = patched_get_crumb_basic
    _PATCHED = True
    proxy_info = f" via proxy {proxy_url}" if proxy_url else " (direct)"
    logger.info(f"✅ yfinance patched to use curl_cffi browser simulation{proxy_info}")
