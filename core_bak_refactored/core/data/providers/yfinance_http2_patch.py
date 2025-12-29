"""yfinance HTTP/2 补丁 + User-Agent 轮换

修复 Yahoo Finance "Too Many Requests" 错误的完整解决方案：
1. HTTP/2 支持 - Yahoo Finance API 要求 HTTP/2
2. User-Agent 轮换 - 避免被识别为爬虫
3. Session 复用 - 保持 cookies 和连接
4. 请求延迟 - 避免"伪高频"请求

依赖: pip install 'httpx[http2]'

参考资料:
- https://github.com/ranaroussi/yfinance/issues/2125
- https://stackoverflow.com/questions/78111453
- https://blog.ni18.in/how-to-fix-the-yfinance-429-client-error
"""

import logging
import random
import time


logger = logging.getLogger(__name__)

# 全局标志，避免重复 patch
_PATCHED = False
_HTTP2_CLIENT = None  # 全局 HTTP/2 客户端
_LAST_REQUEST_TIME = 0  # 上次请求时间
_MIN_REQUEST_INTERVAL = 2.0  # 最小请求间隔（秒）

# User-Agent 池（轮换使用，避免被识别为爬虫）
_USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
]


def patch_yfinance(proxy_url=None):
    """
    给 yfinance 打补丁，使其使用 HTTP/2
    
    Args:
        proxy_url: 代理地址 (e.g., "http://127.0.0.1:8002")
    
    原理:
    - Monkey patch yfinance.data.YfData.get() 方法
    - 用 httpx.Client (支持 HTTP/2) 替换 requests 调用
    
    Note:
        如果代理配置了但代理服务未运行，会自动降级到直连
    """
    global _PATCHED, _HTTP2_CLIENT
    
    if _PATCHED:
        logger.debug("yfinance already patched for HTTP/2")
        return
    
    try:
        import httpx
    except ImportError:
        logger.warning(
            "httpx not installed, cannot patch yfinance for HTTP/2\n"
            "Install with: pip install 'httpx[http2]'"
        )
        return
    
    try:
        import yfinance
        from yfinance import data as yf_data
    except ImportError:
        logger.warning("yfinance not installed")
        return
    
    # 保存原始方法
    original_get = yf_data.YfData.get
    
    # 创建全局 httpx client (复用连接)
    client_kwargs = {
        'http2': True,
        'follow_redirects': True,
        'timeout': 30.0
    }
    
    # 如果有代理，配置 proxy
    proxy_available = False
    if proxy_url:
        try:
            # 先测试代理是否可用
            test_client = httpx.Client(proxy=proxy_url, timeout=5.0)
            try:
                # 测试代理连接（访问一个轻量级的 API）
                response = test_client.get('https://httpbin.org/ip')
                if response.status_code == 200:
                    proxy_available = True
                    logger.info(f"✅ Yahoo Finance: 代理可用 {proxy_url} (IP: {response.json().get('origin', 'unknown')})")
                else:
                    logger.warning(f"⚠️ Yahoo Finance: 代理返回错误状态码 {response.status_code}")
            except Exception as e:
                logger.warning(f"⚠️ Yahoo Finance: 代理测试失败 {proxy_url}: {e}")
            finally:
                test_client.close()
        except Exception as e:
            logger.warning(f"⚠️ Yahoo Finance: 无法连接到代理 {proxy_url}: {e}")
        
        if proxy_available:
            # httpx 0.28+ 参数名是 'proxy' 不是 'proxies'
            client_kwargs['proxy'] = proxy_url
            logger.info(f"Yahoo Finance: 将使用代理 {proxy_url}")
        else:
            logger.warning(
                f"Yahoo Finance: 代理 {proxy_url} 不可用，将使用直连\n"
                f"建议: 1) 检查代理服务是否运行 (如 v2ray, clash)\n"
                f"      2) 检查代理端口是否正确\n"
                f"      3) 或将 data_provider.yml 中 yahoo_finance.use_proxy 设为 false"
            )
    
    _HTTP2_CLIENT = httpx.Client(**client_kwargs)
    
    def patched_get(self, url, user_agent_headers=None, params=None, timeout=30):
        """
        使用 httpx (HTTP/2) 替代 requests
        
        关键优化：
        1. 轮换 User-Agent - 避免被识别为爬虫
        2. 请求限流 - 避免"伪高频"请求
        3. 复用 Session - 保持 cookies
        """
        global _LAST_REQUEST_TIME
        
        try:
            # 1. 请求限流：确保两次请求间隔至少 2 秒
            current_time = time.time()
            time_since_last = current_time - _LAST_REQUEST_TIME
            if time_since_last < _MIN_REQUEST_INTERVAL:
                sleep_time = _MIN_REQUEST_INTERVAL - time_since_last
                logger.debug(f"请求限流: 等待 {sleep_time:.2f}秒")
                time.sleep(sleep_time)
            
            # 2. 轮换 User-Agent（关键！）
            headers = user_agent_headers.copy() if user_agent_headers else {}
            # 随机选择一个 User-Agent
            user_agent = random.choice(_USER_AGENTS)
            headers['User-Agent'] = user_agent
            # 添加其他常见的浏览器头
            headers.setdefault('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8')
            headers.setdefault('Accept-Language', 'en-US,en;q=0.9')
            headers.setdefault('Accept-Encoding', 'gzip, deflate, br')
            headers.setdefault('Connection', 'keep-alive')
            headers.setdefault('Upgrade-Insecure-Requests', '1')
            
            # 3. 发送请求（使用 HTTP/2）
            logger.info(f"📡 HTTP/2 请求: {url[:100]}... (UA: {user_agent[:50]}...)")
            response = _HTTP2_CLIENT.get(url, params=params, headers=headers, timeout=timeout)
            
            # 更新最后请求时间
            _LAST_REQUEST_TIME = time.time()
            
            # 检查状态码
            if response.status_code >= 400:
                logger.error(f"❌ HTTP/2 响应错误: {response.status_code} - {url[:100]}...")
                response.raise_for_status()
            
            logger.info(f"✅ HTTP/2 请求成功: {response.status_code} - {url[:100]}...")
            # 返回 response (httpx.Response 与 requests.Response 兼容)
            return response
            
        except Exception as e:
            logger.warning(f"HTTP/2 request failed: {e}, falling back to original method")
            # 降级到原始方法
            return original_get(self, url, user_agent_headers, params, timeout)
    
    # 应用补丁
    yf_data.YfData.get = patched_get
    
    _PATCHED = True
    proxy_info = f" via proxy {proxy_url}" if proxy_url else " (direct)"
    logger.info(f"✅ yfinance patched to use HTTP/2{proxy_info}")


# 自动应用补丁（导入时执行）
# 注意：代理配置需要在 Yahoo Provider 初始化时传入
# 这里只是预加载 patch 函数，不立即执行
# patch_yfinance()  # 先注释，由 Yahoo Provider 调用
