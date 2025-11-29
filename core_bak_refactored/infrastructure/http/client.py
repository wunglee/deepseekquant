"""
HTTP客户端设置模块（从 DataFetcher._setup_http_client 迁移而来）

职责：
1. 配置异步HTTP客户端（aiohttp）
2. 配置同步HTTP会话（requests）
3. 设置连接池和超时参数
4. 配置重试策略
"""
from typing import Dict, Any
import logging
import aiohttp
import requests

logger = logging.getLogger(__name__)


def setup_http_client(config: Dict[str, Any]) -> tuple:
    """
    设置HTTP客户端（从 DataFetcher._setup_http_client 迁移而来）。
    
    创建并配置异步和同步HTTP客户端：
    - aiohttp.ClientSession：用于异步请求
    - requests.Session：用于同步请求（如果需要）
    
    Args:
        config: 配置字典，包含超时、连接池等参数
    
    Returns:
        (aiohttp_session, requests_session) 元组
    
    Example:
        >>> config = {'request_timeout': 30, 'max_connections': 100}
        >>> aiohttp_sess, requests_sess = setup_http_client(config)
    """
    try:
        # 获取配置参数
        request_timeout = config.get('request_timeout', 30)
        max_connections = config.get('max_connections', 100)
        max_connections_per_host = config.get('max_connections_per_host', 20)
        user_agent = config.get('user_agent', 'DeepSeekQuant/1.0.0')
        
        # 设置异步HTTP客户端（aiohttp）
        aiohttp_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout),
            connector=aiohttp.TCPConnector(
                limit=max_connections,
                limit_per_host=max_connections_per_host,
                ttl_dns_cache=300  # DNS缓存时间（秒）
            ),
            headers={'User-Agent': user_agent}
        )
        
        logger.info(
            f"异步HTTP客户端已配置: "
            f"超时={request_timeout}s, "
            f"最大连接数={max_connections}, "
            f"每主机最大连接数={max_connections_per_host}"
        )
        
        # 设置同步HTTP会话（requests）
        requests_session = requests.Session()
        
        # 配置连接池
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_connections,
            pool_maxsize=max_connections,
            max_retries=3  # 自动重试次数
        )
        
        # 为HTTP和HTTPS都挂载适配器
        requests_session.mount('http://', adapter)
        requests_session.mount('https://', adapter)
        
        # 设置默认请求头
        requests_session.headers.update({'User-Agent': user_agent})
        
        logger.info("同步HTTP会话已配置: 连接池大小={}, 最大重试次数=3".format(max_connections))
        
        return aiohttp_session, requests_session
        
    except Exception as e:
        logger.error(f"HTTP客户端设置失败: {e}")
        raise


async def close_http_client(aiohttp_session: aiohttp.ClientSession, requests_session: requests.Session):
    """
    关闭HTTP客户端，释放资源。
    
    Args:
        aiohttp_session: aiohttp会话对象
        requests_session: requests会话对象
    """
    try:
        # 关闭异步会话
        if aiohttp_session and not aiohttp_session.closed:
            await aiohttp_session.close()
            logger.info("异步HTTP客户端已关闭")
        
        # 关闭同步会话
        if requests_session:
            requests_session.close()
            logger.info("同步HTTP会话已关闭")
            
    except Exception as e:
        logger.error(f"关闭HTTP客户端失败: {e}")
