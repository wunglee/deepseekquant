"""
Redis缓存设置模块（从 DataFetcher._setup_redis_cache 迁移而来）

职责：
1. 配置Redis连接
2. 测试Redis可用性
3. 设置连接参数和超时
4. 处理连接失败情况
"""
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def setup_redis_cache(config: Dict[str, Any]):
    """
    设置Redis缓存（从 DataFetcher._setup_redis_cache 迁移而来）。
    
    根据配置创建Redis客户端连接，并测试连接可用性。
    
    Args:
        config: Redis配置字典，包含host、port、db、password等
    
    Returns:
        Redis客户端对象，失败返回None
    
    Example:
        >>> config = {
        ...     'redis': {
        ...         'enabled': True,
        ...         'host': 'localhost',
        ...         'port': 6379,
        ...         'db': 0,
        ...         'password': None
        ...     }
        ... }
        >>> redis_client = setup_redis_cache(config)
    """
    try:
        import redis
        
        redis_config = config.get('redis', {})
        
        # 检查是否启用Redis
        if not redis_config.get('enabled', False):
            logger.info("Redis缓存未启用")
            return None
        
        # 提取Redis配置参数
        host = redis_config.get('host', 'localhost')
        port = redis_config.get('port', 6379)
        db = redis_config.get('db', 0)
        password = redis_config.get('password')
        socket_timeout = redis_config.get('socket_timeout', 5)
        socket_connect_timeout = redis_config.get('socket_connect_timeout', 5)
        max_connections = redis_config.get('max_connections', 50)
        
        # 创建连接池
        pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=False,  # 不自动解码，保留二进制数据
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            max_connections=max_connections,
            retry_on_timeout=True
        )
        
        # 创建Redis客户端
        redis_client = redis.Redis(connection_pool=pool)
        
        # 测试连接
        redis_client.ping()
        
        logger.info(
            f"Redis缓存连接成功: {host}:{port}/{db}, "
            f"最大连接数={max_connections}, "
            f"超时={socket_timeout}s"
        )
        
        return redis_client
        
    except ImportError:
        logger.warning("Redis库未安装，跳过Redis缓存设置")
        return None
        
    except Exception as e:
        logger.error(f"Redis连接失败: {e}")
        return None


def check_redis_connection(redis_client: Any) -> bool:
    """
    测试Redis连接是否可用。
    
    Args:
        redis_client: Redis客户端对象
    
    Returns:
        True如果连接可用，False否则
    """
    try:
        if redis_client is None:
            return False
        
        # 发送PING命令
        response = redis_client.ping()
        
        if response:
            logger.debug("Redis连接检查成功")
            return True
        else:
            logger.warning("Redis PING响应异常")
            return False
            
    except Exception as e:
        logger.error(f"Redis连接检查失败: {e}")
        return False


def close_redis_connection(redis_client: Any):
    """
    关闭Redis连接。
    
    Args:
        redis_client: Redis客户端对象
    """
    try:
        if redis_client:
            redis_client.close()
            logger.info("Redis连接已关闭")
    except Exception as e:
        logger.error(f"关闭Redis连接失败: {e}")
