"""
API凭证管理模块（从 DataFetcher._setup_api_credentials 迁移而来）

职责：
1. 加载和管理API凭证
2. 支持多种认证类型（API Key, OAuth, Token等）
3. 安全存储敏感信息
4. 凭证验证和更新
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def setup_api_credentials(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    设置API认证信息（从 DataFetcher._setup_api_credentials 迁移而来）。
    
    从配置中提取各数据源的API凭证，支持多种认证类型：
    - api_key: API密钥认证
    - oauth: OAuth 2.0认证
    - token: Bearer Token认证
    - basic: HTTP Basic认证
    
    Args:
        config: 配置字典，包含sources配置
    
    Returns:
        凭证字典，键为数据源类型，值为凭证信息
    
    Example:
        >>> config = {
        ...     'sources': {
        ...         'yahoo_finance': {'enabled': True, 'api_key': 'xxx'},
        ...         'alpha_vantage': {'enabled': True, 'api_key': 'yyy'}
        ...     }
        ... }
        >>> credentials = setup_api_credentials(config)
        >>> # {'yahoo_finance': {'api_key': 'xxx', ...}, ...}
    """
    try:
        api_credentials = {}
        sources_config = config.get('sources', {})
        
        # 定义所有可能的数据源类型
        source_types = [
            'yahoo_finance',
            'alpha_vantage',
            'iex_cloud',
            'polygon',
            'twelve_data',
            'finnhub',
            'tiingo',
            'quandl',
            'intrinio',
            'eod_historical',
            'custom_api',
            'database',
            'broker_api'
        ]
        
        # 遍历每个数据源配置
        for source_type in source_types:
            source_config = sources_config.get(source_type, {})
            
            # 只处理已启用的数据源
            if not source_config.get('enabled', False):
                logger.debug(f"数据源 {source_type} 未启用，跳过凭证设置")
                continue
            
            # 提取凭证信息
            credentials_info = {
                'api_key': source_config.get('api_key', ''),
                'secret_key': source_config.get('secret_key', ''),
                'access_token': source_config.get('access_token', ''),
                'refresh_token': source_config.get('refresh_token', ''),
                'base_url': source_config.get('base_url', ''),
                'rate_limit': source_config.get('rate_limit', {}),
                'authentication_type': source_config.get('authentication_type', 'api_key'),
                'username': source_config.get('username', ''),
                'password': source_config.get('password', '')
            }
            
            # 验证凭证
            if _validate_credentials(source_type, credentials_info):
                api_credentials[source_type] = credentials_info
                logger.info(f"数据源 {source_type} 凭证已配置 (认证类型: {credentials_info['authentication_type']})")
            else:
                logger.warning(f"数据源 {source_type} 凭证验证失败，跳过")
        
        logger.info(f"API凭证设置完成，共配置 {len(api_credentials)} 个数据源")
        return api_credentials
        
    except Exception as e:
        logger.error(f"API凭证设置失败: {e}")
        return {}


def _validate_credentials(source_type: str, credentials: Dict[str, Any]) -> bool:
    """
    验证凭证的有效性。
    
    Args:
        source_type: 数据源类型
        credentials: 凭证信息
    
    Returns:
        True如果凭证有效，False否则
    """
    try:
        auth_type = credentials.get('authentication_type', 'api_key')
        
        # 根据认证类型验证必需字段
        if auth_type == 'api_key':
            # API Key认证必须有api_key
            if not credentials.get('api_key'):
                logger.warning(f"{source_type}: API Key认证缺少api_key")
                return False
        
        elif auth_type == 'oauth':
            # OAuth认证必须有access_token
            if not credentials.get('access_token'):
                logger.warning(f"{source_type}: OAuth认证缺少access_token")
                return False
        
        elif auth_type == 'token':
            # Token认证必须有access_token或api_key
            if not credentials.get('access_token') and not credentials.get('api_key'):
                logger.warning(f"{source_type}: Token认证缺少access_token或api_key")
                return False
        
        elif auth_type == 'basic':
            # Basic认证必须有username和password
            if not credentials.get('username') or not credentials.get('password'):
                logger.warning(f"{source_type}: Basic认证缺少username或password")
                return False
        
        # 所有认证类型都应该有base_url（可选，使用默认值）
        if not credentials.get('base_url'):
            logger.debug(f"{source_type}: 未配置base_url，将使用默认值")
        
        return True
        
    except Exception as e:
        logger.error(f"凭证验证失败 ({source_type}): {e}")
        return False


def get_credentials(api_credentials: Dict[str, Dict[str, Any]], source_type: str) -> Dict[str, Any]:
    """
    获取指定数据源的凭证。
    
    Args:
        api_credentials: 所有凭证字典
        source_type: 数据源类型
    
    Returns:
        凭证信息，如果不存在则返回空字典
    
    Example:
        >>> creds = get_credentials(api_credentials, 'yahoo_finance')
        >>> api_key = creds.get('api_key')
    """
    return api_credentials.get(source_type, {})


def update_credentials(
    api_credentials: Dict[str, Dict[str, Any]], 
    source_type: str, 
    new_credentials: Dict[str, Any]
) -> bool:
    """
    更新指定数据源的凭证。
    
    Args:
        api_credentials: 所有凭证字典（会被修改）
        source_type: 数据源类型
        new_credentials: 新的凭证信息
    
    Returns:
        True如果更新成功，False否则
    
    Example:
        >>> success = update_credentials(
        ...     api_credentials, 
        ...     'yahoo_finance', 
        ...     {'api_key': 'new_key'}
        ... )
    """
    try:
        # 验证新凭证
        if not _validate_credentials(source_type, new_credentials):
            logger.error(f"新凭证验证失败 ({source_type})")
            return False
        
        # 更新凭证
        api_credentials[source_type] = new_credentials
        logger.info(f"凭证已更新: {source_type}")
        return True
        
    except Exception as e:
        logger.error(f"更新凭证失败 ({source_type}): {e}")
        return False


def mask_sensitive_data(credentials: Dict[str, Any]) -> Dict[str, Any]:
    """
    掩码敏感凭证数据（用于日志记录）。
    
    Args:
        credentials: 凭证字典
    
    Returns:
        掩码后的凭证字典
    
    Example:
        >>> masked = mask_sensitive_data({'api_key': 'secret123'})
        >>> # {'api_key': '***cret123'}
    """
    masked = credentials.copy()
    sensitive_keys = ['api_key', 'secret_key', 'access_token', 'refresh_token', 'password']
    
    for key in sensitive_keys:
        if key in masked and masked[key]:
            value = str(masked[key])
            if len(value) > 6:
                # 只显示最后3个字符
                masked[key] = '*' * (len(value) - 3) + value[-3:]
            else:
                masked[key] = '***'
    
    return masked
