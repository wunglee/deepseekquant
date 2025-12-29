"""API路由装饰器

[应用层 - API组件] 第五轮迁移 - 统一错误处理
状态: ✅ 消除路由中的重复错误处理代码
迁移时间: 2025-11-28

包含功能:
- 统一异常处理装饰器
- 统一响应格式装饰器
- 减少样板代码
"""

import logging
from functools import wraps
from typing import Callable

import pandas as pd
from flask import jsonify

logger = logging.getLogger('DeepSeekQuant.App.API.RouteDecorators')


def handle_api_errors(error_code_prefix: str = 'API'):
    """统一API错误处理装饰器
    
    Args:
        error_code_prefix: 错误代码前缀
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # 如果结果已经是Response对象，直接返回
                if hasattr(result, 'status_code'):
                    return result
                # 否则包装成标准响应
                return jsonify({
                    'status': 'success',
                    **result,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
            except Exception as e:
                logger.error(f"{func.__name__} 失败: {e}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'error_code': f'{error_code_prefix}_{func.__name__.upper()}_FAILED'
                }), 500
        return wrapper
    return decorator


def api_response(func: Callable) -> Callable:
    """统一API响应格式装饰器（不处理异常）
    
    Args:
        func: 被装饰的函数
        
    Returns:
        包装后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # 如果结果已经是Response对象或dict，直接返回
        if hasattr(result, 'status_code') or isinstance(result, dict):
            return result
        # 否则包装成标准响应
        return jsonify({
            'status': 'success',
            'data': result,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    return wrapper
