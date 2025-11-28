"""
统一异常处理基础设施
从risk模块107处异常处理模式中提炼的通用装饰器
"""

import functools
import logging
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger('DeepSeekQuant.ErrorHandling')

T = TypeVar('T')


def safe_execute(
    default_return: Any = None,
    log_level: str = 'error',
    exc_info: bool = True,
    reraise: bool = False
) -> Callable:
    """
    通用安全执行装饰器
    
    统一107处异常处理模式：
    try:
        ... 业务逻辑 ...
    except Exception as e:
        logger.error(f"...: {e}")
        return default_value
    
    Args:
        default_return: 异常时的返回值（可以是值或callable）
        log_level: 日志级别 ('debug'/'info'/'warning'/'error')
        exc_info: 是否记录完整堆栈
        reraise: 是否重新抛出异常（用于需要上层感知的场景）
    
    示例:
        @safe_execute(default_return=0.0, log_level='error')
        def calculate_risk(data):
            return complex_calculation(data)
        
        @safe_execute(default_return=lambda: {'var': 0.0}, log_level='warning')
        def get_risk_metrics(data):
            return {'var': calculate_var(data)}
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 获取logger
                log_func = getattr(logger, log_level, logger.error)
                
                # 记录异常
                error_msg = f"{func.__name__}失败: {e}"
                if exc_info:
                    log_func(error_msg, exc_info=True)
                else:
                    log_func(error_msg)
                
                # 重新抛出（如果需要）
                if reraise:
                    raise
                
                # 返回默认值
                if callable(default_return):
                    return default_return()
                return default_return
        
        return wrapper
    return decorator


def safe_numeric_operation(
    default_return: Union[float, int] = 0.0,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> Callable:
    """
    数值计算专用安全装饰器
    
    增强功能：
    - 异常捕获
    - NaN/Inf检查
    - 自动类型转换
    
    Args:
        default_return: 异常/无效时返回值
        allow_nan: 是否允许NaN结果
        allow_inf: 是否允许Inf结果
    
    示例:
        @safe_numeric_operation(default_return=0.0, allow_nan=False)
        def calculate_ratio(a, b):
            return a / b  # 自动处理除零
    """
    def decorator(func: Callable[..., Union[float, int]]) -> Callable[..., Union[float, int]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Union[float, int]:
            try:
                result = func(*args, **kwargs)
                
                # 检查NaN
                if not allow_nan:
                    import numpy as np
                    if np.isnan(result):
                        logger.debug(f"{func.__name__}结果为NaN，返回默认值{default_return}")
                        return default_return
                
                # 检查Inf
                if not allow_inf:
                    import numpy as np
                    if np.isinf(result):
                        logger.debug(f"{func.__name__}结果为Inf，返回默认值{default_return}")
                        return default_return
                
                return result
                
            except Exception as e:
                logger.debug(f"{func.__name__}计算失败: {e}，返回默认值{default_return}")
                return default_return
        
        return wrapper
    return decorator


class ErrorContext:
    """
    异常上下文管理器
    
    用于替代简单的try-except块，提供统一的错误处理
    
    示例:
        with ErrorContext("计算VaR", default_value=0.0) as ctx:
            result = complex_calculation()
            ctx.result = result
        
        final_result = ctx.get_result()
    """
    def __init__(
        self,
        operation_name: str,
        default_value: Any = None,
        log_level: str = 'error',
        reraise: bool = False
    ):
        self.operation_name = operation_name
        self.default_value = default_value
        self.log_level = log_level
        self.reraise = reraise
        self.result = None
        self.error = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error = exc_val
            log_func = getattr(logger, self.log_level, logger.error)
            log_func(f"{self.operation_name}失败: {exc_val}", exc_info=True)
            
            if self.reraise:
                return False  # 重新抛出
            
            return True  # 抑制异常
        return True
    
    def get_result(self) -> Any:
        """获取结果，如果有错误则返回默认值"""
        if self.error is not None:
            return self.default_value() if callable(self.default_value) else self.default_value
        return self.result


def validate_and_execute(
    validator: Callable,
    default_on_invalid: Any = None
) -> Callable:
    """
    验证-执行模式装饰器
    
    先验证输入，验证失败则返回默认值
    
    Args:
        validator: 验证函数，接受(*args, **kwargs)，返回bool
        default_on_invalid: 验证失败时的返回值
    
    示例:
        def validate_data(data, threshold):
            return data is not None and len(data) >= threshold
        
        @validate_and_execute(
            validator=lambda data, threshold: validate_data(data, threshold),
            default_on_invalid=0.0
        )
        def calculate_metric(data, threshold=10):
            return np.mean(data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 执行验证
            try:
                is_valid = validator(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{func.__name__}验证失败: {e}")
                return default_on_invalid() if callable(default_on_invalid) else default_on_invalid
            
            if not is_valid:
                logger.debug(f"{func.__name__}验证未通过，返回默认值")
                return default_on_invalid() if callable(default_on_invalid) else default_on_invalid
            
            # 验证通过，执行函数
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
