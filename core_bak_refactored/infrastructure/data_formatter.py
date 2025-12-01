"""
数据格式化工具 - 基础设施层

职责：提供与业务无关的纯数据格式化函数
- 数值格式化
- 百分比格式化
- 货币格式化
- 日期格式化
- 大数值格式化

架构原则：
- 不包含任何业务领域概念
- 只接收纯数值数据
- 参数全部显式传入，不使用业务默认值
- 函数命名使用通用术语，而非业务术语
"""

from typing import Any, Optional
from datetime import datetime, date
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.DataFormatter')


class DataFormatter:
    """数据格式化工具类（纯工具），不包含业务术语"""
    
    @staticmethod
    def format_number(
        value: Any,
        decimal_places: int = 2
    ) -> str:
        """
        格式化数值
        
        Args:
            value: 数值
            decimal_places: 小数位数
            
        Returns:
            格式化后的字符串
        """
        if value is None:
            return 'N/A'
        
        try:
            num = float(value)
            return f"{num:.{decimal_places}f}"
        except (ValueError, TypeError):
            logger.warning(f"无法格式化数值: {value}")
            return str(value)
    
    @staticmethod
    def format_percentage(
        value: Any,
        decimal_places: int = 2
    ) -> str:
        """
        格式化百分比
        
        Args:
            value: 数值（0.1 = 10%）
            decimal_places: 小数位数
            
        Returns:
            格式化后的字符串
        """
        if value is None:
            return 'N/A'
        
        try:
            num = float(value) * 100
            return f"{num:.{decimal_places}f}%"
        except (ValueError, TypeError):
            logger.warning(f"无法格式化百分比: {value}")
            return str(value)
    
    @staticmethod
    def format_currency(
        value: Any,
        currency_symbol: str = '$',
        decimal_places: int = 2
    ) -> str:
        """
        格式化货币
        
        Args:
            value: 金额
            currency_symbol: 货币符号
            decimal_places: 小数位数
            
        Returns:
            格式化后的字符串
        """
        if value is None:
            return 'N/A'
        
        try:
            num = float(value)
            
            # 添加千位分隔符
            formatted = f"{num:,.{decimal_places}f}"
            return f"{currency_symbol}{formatted}"
        except (ValueError, TypeError):
            logger.warning(f"无法格式化货币: {value}")
            return str(value)
    
    @staticmethod
    def format_date(
        value: Any,
        format_str: str = '%Y-%m-%d %H:%M:%S'
    ) -> str:
        """
        格式化日期
        
        Args:
            value: 日期对象
            format_str: 格式字符串
            
        Returns:
            格式化后的字符串
        """
        if value is None:
            return 'N/A'
        
        try:
            if isinstance(value, str):
                # 尝试解析字符串
                value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            
            if isinstance(value, datetime):
                return value.strftime(format_str)
            elif isinstance(value, date):
                return value.strftime(format_str)
            else:
                return str(value)
        except Exception as e:
            logger.warning(f"无法格式化日期: {value}, {e}")
            return str(value)
    
    @staticmethod
    def format_large_number(
        value: Any,
        decimal_places: int = 1
    ) -> str:
        """
        格式化大数值（K/M/B/T）
        
        Args:
            value: 数值
            decimal_places: 小数位数
            
        Returns:
            格式化后的字符串
        """
        if value is None:
            return 'N/A'
        
        try:
            num = float(value)
            abs_num = abs(num)
            
            if abs_num >= 1_000_000_000_000:  # Trillion
                formatted = num / 1_000_000_000_000
                suffix = 'T'
            elif abs_num >= 1_000_000_000:  # Billion
                formatted = num / 1_000_000_000
                suffix = 'B'
            elif abs_num >= 1_000_000:  # Million
                formatted = num / 1_000_000
                suffix = 'M'
            elif abs_num >= 1_000:  # Thousand
                formatted = num / 1_000
                suffix = 'K'
            else:
                formatted = num
                suffix = ''
            
            return f"{formatted:.{decimal_places}f}{suffix}"
        except (ValueError, TypeError):
            logger.warning(f"无法格式化大数值: {value}")
            return str(value)
    
    @staticmethod
    def format_duration(
        seconds: float
    ) -> str:
        """
        格式化时间长度
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化后的字符串
        """
        if seconds is None:
            return 'N/A'
        
        try:
            sec = int(seconds)
            
            days = sec // 86400
            hours = (sec % 86400) // 3600
            minutes = (sec % 3600) // 60
            secs = sec % 60
            
            parts = []
            if days > 0:
                parts.append(f"{days}d")
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if secs > 0 or not parts:
                parts.append(f"{secs}s")
            
            return ' '.join(parts)
        except (ValueError, TypeError):
            logger.warning(f"无法格式化时间长度: {seconds}")
            return str(seconds)
    
    @staticmethod
    def format_boolean(
        value: Any,
        true_text: str = 'Yes',
        false_text: str = 'No'
    ) -> str:
        """
        格式化布尔值
        
        Args:
            value: 布尔值
            true_text: True的文本
            false_text: False的文本
            
        Returns:
            格式化后的字符串
        """
        if value is None:
            return 'N/A'
        
        return true_text if bool(value) else false_text