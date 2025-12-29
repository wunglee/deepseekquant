"""
数据格式化器

职责：
1. 格式化不同类型的数据
2. 数值格式化和精度控制
3. 时间格式化
4. 货币格式化
"""
import datetime
import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataFormatter:
    """数据格式化器。"""
    
    def __init__(
        self,
        decimal_places: int = 2,
        date_format: str = '%Y-%m-%d',
        datetime_format: str = '%Y-%m-%d %H:%M:%S'
    ):
        """
        初始化格式化器。
        
        Args:
            decimal_places: 小数位数
            date_format: 日期格式
            datetime_format: 日期时间格式
        """
        self.decimal_places = decimal_places
        self.date_format = date_format
        self.datetime_format = datetime_format
    
    def format_number(
        self,
        value: Any,
        decimal_places: Optional[int] = None
    ) -> str:
        """
        格式化数值。
        
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
            places = decimal_places if decimal_places is not None else self.decimal_places
            return f"{num:.{places}f}"
        except (ValueError, TypeError):
            logger.warning(f"无法格式化数值: {value}")
            return str(value)
    
    def format_percentage(
        self,
        value: Any,
        decimal_places: Optional[int] = None
    ) -> str:
        """
        格式化百分比。
        
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
            places = decimal_places if decimal_places is not None else self.decimal_places
            return f"{num:.{places}f}%"
        except (ValueError, TypeError):
            logger.warning(f"无法格式化百分比: {value}")
            return str(value)
    
    def format_currency(
        self,
        value: Any,
        currency_symbol: str = '$',
        decimal_places: Optional[int] = None
    ) -> str:
        """
        格式化货币。
        
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
            places = decimal_places if decimal_places is not None else self.decimal_places
            
            # 添加千位分隔符
            formatted = f"{num:,.{places}f}"
            return f"{currency_symbol}{formatted}"
        except (ValueError, TypeError):
            logger.warning(f"无法格式化货币: {value}")
            return str(value)
    
    def format_date(
        self,
        value: Any,
        format_str: Optional[str] = None
    ) -> str:
        """
        格式化日期。
        
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
                # 🔧 尝试解析字符串：使用 pd.to_datetime
                value = pd.to_datetime(value.replace('Z', '+00:00'))
            
            if isinstance(value, (datetime, pd.Timestamp)):
                fmt = format_str if format_str else self.datetime_format
                return value.strftime(fmt)
            elif isinstance(value, datetime.date):
                fmt = format_str if format_str else self.date_format
                return value.strftime(fmt)
            else:
                return str(value)
        except Exception as e:
            logger.warning(f"无法格式化日期: {value}, {e}")
            return str(value)
    
    def format_large_number(
        self,
        value: Any,
        decimal_places: int = 1
    ) -> str:
        """
        格式化大数值（K/M/B/T）。
        
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
    
    def format_duration(
        self,
        seconds: float
    ) -> str:
        """
        格式化时间长度。
        
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
    
    def format_boolean(
        self,
        value: Any,
        true_text: str = 'Yes',
        false_text: str = 'No'
    ) -> str:
        """
        格式化布尔值。
        
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
    
    def truncate_string(
        self,
        value: str,
        max_length: int,
        suffix: str = '...'
    ) -> str:
        """
        截断字符串。
        
        Args:
            value: 字符串
            max_length: 最大长度
            suffix: 后缀
        
        Returns:
            截断后的字符串
        """
        if value is None:
            return 'N/A'
        
        str_value = str(value)
        if len(str_value) <= max_length:
            return str_value
        
        return str_value[:max_length - len(suffix)] + suffix
    
    def format_list(
        self,
        values: list,
        separator: str = ', ',
        max_items: Optional[int] = None
    ) -> str:
        """
        格式化列表。
        
        Args:
            values: 列表
            separator: 分隔符
            max_items: 最大项数
        
        Returns:
            格式化后的字符串
        """
        if not values:
            return 'N/A'
        
        items = values[:max_items] if max_items else values
        result = separator.join(str(v) for v in items)
        
        if max_items and len(values) > max_items:
            result += f" (+{len(values) - max_items} more)"
        
        return result
