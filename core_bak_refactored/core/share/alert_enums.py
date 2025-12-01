"""
告警枚举定义（共享模块）

职责：定义标准化的告警枚举类型
用途：替换项目中所有字符串常量，提供类型安全和自动补全
"""

from enum import Enum


class AlertSeverity(str, Enum):
    """
    告警严重程度枚举
    
    继承自str使其可直接用于字符串比较和字典键
    """
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


class AlertChannel(str, Enum):
    """
    告警通道枚举
    """
    EMAIL = 'email'
    SMS = 'sms'
    WECHAT = 'wechat'
    DINGTALK = 'dingtalk'
    SLACK = 'slack'
    WEBHOOK = 'webhook'
    LOG = 'log'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value


class DataQualityIssueType(str, Enum):
    """
    数据质量问题类型枚举
    """
    MISSING_DATA = 'missing_data'
    DUPLICATE_DATA = 'duplicate_data'
    OUTLIER = 'outlier'
    INCONSISTENCY = 'inconsistency'
    TIMELINESS = 'timeliness'
    COMPLETENESS = 'completeness'
    ACCURACY = 'accuracy'
    
    def __str__(self) -> str:
        """支持直接字符串转换"""
        return self.value