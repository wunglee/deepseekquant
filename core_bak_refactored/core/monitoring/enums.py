"""
监控模块枚举定义

职责：
- 定义监控和告警相关的枚举类型
- 告警严重程度、告警通道等
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
        return self.value


class AlertChannel(str, Enum):
    """告警通道枚举"""
    WECHAT = 'wechat'        # 企业微信
    SMS = 'sms'              # 短信
    PHONE = 'phone'          # 电话
    EMAIL = 'email'          # 邮件（补充通道）
    DINGTALK = 'dingtalk'    # 钉钉
    SLACK = 'slack'          # Slack
    WEBHOOK = 'webhook'      # Webhook（补充通道）
    LOG = 'log'              # 日志
    
    def __str__(self) -> str:
        return self.value


__all__ = [
    'AlertSeverity',
    'AlertChannel',
]
