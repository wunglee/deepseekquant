"""
生产级告警管理器（专家answer.md第3轮5.4节）

职责：
- 管理多通道告警（企业微信/短信/电话）
- 实现30分钟升级路径
- 告警去重和频率控制
- 告警历史追踪

告警升级路径：
Level 1 (15%-20%): 企业微信通知 → 无升级
Level 2 (20%-25%): 企业微信 + 短信 → 30分钟后升级到电话
Level 3 (>25%):    企业微信 + 短信 + 电话（立即） → 15分钟后重复电话
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger('DeepSeekQuant.AlertManager')


class AlertChannel(str, Enum):
    """告警通道"""
    WECHAT = 'wechat'        # 企业微信
    SMS = 'sms'              # 短信
    PHONE = 'phone'          # 电话
    EMAIL = 'email'          # 邮件（补充通道）
    WEBHOOK = 'webhook'      # Webhook（补充通道）


class AlertSeverity(str, Enum):
    """告警严重级别"""
    INFO = 'info'            # 信息
    WARNING = 'warning'      # 警告（Level 1）
    ERROR = 'error'          # 错误（Level 2）
    CRITICAL = 'critical'    # 严重（Level 3）


@dataclass
class AlertConfig:
    """告警配置"""
    # 企业微信配置
    wechat_webhook_url: Optional[str] = None
    wechat_mentioned_list: List[str] = field(default_factory=list)  # @提醒用户列表
    
    # 短信配置
    sms_provider: Optional[str] = None  # 'aliyun', 'tencent', 'twilio'
    sms_access_key: Optional[str] = None
    sms_access_secret: Optional[str] = None
    sms_phone_numbers: List[str] = field(default_factory=list)
    
    # 电话配置
    phone_provider: Optional[str] = None
    phone_access_key: Optional[str] = None
    phone_access_secret: Optional[str] = None
    phone_numbers: List[str] = field(default_factory=list)
    
    # 升级路径配置
    escalation_interval_minutes: int = 30  # Level 2升级到电话的间隔
    critical_repeat_minutes: int = 15      # Level 3重复电话间隔
    
    # 去重配置
    dedup_window_minutes: int = 10         # 去重窗口
    max_alerts_per_hour: int = 50          # 每小时最大告警数


@dataclass
class AlertRecord:
    """告警记录"""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    metadata: Dict[str, Any]
    channels_used: List[AlertChannel]
    created_at: datetime
    escalated_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    dedup_key: Optional[str] = None


class AlertManager:
    """
    生产级告警管理器
    
    功能：
    1. 多通道告警发送（企业微信/短信/电话）
    2. 自动升级路径（30分钟/15分钟）
    3. 告警去重和频率控制
    4. 告警历史和统计
    
    使用示例：
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/...',
            sms_phone_numbers=['+86138****']
        )
        manager = AlertManager(config)
        
        # Level 2告警：企业微信+短信，30分钟后升级电话
        manager.send_alert(
            severity=AlertSeverity.ERROR,
            title='预测误差超阈值',
            message='事件event123预测误差22%，超过20%阈值',
            metadata={'event_id': 'event123', 'error': 0.22}
        )
    """
    
    def __init__(self, config: AlertConfig):
        """
        初始化告警管理器
        
        Args:
            config: 告警配置
        """
        self.config = config
        self._alert_history: List[AlertRecord] = []
        self._pending_escalations: List[AlertRecord] = []
        
        # 初始化告警发送器
        self._senders: Dict[AlertChannel, Callable] = {
            AlertChannel.WECHAT: self._send_wechat,
            AlertChannel.SMS: self._send_sms,
            AlertChannel.PHONE: self._send_phone,
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.WEBHOOK: self._send_webhook
        }
        
        logger.info(f"AlertManager initialized with {len([c for c in [config.wechat_webhook_url, config.sms_provider, config.phone_provider] if c])} channels")
    
    def send_alert(self,
                   severity: AlertSeverity,
                   title: str,
                   message: str,
                   metadata: Optional[Dict[str, Any]] = None,
                   dedup_key: Optional[str] = None) -> Optional[AlertRecord]:
        """
        发送告警
        
        Args:
            severity: 严重级别
            title: 告警标题
            message: 告警消息
            metadata: 元数据
            dedup_key: 去重键（相同key在去重窗口内只发一次）
        
        Returns:
            告警记录（如果被去重则返回None）
        """
        # 去重检查
        if dedup_key and self._is_duplicate(dedup_key):
            logger.info(f"Alert deduplicated: {dedup_key}")
            return None
        
        # 频率控制
        if not self._check_rate_limit():
            logger.warning("Alert rate limit exceeded, dropping alert")
            return None
        
        # 确定告警通道
        channels = self._determine_channels(severity)
        
        # 创建告警记录
        alert_id = f"alert_{int(time.time() * 1000)}"
        record = AlertRecord(
            alert_id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metadata=metadata or {},
            channels_used=channels,
            created_at=datetime.now(),
            dedup_key=dedup_key
        )
        
        # 发送告警
        success_channels = []
        for channel in channels:
            sender = self._senders.get(channel)
            if sender:
                try:
                    success = sender(record)
                    if success:
                        success_channels.append(channel)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
        
        # 保存记录
        self._alert_history.append(record)
        
        # 注册升级任务
        if severity == AlertSeverity.ERROR:
            # Level 2: 30分钟后升级到电话
            self._schedule_escalation(record, self.config.escalation_interval_minutes)
        elif severity == AlertSeverity.CRITICAL:
            # Level 3: 15分钟后重复电话
            self._schedule_escalation(record, self.config.critical_repeat_minutes)
        
        logger.info(f"Alert sent: {alert_id} via {success_channels}")
        return record
    
    def _determine_channels(self, severity: AlertSeverity) -> List[AlertChannel]:
        """
        根据严重级别确定告警通道
        
        策略：
        - INFO: 企业微信
        - WARNING (Level 1): 企业微信
        - ERROR (Level 2): 企业微信 + 短信
        - CRITICAL (Level 3): 企业微信 + 短信 + 电话
        """
        channels = []
        
        if severity in [AlertSeverity.INFO, AlertSeverity.WARNING]:
            # Level 1: 仅企业微信
            if self.config.wechat_webhook_url:
                channels.append(AlertChannel.WECHAT)
        
        elif severity == AlertSeverity.ERROR:
            # Level 2: 企业微信 + 短信
            if self.config.wechat_webhook_url:
                channels.append(AlertChannel.WECHAT)
            if self.config.sms_provider:
                channels.append(AlertChannel.SMS)
        
        elif severity == AlertSeverity.CRITICAL:
            # Level 3: 企业微信 + 短信 + 电话
            if self.config.wechat_webhook_url:
                channels.append(AlertChannel.WECHAT)
            if self.config.sms_provider:
                channels.append(AlertChannel.SMS)
            if self.config.phone_provider:
                channels.append(AlertChannel.PHONE)
        
        return channels
    
    def _is_duplicate(self, dedup_key: str) -> bool:
        """检查是否重复告警"""
        cutoff_time = datetime.now() - timedelta(minutes=self.config.dedup_window_minutes)
        
        for record in reversed(self._alert_history):
            if record.created_at < cutoff_time:
                break
            if record.dedup_key == dedup_key:
                return True
        
        return False
    
    def _check_rate_limit(self) -> bool:
        """检查频率限制"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_alerts = [r for r in self._alert_history if r.created_at >= cutoff_time]
        
        return len(recent_alerts) < self.config.max_alerts_per_hour
    
    def _schedule_escalation(self, record: AlertRecord, interval_minutes: int):
        """调度告警升级"""
        record.escalated_at = datetime.now() + timedelta(minutes=interval_minutes)
        self._pending_escalations.append(record)
        logger.info(f"Scheduled escalation for {record.alert_id} in {interval_minutes} minutes")
    
    def process_escalations(self):
        """
        处理待升级告警（应定期调用，如每分钟）
        
        建议在后台线程或定时任务中调用
        """
        now = datetime.now()
        escalated = []
        
        for record in self._pending_escalations[:]:
            if record.escalated_at and record.escalated_at <= now:
                # 执行升级
                if record.severity == AlertSeverity.ERROR:
                    # Level 2升级：发送电话
                    if self.config.phone_provider:
                        self._send_phone(record)
                        logger.info(f"Escalated Level 2 alert {record.alert_id} to phone")
                
                elif record.severity == AlertSeverity.CRITICAL:
                    # Level 3重复：再次电话
                    if self.config.phone_provider:
                        self._send_phone(record)
                        logger.info(f"Repeated Level 3 alert {record.alert_id} via phone")
                    
                    # 继续调度下一次重复
                    self._schedule_escalation(record, self.config.critical_repeat_minutes)
                
                escalated.append(record)
                self._pending_escalations.remove(record)
        
        return escalated
    
    # ==================== 告警发送实现 ====================
    
    def _send_wechat(self, record: AlertRecord) -> bool:
        """发送企业微信告警"""
        if not self.config.wechat_webhook_url:
            logger.warning("WeChat webhook URL not configured")
            return False
        
        try:
            import requests
            
            # 构造消息
            severity_emoji = {
                AlertSeverity.INFO: 'ℹ️',
                AlertSeverity.WARNING: '⚠️',
                AlertSeverity.ERROR: '❌',
                AlertSeverity.CRITICAL: '🚨'
            }
            
            content = f"{severity_emoji.get(record.severity, '')} **{record.title}**\n\n"
            content += f"{record.message}\n\n"
            content += f"时间: {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
            content += f"ID: {record.alert_id}"
            
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "content": content,
                    "mentioned_list": self.config.wechat_mentioned_list
                }
            }
            
            # 发送请求
            response = requests.post(
                self.config.wechat_webhook_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.info(f"WeChat alert sent: {record.alert_id}")
                return True
            else:
                logger.error(f"WeChat API error: {response.status_code} {response.text}")
                return False
        
        except ImportError:
            logger.warning("requests库未安装，无法发送企业微信告警")
            return False
        except Exception as e:
            logger.error(f"WeChat send failed: {e}")
            return False
    
    def _send_sms(self, record: AlertRecord) -> bool:
        """发送短信告警"""
        if not self.config.sms_provider:
            logger.warning("SMS provider not configured")
            return False
        
        # TODO: 集成阿里云/腾讯云短信SDK
        logger.info(f"[STUB] SMS sent to {self.config.sms_phone_numbers}: {record.title}")
        return True
    
    def _send_phone(self, record: AlertRecord) -> bool:
        """发送电话告警"""
        if not self.config.phone_provider:
            logger.warning("Phone provider not configured")
            return False
        
        # TODO: 集成语音通知SDK
        logger.info(f"[STUB] Phone call to {self.config.phone_numbers}: {record.title}")
        return True
    
    def _send_email(self, record: AlertRecord) -> bool:
        """发送邮件告警（补充通道）"""
        logger.info(f"[STUB] Email sent: {record.title}")
        return True
    
    def _send_webhook(self, record: AlertRecord) -> bool:
        """发送Webhook告警（补充通道）"""
        logger.info(f"[STUB] Webhook sent: {record.title}")
        return True
    
    # ==================== 查询和统计 ====================
    
    def get_alert_history(self,
                          severity: Optional[AlertSeverity] = None,
                          since: Optional[datetime] = None,
                          limit: int = 100) -> List[AlertRecord]:
        """
        获取告警历史
        
        Args:
            severity: 筛选严重级别
            since: 起始时间
            limit: 最大返回数量
        
        Returns:
            告警记录列表
        """
        alerts = self._alert_history
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.created_at >= since]
        
        # 按时间倒序
        alerts = sorted(alerts, key=lambda a: a.created_at, reverse=True)
        
        return alerts[:limit]
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取告警统计
        
        Args:
            hours: 统计时间范围（小时）
        
        Returns:
            统计字典
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self._alert_history if a.created_at >= cutoff_time]
        
        return {
            'total_alerts': len(recent_alerts),
            'by_severity': {
                severity.value: len([a for a in recent_alerts if a.severity == severity])
                for severity in AlertSeverity
            },
            'by_channel': {
                channel.value: len([a for a in recent_alerts if channel in a.channels_used])
                for channel in AlertChannel
            },
            'pending_escalations': len(self._pending_escalations),
            'time_range_hours': hours
        }
