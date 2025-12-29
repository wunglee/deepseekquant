"""
告警管理器测试（专家answer.md第3轮5.4节）

测试覆盖：
1. 多通道告警发送
2. 告警升级路径（30分钟/15分钟）
3. 去重和频率控制
4. 告警历史和统计
"""

import pandas as pd

from core_bak_refactored.core.monitoring import (
    AlertManager,
    AlertConfig,
    AlertChannel,
    AlertSeverity
)


class TestAlertManager:
    """告警管理器测试"""
    
    def test_alert_manager_initialization(self):
        """测试告警管理器初始化"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test',
            sms_provider='aliyun',
            phone_provider='tencent'
        )
        
        manager = AlertManager(config)
        
        assert manager.config.wechat_webhook_url == 'https://qyapi.weixin.qq.com/test'
        assert manager.config.sms_provider == 'aliyun'
        assert len(manager._alert_history) == 0
    
    def test_send_level1_alert(self):
        """测试Level 1告警：仅企业微信"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test'
        )
        manager = AlertManager(config)
        
        # 发送WARNING级别告警
        record = manager.send_alert(
            severity=AlertSeverity.WARNING,
            title='Level 1告警测试',
            message='预测误差17%，触发Level 1',
            metadata={'error': 0.17}
        )
        
        assert record is not None
        assert record.severity == AlertSeverity.WARNING
        assert AlertChannel.WECHAT in record.channels_used
        assert AlertChannel.SMS not in record.channels_used
        assert AlertChannel.PHONE not in record.channels_used
    
    def test_send_level2_alert(self):
        """测试Level 2告警：企业微信+短信"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test',
            sms_provider='aliyun'
        )
        manager = AlertManager(config)
        
        # 发送ERROR级别告警
        record = manager.send_alert(
            severity=AlertSeverity.ERROR,
            title='Level 2告警测试',
            message='预测误差22%，触发Level 2',
            metadata={'error': 0.22}
        )
        
        assert record is not None
        assert record.severity == AlertSeverity.ERROR
        assert AlertChannel.WECHAT in record.channels_used
        assert AlertChannel.SMS in record.channels_used
        assert AlertChannel.PHONE not in record.channels_used
        
        # Level 2应调度30分钟后升级
        assert len(manager._pending_escalations) == 1
        assert manager._pending_escalations[0].alert_id == record.alert_id
    
    def test_send_level3_alert(self):
        """测试Level 3告警：企业微信+短信+电话"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test',
            sms_provider='aliyun',
            phone_provider='tencent'
        )
        manager = AlertManager(config)
        
        # 发送CRITICAL级别告警
        record = manager.send_alert(
            severity=AlertSeverity.CRITICAL,
            title='Level 3告警测试',
            message='预测误差30%，触发Level 3',
            metadata={'error': 0.30}
        )
        
        assert record is not None
        assert record.severity == AlertSeverity.CRITICAL
        assert AlertChannel.WECHAT in record.channels_used
        assert AlertChannel.SMS in record.channels_used
        assert AlertChannel.PHONE in record.channels_used
        
        # Level 3应调度15分钟后重复
        assert len(manager._pending_escalations) == 1
    
    def test_alert_deduplication(self):
        """测试告警去重"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test',
            dedup_window_minutes=10
        )
        manager = AlertManager(config)
        
        # 发送第一次告警
        record1 = manager.send_alert(
            severity=AlertSeverity.WARNING,
            title='重复告警测试',
            message='测试消息',
            dedup_key='test_dedup_key'
        )
        assert record1 is not None
        
        # 10分钟内发送相同dedup_key的告警，应被去重
        record2 = manager.send_alert(
            severity=AlertSeverity.WARNING,
            title='重复告警测试',
            message='测试消息',
            dedup_key='test_dedup_key'
        )
        assert record2 is None  # 被去重
        
        # 不同dedup_key，应正常发送
        record3 = manager.send_alert(
            severity=AlertSeverity.WARNING,
            title='重复告警测试',
            message='测试消息',
            dedup_key='different_key'
        )
        assert record3 is not None
    
    def test_rate_limiting(self):
        """测试频率限制"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test',
            max_alerts_per_hour=5  # 设置较小值便于测试
        )
        manager = AlertManager(config)
        
        # 发送5次告警（达到限制）
        for i in range(5):
            record = manager.send_alert(
                severity=AlertSeverity.INFO,
                title=f'频率测试 {i}',
                message='测试消息'
            )
            assert record is not None
        
        # 第6次应被限制
        record = manager.send_alert(
            severity=AlertSeverity.INFO,
            title='频率测试 超限',
            message='测试消息'
        )
        assert record is None  # 被限流
    
    def test_escalation_processing(self):
        """测试告警升级处理"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test',
            sms_provider='aliyun',
            phone_provider='tencent',
            escalation_interval_minutes=0  # 立即升级以便测试
        )
        manager = AlertManager(config)
        
        # 发送Level 2告警
        record = manager.send_alert(
            severity=AlertSeverity.ERROR,
            title='升级测试',
            message='测试30分钟升级'
        )
        
        assert len(manager._pending_escalations) == 1
        
        # 处理升级
        escalated = manager.process_escalations()
        
        assert len(escalated) == 1
        assert escalated[0].alert_id == record.alert_id
        assert len(manager._pending_escalations) == 0  # Level 2不重复
    
    def test_critical_alert_repeat(self):
        """测试Level 3告警重复"""
        config = AlertConfig(
            wechat_webhook_url='https://qyapi.weixin.qq.com/test',
            phone_provider='tencent',
            critical_repeat_minutes=0  # 立即重复以便测试
        )
        manager = AlertManager(config)
        
        # 发送Level 3告警
        record = manager.send_alert(
            severity=AlertSeverity.CRITICAL,
            title='重复测试',
            message='测试15分钟重复'
        )
        
        assert len(manager._pending_escalations) == 1
        
        # 处理第一次重复
        escalated1 = manager.process_escalations()
        assert len(escalated1) == 1
        
        # Level 3应继续调度下一次重复
        assert len(manager._pending_escalations) == 1
        
        # 处理第二次重复
        escalated2 = manager.process_escalations()
        assert len(escalated2) == 1
    
    def test_get_alert_history(self):
        """测试获取告警历史"""
        config = AlertConfig(wechat_webhook_url='https://test')
        manager = AlertManager(config)
        
        # 发送不同级别的告警
        manager.send_alert(AlertSeverity.WARNING, '告警1', '消息1')
        manager.send_alert(AlertSeverity.ERROR, '告警2', '消息2')
        manager.send_alert(AlertSeverity.CRITICAL, '告警3', '消息3')
        
        # 获取全部历史
        all_alerts = manager.get_alert_history()
        assert len(all_alerts) == 3
        
        # 按级别筛选
        error_alerts = manager.get_alert_history(severity=AlertSeverity.ERROR)
        assert len(error_alerts) == 1
        assert error_alerts[0].severity == AlertSeverity.ERROR
        
        # 按时间筛选
        since = pd.Timestamp.now() - pd.Timedelta(minutes=1)
        recent_alerts = manager.get_alert_history(since=since)
        assert len(recent_alerts) == 3
    
    def test_get_statistics(self):
        """测试获取告警统计"""
        config = AlertConfig(
            wechat_webhook_url='https://test',
            sms_provider='aliyun'
        )
        manager = AlertManager(config)
        
        # 发送不同级别的告警
        manager.send_alert(AlertSeverity.WARNING, '告警1', '消息1')
        manager.send_alert(AlertSeverity.ERROR, '告警2', '消息2')
        manager.send_alert(AlertSeverity.ERROR, '告警3', '消息3')
        
        # 获取统计
        stats = manager.get_statistics(hours=24)
        
        assert stats['total_alerts'] == 3
        assert stats['by_severity'][AlertSeverity.WARNING.value] == 1
        assert stats['by_severity'][AlertSeverity.ERROR.value] == 2
        assert stats['by_channel'][AlertChannel.WECHAT.value] == 3
        assert stats['by_channel'][AlertChannel.SMS.value] == 2  # ERROR级别
