"""
风险监控器 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 实时风险监控、告警
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import time
import threading
from collections import deque

from .risk_models import RiskLevel, RiskType, RiskEvent, RiskAssessment

logger = logging.getLogger('DeepSeekQuant.RiskMonitor')


class RiskMonitor:
    """独立风险监控器 - 用于实时风险监控"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_thresholds = config.get('risk_thresholds', {})
        self.alert_handlers = []
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # 风险事件历史
        self.risk_events: deque = deque(maxlen=config.get('max_event_history', 100))
        
        # 监控状态
        self.last_check_time = None
        self.check_count = 0
        self.alert_count = 0
        
        # 风险评估缓存
        self.latest_assessment: Optional[RiskAssessment] = None
        
        logger.info("风险监控器初始化完成")

    def add_alert_handler(self, handler):
        """添加警报处理器"""
        self.alert_handlers.append(handler)

    def start_monitoring(self, interval: int = 60):
        """
        开始风险监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self.is_monitoring:
            logger.warning("监控已经在运行")
            return
        
        self.is_monitoring = True
        self.config['monitoring_interval'] = interval
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info(f"风险监控已启动，间隔={interval}秒")

    def stop_monitoring(self):
        """停止风险监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("风险监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        logger.info("监控循环开始")
        while self.is_monitoring:
            try:
                # 执行风险检查
                risk_status = self._check_risk_status()
                self.check_count += 1
                self.last_check_time = datetime.now()

                # 触发警报
                if risk_status['alert_level'] > 0:
                    self._trigger_alerts(risk_status)
                    self.alert_count += 1

                # 等待下一次检查
                time.sleep(self.config.get('monitoring_interval', 60))

            except Exception as e:
                logger.error(f"风险监控循环错误: {e}")
                time.sleep(10)  # 错误后等待10秒
        
        logger.info("监控循环结束")
    
    def check_real_time_risk(self, portfolio_state, risk_assessment: RiskAssessment) -> Dict[str, Any]:
        """
        实时检查风险
        
        Args:
            portfolio_state: 组合状态
            risk_assessment: 风险评估结果
            
        Returns:
            检查结果
        """
        try:
            self.latest_assessment = risk_assessment
            
            # 检查风险等级
            alert_level = self._determine_alert_level(risk_assessment)
            
            # 检查限额违规
            has_breaches = len(risk_assessment.limit_breaches) > 0
            
            # 生成风险事件
            if alert_level > 1 or has_breaches:
                event = self._create_risk_event(portfolio_state, risk_assessment, alert_level)
                self.risk_events.append(event)
            
            result = {
                'alert_level': alert_level,
                'timestamp': datetime.now(),
                'risk_score': risk_assessment.risk_score,
                'risk_level': risk_assessment.overall_risk_level.value,
                'limit_breaches': risk_assessment.limit_breaches,
                'requires_action': alert_level >= 2
            }
            
            logger.debug(f"实时风险检查: 警报等级={alert_level}, 风险评分={risk_assessment.risk_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"实时风险检查失败: {e}")
            return {'alert_level': 0, 'error': str(e)}
    
    def _trigger_alerts(self, risk_status: Dict[str, Any]):
        """触发警报"""
        logger.warning(f"风险警报触发: 等级={risk_status['alert_level']}, 时间={risk_status['timestamp']}")
        
        for handler in self.alert_handlers:
            try:
                handler(risk_status)
            except Exception as e:
                logger.error(f"警报处理器错误: {e}")
    
    def trigger_alert(self, risk_event: RiskEvent):
        """
        手动触发警报
        
        Args:
            risk_event: 风险事件
        """
        risk_status = {
            'alert_level': self._map_severity_to_level(risk_event.severity),
            'timestamp': risk_event.timestamp,
            'event': risk_event.to_dict()
        }
        self._trigger_alerts(risk_status)
    
    def _map_severity_to_level(self, severity: RiskLevel) -> int:
        """将严重程度映射为警报等级"""
        mapping = {
            RiskLevel.VERY_LOW: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.VERY_HIGH: 4,
            RiskLevel.EXTREME: 4
        }
        return mapping.get(severity, 2)
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态
        
        Returns:
            监控状态信息
        """
        return {
            'is_monitoring': self.is_monitoring,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'check_count': self.check_count,
            'alert_count': self.alert_count,
            'event_history_count': len(self.risk_events),
            'latest_risk_score': self.latest_assessment.risk_score if self.latest_assessment else None
        }
    
    def _check_risk_status(self) -> Dict[str, Any]:
        """
        检查风险状态
        
        注：该方法在监控循环中调用，需要外部提供风险评估数据
        或者使用 check_real_time_risk 方法直接检查
        """
        if self.latest_assessment is None:
            return {
                'alert_level': 0,
                'timestamp': datetime.now(),
                'metrics': {}
            }
        
        return self.check_real_time_risk(None, self.latest_assessment)
    
    def _determine_alert_level(self, risk_assessment: RiskAssessment) -> int:
        """
        确定警报等级
        
        Returns:
            0: 无警报
            1: 低级警报
            2: 中级警报
            3: 高级警报
            4: 严重警报
        """
        risk_score = risk_assessment.risk_score
        breach_count = len(risk_assessment.limit_breaches)
        
        # 基于风险评分
        if risk_score >= 90:
            return 4  # 严重
        elif risk_score >= 75:
            return 3  # 高级
        elif risk_score >= 60:
            return 2  # 中级
        elif risk_score >= 40:
            return 1  # 低级
        
        # 基于限额违规
        if breach_count >= 3:
            return max(3, int(risk_score / 25))
        elif breach_count >= 1:
            return max(2, int(risk_score / 25))
        
        return 0
    
    def _create_risk_event(self, portfolio_state, risk_assessment: RiskAssessment, alert_level: int) -> RiskEvent:
        """创建风险事件"""
        from .risk_models import RiskControlAction
        
        severity_map = {
            1: RiskLevel.LOW,
            2: RiskLevel.MODERATE,
            3: RiskLevel.HIGH,
            4: RiskLevel.VERY_HIGH
        }
        
        action_map = {
            1: RiskControlAction.WARN,
            2: RiskControlAction.WARN,
            3: RiskControlAction.REDUCE,
            4: RiskControlAction.REJECT
        }
        
        return RiskEvent(
            event_id=f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.alert_count}",
            event_type=RiskType.MARKET_RISK,
            severity=severity_map.get(alert_level, RiskLevel.MODERATE),
            timestamp=datetime.now().isoformat(),
            description=f"风险评分{risk_assessment.risk_score:.2f}, {len(risk_assessment.limit_breaches)}项限额违规",
            triggered_by="risk_monitor",
            impact_assessment={'risk_score': risk_assessment.risk_score, 'breaches': len(risk_assessment.limit_breaches)},
            action_taken=action_map.get(alert_level, RiskControlAction.WARN)
        )
