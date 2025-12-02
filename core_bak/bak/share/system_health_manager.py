"""
系统健康管理器（共享模块）

职责：提供标准化的系统健康状态管理接口
用途：统一管理系统健康度计算、监控和报告
"""

from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger('DeepSeekQuant.Core.Share.SystemHealthManager')


@dataclass
class HealthMetrics:
    """健康度量数据类"""
    overall_score: float = 0.0
    uptime_seconds: float = 0.0
    success_rate: float = 1.0
    alert_count: int = 0
    data_points_processed: int = 0
    anomalies_detected: int = 0
    last_check_time: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'overall_score': self.overall_score,
            'uptime_seconds': self.uptime_seconds,
            'success_rate': self.success_rate,
            'alert_count': self.alert_count,
            'data_points_processed': self.data_points_processed,
            'anomalies_detected': self.anomalies_detected,
            'last_check_time': self.last_check_time.isoformat()
        }


class SystemHealthManager:
    """
    系统健康管理器
    
    职责：提供标准化的系统健康状态管理接口
    """
    
    def __init__(self):
        self._health_history: List[HealthMetrics] = []
        self._current_metrics = HealthMetrics()
        self._start_time = datetime.now()
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        更新健康度量
        
        Args:
            metrics: 健康度量字典
        """
        # 更新当前度量
        for key, value in metrics.items():
            if hasattr(self._current_metrics, key):
                setattr(self._current_metrics, key, value)
        
        # 更新时间戳
        self._current_metrics.last_check_time = datetime.now()
        
        # 添加到历史记录
        self._health_history.append(HealthMetrics(**self._current_metrics.__dict__))
        
        # 保持历史记录长度
        if len(self._health_history) > 1000:
            self._health_history = self._health_history[-1000:]
        
        logger.debug(f"健康度量已更新: {self._current_metrics.overall_score:.2f}")
    
    def calculate_overall_health(self, 
                               quality_score: float = 1.0,
                               success_rate: float = 1.0,
                               alert_count: int = 0) -> float:
        """
        计算总体健康度（0-100）
        
        Args:
            quality_score: 数据质量得分
            success_rate: 成功率
            alert_count: 告警数量
            
        Returns:
            总体健康度分数
        """
        # 告警惩罚因子
        alert_penalty = min(1.0, alert_count / 50.0)  # 每50个告警扣10%分数
        
        # 加权计算总体健康度
        health_score = (
            quality_score * 0.4 +      # 数据质量权重40%
            success_rate * 0.4 +       # 成功率权重40%
            (1.0 - alert_penalty) * 0.2  # 告警惩罚权重20%
        )
        
        return max(0.0, min(1.0, health_score)) * 100
    
    def determine_health_status(self, health_score: float) -> str:
        """
        确定健康状态
        
        Args:
            health_score: 健康度分数
            
        Returns:
            健康状态字符串
        """
        if health_score >= 90:
            return 'healthy'
        elif health_score >= 70:
            return 'degraded'
        elif health_score >= 50:
            return 'warning'
        else:
            return 'critical'
    
    def generate_recommendations(self, 
                              stats: Dict[str, Any], 
                              health_score: float) -> List[Dict[str, str]]:
        """
        生成健康度建议
        
        Args:
            stats: 统计信息
            health_score: 健康度分数
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 成功率建议
        success_rate = stats.get('success_rate', 1.0)
        if success_rate < 0.95:
            recommendations.append({
                'type': 'performance',
                'priority': 'high' if success_rate < 0.9 else 'medium',
                'message': f'成功率较低 ({success_rate:.1%})，建议检查系统性能和错误处理'
            })
        
        # 告警建议
        alert_count = stats.get('alerts_triggered', 0)
        if alert_count > 10:
            recommendations.append({
                'type': 'alerting',
                'priority': 'high' if alert_count > 50 else 'medium',
                'message': f'告警数量较多 ({alert_count})，建议优化告警阈值和频率控制'
            })
        
        # 健康度建议
        if health_score < 80:
            recommendations.append({
                'type': 'general',
                'priority': 'high' if health_score < 60 else 'medium',
                'message': f'系统健康度较低 ({health_score:.1f})，建议全面检查系统状态'
            })
        
        return recommendations
    
    def get_current_health(self) -> HealthMetrics:
        """
        获取当前健康状态
        
        Returns:
            当前健康度量
        """
        return self._current_metrics
    
    def get_health_history(self, limit: int = 50) -> List[HealthMetrics]:
        """
        获取健康历史记录
        
        Args:
            limit: 返回记录数量限制
            
        Returns:
            健康历史记录列表
        """
        return self._health_history[-limit:] if self._health_history else []
    
    def get_uptime(self) -> float:
        """
        获取运行时间（秒）
        
        Returns:
            运行时间秒数
        """
        return (datetime.now() - self._start_time).total_seconds()
    
    def format_uptime(self, seconds: float) -> str:
        """
        格式化运行时间为人类可读格式
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}天 {hours}小时"
        elif hours > 0:
            return f"{hours}小时 {minutes}分钟"
        else:
            return f"{minutes}分钟"
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        获取健康摘要
        
        Returns:
            健康摘要字典
        """
        current = self.get_current_health()
        uptime_seconds = self.get_uptime()
        
        return {
            'overall_score': current.overall_score,
            'status': self.determine_health_status(current.overall_score),
            'uptime_seconds': uptime_seconds,
            'uptime_formatted': self.format_uptime(uptime_seconds),
            'success_rate': current.success_rate,
            'alert_count': current.alert_count,
            'last_check_time': current.last_check_time.isoformat()
        }