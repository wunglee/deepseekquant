"""
数据质量监控适配器（应用层）

职责：
- 为api_service.py提供DataQualityMonitor接口
- 整合重构后的组件：Alert Manager, DataQualityChecker等
- 适配遗留API到新架构

设计原则：
- 适配器模式：提供遗留接口，内部使用重构后组件
- 应用层组件：不包含核心业务逻辑，仅做数据转换和组合
- 可测试：依赖注入，方便Mock
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from core_bak_refactored.core.monitoring.alert_manager import AlertManager, AlertRecord, AlertSeverity

logger = logging.getLogger('DeepSeekQuant.QualityMonitorAdapter')


class QualityMonitorAdapter:
    """
    质量监控适配器
    
    为api_service.py提供必要的接口，内部整合重构后的组件
    
    必需接口（api_service.py依赖）：
    1. get_quality_history(hours) -> List[Dict]
    2. get_alert_history(hours) -> List[Dict]
    3. get_performance_statistics() -> Dict
    4. generate_comprehensive_report(period) -> Dict
    
    使用示例：
        alert_manager = AlertManager(alert_config)
        monitor = QualityMonitorAdapter(alert_manager=alert_manager)
        
        # API服务使用
        api_service = DataQualityAPIService(monitor)
    """
    
    def __init__(self,
                 alert_manager: Optional[AlertManager] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化适配器
        
        Args:
            alert_manager: 告警管理器（如果为None则创建默认实例）
            config: 配置字典
        """
        self.config = config or {}
        self.alert_manager = alert_manager
        
        # 质量历史（应用层临时存储，生产环境应使用数据库）
        self._quality_history: List[Dict[str, Any]] = []
        
        # 性能统计
        self._performance_stats = {
            'monitoring_cycles': 0,
            'data_points_processed': 0,
            'anomalies_detected': 0,
            'validation_errors': 0,
            'alerts_triggered': 0,
            'success_rate': 1.0,
            'uptime_seconds': 0,
            'throughput': 0,
            'stability_score': 1.0,
            'start_time': datetime.now().isoformat()
        }
        
        logger.info("QualityMonitorAdapter initialized")
    
    def get_quality_history(self, hours: int = 24) -> List[Dict]:
        """
        获取质量历史数据
        
        Args:
            hours: 查询的小时数
        
        Returns:
            质量记录列表，每条包含：
            - timestamp: 时间戳
            - overall_score: 总体评分
            - anomaly_count: 异常数量
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        result = []
        for record in self._quality_history:
            record_time = datetime.fromisoformat(record['timestamp'])
            if record_time >= cutoff_time:
                result.append(record)
        
        return result
    
    def get_alert_history(self, hours: int = 24) -> List[Dict]:
        """
        获取警报历史
        
        Args:
            hours: 查询的小时数
        
        Returns:
            警报记录列表，每条包含：
            - timestamp: 时间戳
            - level: 级别 (critical, warning等)
            - severity: 严重性 (high, medium, low)
            - data_source: 数据源
            - message: 消息
        """
        if not self.alert_manager:
            return []
        
        # 从AlertManager获取告警
        since = datetime.now() - timedelta(hours=hours)
        alert_records: List[AlertRecord] = self.alert_manager.get_alert_history(
            since=since,
            limit=1000
        )
        
        # 转换为字典格式
        result = []
        for record in alert_records:
            result.append({
                'timestamp': record.created_at.isoformat(),
                'level': self._map_severity_to_level(record.severity),
                'severity': self._map_severity_to_legacy(record.severity),
                'data_source': record.metadata.get('data_source', 'unknown'),
                'message': record.message,
                'title': record.title,
                'alert_id': record.alert_id
            })
        
        return result
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        获取性能统计
        
        Returns:
            性能统计字典，包含：
            - success_rate: 成功率
            - uptime_seconds: 运行时间（秒）
            - throughput: 吞吐量
            - stability_score: 稳定性评分
        """
        # 计算运行时间
        start_time = datetime.fromisoformat(self._performance_stats['start_time'])
        uptime_seconds = (datetime.now() - start_time).total_seconds()
        
        stats = self._performance_stats.copy()
        stats['uptime_seconds'] = int(uptime_seconds)
        
        # 计算吞吐量
        if uptime_seconds > 0:
            stats['throughput'] = stats['data_points_processed'] / uptime_seconds
        
        return stats
    
    def generate_comprehensive_report(self, period: str = '7d') -> Dict[str, Any]:
        """
        生成综合报告
        
        Args:
            period: 报告周期 (1d, 7d, 30d)
        
        Returns:
            综合报告字典
        """
        # 解析周期
        period_map = {'1d': 24, '7d': 168, '30d': 720}
        hours = period_map.get(period, 168)
        
        # 获取数据
        quality_data = self.get_quality_history(hours)
        alert_data = self.get_alert_history(hours)
        perf_stats = self.get_performance_statistics()
        
        # 计算统计
        if quality_data:
            avg_score = sum(q['overall_score'] for q in quality_data) / len(quality_data)
            total_anomalies = sum(q.get('anomaly_count', 0) for q in quality_data)
        else:
            avg_score = 0.0
            total_anomalies = 0
        
        # 按级别统计警报
        alert_by_level = {}
        for alert in alert_data:
            level = alert['level']
            alert_by_level[level] = alert_by_level.get(level, 0) + 1
        
        return {
            'report_id': f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'period': period,
            'generation_time': datetime.now().isoformat(),
            'quality_summary': {
                'average_score': avg_score,
                'total_anomalies': total_anomalies,
                'data_points': len(quality_data)
            },
            'alert_summary': {
                'total_alerts': len(alert_data),
                'by_level': alert_by_level
            },
            'performance_summary': perf_stats,
            'quality_data': quality_data,
            'alert_data': alert_data
        }
    
    # ==================== 辅助方法 ====================
    
    def _map_severity_to_level(self, severity: AlertSeverity) -> str:
        """将AlertSeverity映射到遗留的level字段"""
        mapping = {
            AlertSeverity.INFO: 'info',
            AlertSeverity.WARNING: 'warning',
            AlertSeverity.ERROR: 'critical',  # Level 2 -> critical
            AlertSeverity.CRITICAL: 'critical'  # Level 3 -> critical
        }
        return mapping.get(severity, 'warning')
    
    def _map_severity_to_legacy(self, severity: AlertSeverity) -> str:
        """将AlertSeverity映射到遗留的severity字段"""
        mapping = {
            AlertSeverity.INFO: 'low',
            AlertSeverity.WARNING: 'medium',
            AlertSeverity.ERROR: 'high',
            AlertSeverity.CRITICAL: 'high'
        }
        return mapping.get(severity, 'medium')
    
    def record_quality(self,
                      overall_score: float,
                      anomaly_count: int = 0,
                      details: Optional[Dict[str, Any]] = None):
        """
        记录质量数据（供内部使用）
        
        Args:
            overall_score: 总体质量评分 (0-1)
            anomaly_count: 异常数量
            details: 详细信息
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'anomaly_count': anomaly_count
        }
        
        if details:
            record.update(details)
        
        self._quality_history.append(record)
        
        # 限制历史长度
        max_history = self.config.get('max_history', 10000)
        if len(self._quality_history) > max_history:
            self._quality_history = self._quality_history[-max_history:]
        
        # 更新统计
        self._performance_stats['data_points_processed'] += 1
        self._performance_stats['anomalies_detected'] += anomaly_count
    
    def update_performance_stats(self, **kwargs):
        """更新性能统计（供内部使用）"""
        self._performance_stats.update(kwargs)
