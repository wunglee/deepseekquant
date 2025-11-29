"""数据质量API控制器 - 辅助方法集合

从api_service.py迁移的辅助方法
迁移时间: 2025-11-29
状态: 进行中
"""

from typing import Any, Dict, List
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger('DeepSeekQuant.App.APIControllers')


class DataQualityControllers:
    """API控制器 - 处理业务逻辑和数据转换"""

    def __init__(self, quality_monitor: Any) -> None:
        self._qm = quality_monitor

    def get_quality_current(self, hours: int = 24) -> Dict:
        """获取当前质量数据（带元数据）"""
        quality_data = self._qm.get_quality_history(hours)
        return {
            'data': quality_data,
            'metadata': {
                'data_points': len(quality_data),
                'time_range': f'last_{hours}_hours',
                'quality_score_avg': np.mean(
                    [q.get('overall_score', 0) for q in quality_data]) if quality_data else 0,
                'anomaly_count_total': sum(q.get('anomaly_count', 0) for q in quality_data)
            }
        }

    def get_alerts_with_pagination(self, hours: int, level: str = None,
                                   severity: str = None, data_source: str = None,
                                   page: int = 1, per_page: int = 50) -> Dict:
        """获取警报历史（支持过滤和分页）"""
        alerts = self._qm.get_alert_history(hours)

        # 应用过滤器
        if level:
            alerts = [a for a in alerts if a.get('level') == level]
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        if data_source:
            alerts = [a for a in alerts if a.get('data_source') == data_source]

        # 分页
        total_alerts = len(alerts)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_alerts = alerts[start_idx:end_idx]

        return {
            'alerts': paginated_alerts,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_alerts,
                'pages': (total_alerts + per_page - 1) // per_page
            },
            'summary': {
                'total_alerts': total_alerts,
                'by_level': self.group_by_level(alerts),
                'by_severity': self.group_by_severity(alerts),
                'by_source': self.group_by_source(alerts)
            }
        }

    def get_enhanced_performance(self) -> Dict:
        """获取增强的性能统计"""
        stats = self._qm.get_performance_statistics()
        return {
            **stats,
            'system_health': self.calculate_system_health(stats),
            'trend_analysis': self.analyze_performance_trend(stats)
        }

    # 辅助方法 - 从api_service.py迁移

    def group_by_level(self, alerts: List[Dict]) -> Dict[str, int]:
        """按级别分组警报"""
        levels = {}
        for alert in alerts:
            level = alert.get('level', 'unknown')
            if level not in levels:
                levels[level] = 0
            levels[level] += 1
        return levels

    def group_by_severity(self, alerts: List[Dict]) -> Dict[str, int]:
        """按严重性分组警报"""
        severities = {}
        for alert in alerts:
            severity = alert.get('severity', 'medium')
            if severity not in severities:
                severities[severity] = 0
            severities[severity] += 1
        return severities

    def group_by_source(self, alerts: List[Dict]) -> Dict[str, int]:
        """按数据源分组警报"""
        sources = {}
        for alert in alerts:
            source = alert.get('data_source', 'unknown')
            if source not in sources:
                sources[source] = 0
            sources[source] += 1
        return sources

    def calculate_system_health(self, stats: Dict) -> Dict[str, Any]:
        """计算系统健康度"""
        success_rate = stats.get('success_rate', 0)
        error_rate = 1 - success_rate
        uptime = stats.get('uptime_seconds', 0)

        health_score = min(100, max(0, success_rate * 100 - error_rate * 20))

        return {
            'score': health_score,
            'status': 'healthy' if health_score >= 80 else ('degraded' if health_score >= 60 else 'unhealthy'),
            'indicators': {
                'success_rate': success_rate,
                'error_rate': error_rate,
                'uptime': uptime,
                'stability': stats.get('stability_score', 0)
            }
        }

    def analyze_performance_trend(self, stats: Dict) -> Dict[str, Any]:
        """分析性能趋势"""
        return {
            'trend': 'stable',
            'change_rate': 0.0,
            'prediction': {
                'next_hour': stats.get('throughput', 0),
                'confidence': 0.85
            }
        }
