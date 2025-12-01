"""系统指标和资源监控模块

[应用层 - API组件] 从api_service.py拆分而来
状态: ✅ 第二轮迁移 - 资源监控和指标收集
来源: api_service_bak.py 相关方法
迁移时间: 2025-11-28

包含功能:
- 系统资源利用率监控
- 网络IO统计
- 性能指标收集
- 资源使用建议生成
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List

import psutil

logger = logging.getLogger('DeepSeekQuant.App.API.Metrics')


class MetricsCollector:
    """系统指标收集器 - 监控系统资源和性能指标"""

    def __init__(self, quality_monitor: Any) -> None:
        """初始化指标收集器
        
        Args:
            quality_monitor: 质量监控器实例
        """
        self._qm = quality_monitor

    def get_system_metrics(self, metric_type: str, time_range: str, aggregation: str) -> Dict[str, Any]:
        """获取系统指标
        
        Args:
            metric_type: 指标类型 ('all', 'cpu', 'memory', 'disk', 'network')
            time_range: 时间范围 ('1h', '24h', '7d')
            aggregation: 聚合方式 ('hourly', 'daily', 'raw')
            
        Returns:
            包含指标数据的字典
        """
        return {
            'metric_type': metric_type,
            'time_range': time_range,
            'aggregation': aggregation,
            'data': [],
            'summary': {}
        }

    def get_resource_utilization(self) -> Dict[str, Any]:
        """获取资源利用率
        
        Returns:
            包含CPU、内存、磁盘利用率的字典
        """
        try:
            return {
                'cpu': {
                    'percent': psutil.cpu_percent(interval=1),
                    'count': psutil.cpu_count(),
                    'per_cpu': psutil.cpu_percent(interval=1, percpu=True)
                },
                'memory': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'percent': psutil.virtual_memory().percent,
                    'used': psutil.virtual_memory().used
                },
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
        except Exception as e:
            logger.warning(f"获取资源利用率失败: {e}")
            return {'error': str(e)}

    def get_network_io(self) -> Dict[str, Any]:
        """获取网络IO统计
        
        Returns:
            包含网络IO统计数据的字典
        """
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout
            }
        except Exception as e:
            logger.warning(f"获取网络IO失败: {e}")
            return {'error': str(e)}

    def generate_performance_recommendations(self, stats: Dict) -> List[Dict]:
        """生成性能建议
        
        Args:
            stats: 性能统计数据
            
        Returns:
            性能建议列表
        """
        recommendations = []

        success_rate = stats.get('success_rate', 0)
        if success_rate < 0.9:
            recommendations.append({
                'priority': 'high',
                'action': '提高系统成功率',
                'reason': f'当前成功率较低: {success_rate:.1%}',
                'impact': 'high',
                'effort': 'medium'
            })

        avg_processing_time = stats.get('avg_processing_time', 0)
        if avg_processing_time > 5.0:  # 超过5秒
            recommendations.append({
                'priority': 'medium',
                'action': '优化处理性能',
                'reason': f'平均处理时间较长: {avg_processing_time:.2f}秒',
                'impact': 'medium',
                'effort': 'high'
            })

        return recommendations

    def generate_health_recommendations(self, health_score: float, stats: Dict) -> List[Dict]:
        """生成健康度建议
        
        Args:
            health_score: 健康度评分 (0-100)
            stats: 统计数据
            
        Returns:
            健康度建议列表
        """
        recommendations = []

        if health_score < 60:
            recommendations.append({
                'priority': 'critical',
                'action': '立即检查系统健康状况',
                'reason': f'系统健康度严重不足: {health_score:.1f}',
                'impact': 'critical',
                'effort': 'high'
            })
        elif health_score < 80:
            recommendations.append({
                'priority': 'high',
                'action': '优化系统性能',
                'reason': f'系统健康度需要改善: {health_score:.1f}',
                'impact': 'high',
                'effort': 'medium'
            })

        return recommendations
