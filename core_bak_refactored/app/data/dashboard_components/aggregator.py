"""仪表板数据聚合器

[应用层 - Dashboard组件] 从dashboard.py拆分而来
状态: ✅ 第四轮迁移 - 数据聚合和转换
来源: dashboard_bak.py 相关方法
迁移时间: 2025-11-28

包含功能:
- 质量数据聚合
- 异常数据准备
- 性能数据转换
- 错误分布计算
- 警报分组
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List
from core_bak_refactored.infrastructure import QualityAnalysisCalculators

logger = logging.getLogger('DeepSeekQuant.App.Dashboard.Aggregator')


class DashboardDataAggregator:
    """仪表板数据聚合器 - 处理数据聚合和转换"""

    def __init__(self, quality_monitor: Any) -> None:
        """初始化数据聚合器
        
        Args:
            quality_monitor: 质量监控器实例
        """
        self._qm = quality_monitor

    def get_current_quality_data(self) -> Dict[str, Any]:
        """获取当前质量数据
        
        Returns:
            包含质量数据和统计信息的字典
        """
        # 获取最近24小时的质量数据
        quality_data = self._qm.get_quality_history(24)
        
        # 计算汇总统计
        if quality_data:
            latest = quality_data[-1]
            avg_score = sum(q.get('overall_score', 0) for q in quality_data) / len(quality_data)
            total_anomalies = sum(q.get('anomaly_count', 0) for q in quality_data)
        else:
            latest = {}
            avg_score = 0
            total_anomalies = 0

        return {
            'current_score': latest.get('overall_score', 0),
            'average_score': avg_score,
            'total_anomalies': total_anomalies,
            'data_points': len(quality_data),
            'trend': self.calculate_quality_trend(quality_data),
            'anomaly_data': self.prepare_anomaly_data(quality_data),
            'timestamp': latest.get('timestamp', '')
        }

    def calculate_quality_trend(self, quality_data: List[Dict]) -> List[Dict]:
        """计算质量趋势
        
        Args:
            quality_data: 质量数据列表
            
        Returns:
            趋势数据列表
        """
        return QualityAnalysisCalculators.calculate_quality_trend(quality_data)

    def prepare_anomaly_data(self, quality_data: List[Dict]) -> List[Dict]:
        """准备异常数据
        
        Args:
            quality_data: 质量数据列表
            
        Returns:
            异常数据列表
        """
        return QualityAnalysisCalculators.prepare_anomaly_data(quality_data)

    def determine_anomaly_level(self, count: int) -> str:
        """确定异常级别
        
        Args:
            count: 异常数量
            
        Returns:
            级别 ('low', 'medium', 'high', 'critical')
        """
        return QualityAnalysisCalculators.determine_anomaly_level(count)

    def prepare_performance_data(self, performance_stats: Dict) -> Dict[str, Any]:
        """准备性能数据
        
        Args:
            performance_stats: 性能统计数据
            
        Returns:
            格式化的性能数据
        """
        return {
            'throughput': performance_stats.get('throughput', 0),
            'reliability': performance_stats.get('success_rate', 0),
            'accuracy': performance_stats.get('accuracy', 0),
            'timeliness': performance_stats.get('timeliness', 0),
            'completeness': performance_stats.get('completeness', 0)
        }

    def calculate_error_distribution(self, quality_data: List[Dict]) -> Dict[str, int]:
        """计算错误类型分布
        
        Args:
            quality_data: 质量数据列表
            
        Returns:
            错误类型分布字典
        """
        return QualityAnalysisCalculators.calculate_error_distribution(quality_data)

    def group_alerts_by_level(self, alerts: List[Dict]) -> Dict[str, int]:
        """按级别分组警报
        
        Args:
            alerts: 警报列表
            
        Returns:
            按级别分组的统计
        """
        # 使用质量分析计算工具进行分组
        result = {}
        for alert in alerts:
            level = alert.get('level', 'unknown')
            result[level] = result.get(level, 0) + 1
        return result

    def get_report_data(self, report_id: str) -> Dict[str, Any]:
        """获取报告数据
        
        Args:
            report_id: 报告ID
            
        Returns:
            报告数据
        """
        # 这里实现报告数据获取逻辑
        return {
            'report_id': report_id,
            'status': 'completed',
            'data': {},
            'generated_at': ''
        }
