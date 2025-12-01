"""
质量分析计算工具 - 基础设施层

职责：提供与业务无关的纯数学/统计计算函数，用于质量趋势和异常分析
- 质量趋势计算算法
- 异常检测算法
- 错误分布分析算法

架构原则：
- 不包含任何业务领域概念
- 只接收纯数值数据
- 参数全部显式传入，不使用业务默认值
- 函数命名使用数学/统计术语，而非业务术语
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.QualityAnalysisCalculators')


class QualityAnalysisCalculators:
    """质量分析计算工具类（纯数学/统计），不包含业务术语"""
    
    @staticmethod
    def calculate_quality_trend(quality_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        计算质量趋势
        
        Args:
            quality_data: 质量数据列表
            
        Returns:
            趋势数据列表
        """
        trend_data = []
        for item in quality_data:
            trend_data.append({
                'timestamp': item.get('timestamp', ''),
                'score': item.get('overall_score', 0),
                'anomaly_count': item.get('anomaly_count', 0)
            })
        return trend_data
    
    @staticmethod
    def prepare_anomaly_data(quality_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        准备异常数据
        
        Args:
            quality_data: 质量数据列表
            
        Returns:
            异常数据列表
        """
        anomaly_data = []
        for item in quality_data:
            count = item.get('anomaly_count', 0)
            if count > 0:
                anomaly_data.append({
                    'timestamp': item.get('timestamp', ''),
                    'count': count,
                    'level': QualityAnalysisCalculators.determine_anomaly_level(count),
                    'details': item.get('anomaly_details', {})
                })
        return anomaly_data
    
    @staticmethod
    def determine_anomaly_level(count: int) -> str:
        """
        确定异常级别
        
        Args:
            count: 异常数量
            
        Returns:
            级别 ('none', 'low', 'medium', 'high', 'critical')
        """
        if count == 0:
            return 'none'
        elif count <= 5:
            return 'low'
        elif count <= 15:
            return 'medium'
        elif count <= 30:
            return 'high'
        else:
            return 'critical'
    
    @staticmethod
    def calculate_error_distribution(quality_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        计算错误类型分布
        
        Args:
            quality_data: 质量数据列表
            
        Returns:
            错误类型分布字典
        """
        error_dist = {}
        for item in quality_data:
            errors = item.get('errors', {})
            for error_type, count in errors.items():
                error_dist[error_type] = error_dist.get(error_type, 0) + count
        return error_dist
    
    @staticmethod
    def analyze_quality_history(quality_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析质量历史
        
        Args:
            quality_history: 质量历史数据列表
            
        Returns:
            分析结果
        """
        if not quality_history:
            return {
                'avg_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'trend': 'unknown',
                'total_issues': 0,
                'issue_breakdown': {}
            }
        
        scores = [q['overall_score'] for q in quality_history]
        
        return {
            'avg_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'trend': 'improving' if scores[-1] > scores[0] else 'degrading',
            'total_issues': sum(q.get('error_count', 0) for q in quality_history),
            'data_points': len(quality_history)
        }
    
    @staticmethod
    def analyze_alerts_history(alert_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析告警历史
        
        Args:
            alert_history: 告警历史数据列表
            
        Returns:
            分析结果
        """
        if not alert_history:
            return {
                'total_alerts': 0,
                'by_level': {},
                'by_severity': {},
                'by_source': {},
                'trend': 'stable'
            }
        
        # 按级别统计
        by_level = {}
        by_severity = {}
        by_source = {}
        
        for alert in alert_history:
            level = alert.get('level', 'unknown')
            severity = alert.get('severity', 'unknown')
            source = alert.get('data_source', 'unknown')
            
            by_level[level] = by_level.get(level, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            'total_alerts': len(alert_history),
            'by_level': by_level,
            'by_severity': by_severity,
            'by_source': by_source,
            'trend': 'increasing' if len(alert_history) > 10 else 'stable'
        }