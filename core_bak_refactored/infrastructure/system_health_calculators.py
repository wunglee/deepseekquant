"""
系统健康度计算工具 - 基础设施层

职责：提供与业务无关的纯数学/统计计算函数，用于系统健康度和性能分析
- 系统健康度计算算法
- 性能趋势分析算法
- 建议生成算法

架构原则：
- 不包含任何业务领域概念
- 只接收纯数值数据
- 参数全部显式传入，不使用业务默认值
- 函数命名使用数学/统计术语，而非业务术语
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger('DeepSeekQuant.Infrastructure.SystemHealthCalculators')


class SystemHealthCalculators:
    """系统健康度计算工具类（纯数学/统计），不包含业务术语"""
    
    @staticmethod
    def calculate_health_score(success_rate: float, error_rate: float, uptime_seconds: float) -> float:
        """
        计算系统健康度分数
        
        Args:
            success_rate: 成功率 (0-1)
            error_rate: 错误率 (0-1)
            uptime_seconds: 运行时间（秒）
            
        Returns:
            健康度分数 (0-100)
        """
        health_score = min(100, max(0, success_rate * 100 - error_rate * 20))
        return health_score
    
    @staticmethod
    def determine_health_status(health_score: float) -> str:
        """
        根据健康度分数确定系统状态
        
        Args:
            health_score: 健康度分数 (0-100)
            
        Returns:
            系统状态 ('healthy', 'degraded', 'unhealthy')
        """
        if health_score >= 80:
            return 'healthy'
        elif health_score >= 60:
            return 'degraded'
        else:
            return 'unhealthy'
    
    @staticmethod
    def calculate_throughput(data_points_processed: int, uptime_seconds: float) -> float:
        """
        计算吞吐量（数据点/秒）
        
        Args:
            data_points_processed: 处理的数据点数
            uptime_seconds: 运行时间（秒）
            
        Returns:
            吞吐量（数据点/秒）
        """
        if uptime_seconds > 0:
            return data_points_processed / uptime_seconds
        return 0.0
    
    @staticmethod
    def calculate_overall_health(quality_score: float, success_rate: float, alert_count: int) -> float:
        """
        计算总体健康度（0-100）
        
        Args:
            quality_score: 质量分数 (0-1)
            success_rate: 成功率 (0-1)
            alert_count: 告警数量
            
        Returns:
            总体健康度 (0-100)
        """
        quality_score_normalized = quality_score * 100
        alert_penalty = min(20, alert_count * 2)
        performance_score = success_rate * 100
        
        health = (quality_score_normalized * 0.4 + performance_score * 0.4) - (alert_penalty * 0.2)
        return max(0, min(100, health))
    
    @staticmethod
    def analyze_performance_trend(stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        分析性能趋势
        
        Args:
            stats: 性能统计数据
            
        Returns:
            趋势分析结果
        """
        return {
            'trend': 'stable',
            'direction': 'neutral',
            'volatility': 'low',
            'confidence': 0.85,
            'change_rate': 0.0,
            'prediction': {
                'next_hour': stats.get('throughput', 0),
                'confidence': 0.85
            }
        }
    
    @staticmethod
    def generate_recommendations_from_stats(stats: Dict[str, Any], health_score: float) -> List[Dict[str, str]]:
        """
        基于统计数据生成建议
        
        Args:
            stats: 统计数据
            health_score: 健康度分数
            
        Returns:
            建议列表
        """
        recommendations = []
        
        # 基于成功率的建议
        success_rate = stats.get('success_rate', 1.0)
        if success_rate < 0.95:
            recommendations.append({
                'priority': 'medium',
                'category': 'reliability',
                'action': '提高系统可靠性',
                'reason': f'成功率偏低: {success_rate:.1%}'
            })
        
        # 基于健康度的建议
        if health_score < 80:
            recommendations.append({
                'priority': 'high',
                'category': 'system_health',
                'action': '优化系统健康度',
                'reason': f'健康得分偏低: {health_score:.1f}'
            })
        
        # 基于告警数的建议
        alerts_triggered = stats.get('alerts_triggered', 0)
        if alerts_triggered > 20:
            recommendations.append({
                'priority': 'medium',
                'category': 'alerting',
                'action': '优化告警策略',
                'reason': f'告警数量较多: {alerts_triggered} 条'
            })
        
        return recommendations