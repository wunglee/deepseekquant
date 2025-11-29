"""
数据质量评估模块（从 DataFetcher.get_data_quality_metrics 迁移而来）

职责：
1. 评估数据完整性
2. 检测数据异常
3. 计算数据质量指标
4. 生成质量报告
"""
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def get_data_quality_metrics(
    cache_stats: Dict[str, Any],
    performance_metrics: Dict[str, Any],
    recent_errors: List[str] = None
) -> Dict[str, Any]:
    """
    获取数据质量指标（从 DataFetcher.get_data_quality_metrics 迁移而来）。
    
    Args:
        cache_stats: 缓存统计信息
        performance_metrics: 性能指标
        recent_errors: 最近的错误列表
    
    Returns:
        数据质量指标字典，包含：
        - completeness: 数据完整性评分(0-1)
        - timeliness: 数据及时性评分(0-1)
        - accuracy: 数据准确性评分(0-1)
        - consistency: 数据一致性评分(0-1)
        - overall_quality: 总体质量评分(0-1)
        - cache_hit_rate: 缓存命中率
        - error_rate: 错误率
        - data_freshness: 数据新鲜度
    
    Example:
        >>> quality = get_data_quality_metrics(cache_stats, perf_metrics)
        >>> # {'completeness': 0.95, 'timeliness': 0.90, ...}
    """
    try:
        # 计算缓存命中率
        total_requests = cache_stats.get('hits', 0) + cache_stats.get('misses', 0)
        cache_hit_rate = cache_stats.get('hits', 0) / total_requests if total_requests > 0 else 0

        # 计算错误率
        total_api_requests = performance_metrics.get('requests_total', 0)
        failed_requests = performance_metrics.get('requests_failed', 0)
        error_rate = failed_requests / total_api_requests if total_api_requests > 0 else 0

        # 评估数据完整性（基于成功率）
        completeness_score = 1.0 - error_rate

        # 评估数据及时性（基于响应时间）
        avg_response_time = performance_metrics.get('avg_response_time', 0)
        if avg_response_time < 1.0:
            timeliness_score = 1.0
        elif avg_response_time < 5.0:
            timeliness_score = 0.8
        elif avg_response_time < 10.0:
            timeliness_score = 0.6
        else:
            timeliness_score = 0.4

        # 评估数据准确性（基于错误类型）
        # 简化实现：基于错误数量
        recent_error_count = len(recent_errors) if recent_errors else 0
        if recent_error_count == 0:
            accuracy_score = 1.0
        elif recent_error_count < 5:
            accuracy_score = 0.9
        elif recent_error_count < 10:
            accuracy_score = 0.7
        else:
            accuracy_score = 0.5

        # 评估数据一致性（基于缓存一致性）
        # 简化实现：基于缓存命中率
        consistency_score = cache_hit_rate

        # 计算总体质量评分（加权平均）
        overall_quality = (
            completeness_score * 0.3 +
            timeliness_score * 0.3 +
            accuracy_score * 0.2 +
            consistency_score * 0.2
        )

        # 评估数据新鲜度
        last_update_str = performance_metrics.get('last_update')
        if last_update_str:
            try:
                last_update = datetime.fromisoformat(last_update_str)
                now = datetime.now()
                age_seconds = (now - last_update).total_seconds()
                
                if age_seconds < 60:
                    data_freshness = 'excellent'
                elif age_seconds < 300:
                    data_freshness = 'good'
                elif age_seconds < 3600:
                    data_freshness = 'moderate'
                else:
                    data_freshness = 'stale'
            except Exception:
                data_freshness = 'unknown'
        else:
            data_freshness = 'unknown'

        result = {
            'completeness': completeness_score,
            'timeliness': timeliness_score,
            'accuracy': accuracy_score,
            'consistency': consistency_score,
            'overall_quality': overall_quality,
            'cache_hit_rate': cache_hit_rate,
            'error_rate': error_rate,
            'data_freshness': data_freshness,
            'total_requests': total_api_requests,
            'failed_requests': failed_requests,
            'avg_response_time': avg_response_time,
            'recent_errors': recent_error_count,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"数据质量评估完成: 总体质量={overall_quality:.2%}, "
            f"完整性={completeness_score:.2%}, 及时性={timeliness_score:.2%}"
        )

        return result

    except Exception as e:
        logger.error(f"获取数据质量指标失败: {e}")
        return {
            'completeness': 0,
            'timeliness': 0,
            'accuracy': 0,
            'consistency': 0,
            'overall_quality': 0,
            'cache_hit_rate': 0,
            'error_rate': 1.0,
            'data_freshness': 'unknown',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def detect_data_anomalies(data: List[Any], threshold: float = 3.0) -> Dict[str, Any]:
    """
    检测数据异常（基于统计方法）。
    
    Args:
        data: 数据列表（MarketData对象或数据字典）
        threshold: 异常检测阈值（标准差倍数），默认3.0
    
    Returns:
        异常检测结果，包含：
        - has_anomalies: 是否存在异常
        - anomaly_count: 异常数量
        - anomaly_indices: 异常数据的索引列表
        - anomaly_reasons: 异常原因列表
    
    Example:
        >>> anomalies = detect_data_anomalies(market_data_list)
        >>> # {'has_anomalies': True, 'anomaly_count': 2, ...}
    """
    if not data or len(data) < 3:
        return {
            'has_anomalies': False,
            'anomaly_count': 0,
            'anomaly_indices': [],
            'anomaly_reasons': [],
            'message': '数据不足，无法进行异常检测'
        }

    try:
        import numpy as np
        
        # 提取收盘价
        closes = []
        for item in data:
            if hasattr(item, 'close'):
                closes.append(item.close)
            elif isinstance(item, dict):
                closes.append(item.get('close', 0))
        
        if len(closes) < 3:
            return {
                'has_anomalies': False,
                'anomaly_count': 0,
                'anomaly_indices': [],
                'anomaly_reasons': []
            }

        # 计算收益率
        returns = []
        for i in range(1, len(closes)):
            if closes[i-1] > 0:
                ret = (closes[i] - closes[i-1]) / closes[i-1]
                returns.append(ret)
            else:
                returns.append(0)

        # 计算均值和标准差
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # 检测异常值
        anomaly_indices = []
        anomaly_reasons = []
        
        for i, ret in enumerate(returns):
            z_score = abs((ret - mean_return) / std_return) if std_return > 0 else 0
            
            if z_score > threshold:
                anomaly_indices.append(i + 1)  # +1因为returns从索引1开始
                anomaly_reasons.append(f"收益率异常: {ret:.2%} (Z-score: {z_score:.2f})")

        has_anomalies = len(anomaly_indices) > 0

        result = {
            'has_anomalies': has_anomalies,
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_reasons': anomaly_reasons,
            'mean_return': mean_return,
            'std_return': std_return,
            'threshold': threshold
        }

        if has_anomalies:
            logger.warning(f"检测到 {len(anomaly_indices)} 个数据异常点")

        return result

    except Exception as e:
        logger.error(f"数据异常检测失败: {e}")
        return {
            'has_anomalies': False,
            'anomaly_count': 0,
            'anomaly_indices': [],
            'anomaly_reasons': [],
            'error': str(e)
        }
