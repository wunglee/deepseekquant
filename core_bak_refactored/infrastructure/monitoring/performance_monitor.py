"""
性能监控器

职责：
1. 细粒度跟踪单个API请求性能
2. 统计缓存命中率
3. 记录数据源使用情况
4. 提供原始性能指标供上层聚合

来源：从 core/data/analytics/performance.py 迁移而来（属于通用基础设施）

使用方：可被所有模块（data、backtest、risk、portfolio等）复用
"""
from typing import Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    底层请求追踪器（通用性能监控）
    
    职责：
    - 跟踪单个API请求的性能数据
    - 提供细粒度的缓存、错误、数据源统计
    - 不包含业务逻辑，仅收集原始指标
    
    使用示例：
        >>> tracker = PerformanceMonitor()
        >>> tracker.record_request('AAPL', True, 0.5, 'yahoo')
        >>> tracker.record_cache_hit()
        >>> summary = tracker.get_summary()
    """
    
    def __init__(self):
        """初始化性能监控器"""
        self.metrics = {
            'requests_total': 0,
            'requests_failed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'data_points_processed': 0,
            'avg_response_time': 0.0,
            'last_update': None,
            'start_time': datetime.now(),
            'error_counts': {},
            'source_usage': {}
        }
    
    def record_request(self, symbol: str, success: bool, response_time: float, source: str = None):
        """
        记录一次请求
        
        Args:
            symbol: 股票代码
            success: 是否成功
            response_time: 响应时间（秒）
            source: 数据源
        """
        self.metrics['requests_total'] += 1
        
        if not success:
            self.metrics['requests_failed'] += 1
        
        # 更新平均响应时间（指数移动平均）
        alpha = 0.1
        if self.metrics['avg_response_time'] == 0:
            self.metrics['avg_response_time'] = response_time
        else:
            self.metrics['avg_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self.metrics['avg_response_time']
            )
        
        # 记录数据源使用
        if source:
            self.metrics['source_usage'][source] = self.metrics['source_usage'].get(source, 0) + 1
        
        self.metrics['last_update'] = datetime.now().isoformat()
    
    def record_cache_hit(self):
        """记录缓存命中"""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        self.metrics['cache_misses'] += 1
    
    def record_error(self, error_type: str):
        """
        记录错误
        
        Args:
            error_type: 错误类型
        """
        self.metrics['error_counts'][error_type] = self.metrics['error_counts'].get(error_type, 0) + 1
    
    def record_data_points(self, count: int):
        """
        记录处理的数据点数量
        
        Args:
            count: 数据点数量
        """
        self.metrics['data_points_processed'] += count
    
    def get_cache_hit_rate(self) -> float:
        """
        获取缓存命中率
        
        Returns:
            缓存命中率（0-1）
        """
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total == 0:
            return 0.0
        return self.metrics['cache_hits'] / total
    
    def get_error_rate(self) -> float:
        """
        获取错误率
        
        Returns:
            错误率（0-1）
        """
        total = self.metrics['requests_total']
        if total == 0:
            return 0.0
        return self.metrics['requests_failed'] / total
    
    def get_uptime(self) -> timedelta:
        """
        获取运行时长
        
        Returns:
            运行时长
        """
        return datetime.now() - self.metrics['start_time']
    
    def get_throughput(self) -> float:
        """
        获取吞吐量（请求数/秒）
        
        Returns:
            吞吐量
        """
        uptime_seconds = self.get_uptime().total_seconds()
        if uptime_seconds == 0:
            return 0.0
        return self.metrics['requests_total'] / uptime_seconds
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Returns:
            性能摘要字典
        """
        cache_hit_rate = self.get_cache_hit_rate()
        error_rate = self.get_error_rate()
        uptime = self.get_uptime()
        throughput = self.get_throughput()
        
        return {
            'total_requests': self.metrics['requests_total'],
            'failed_requests': self.metrics['requests_failed'],
            'success_rate': 1.0 - error_rate,
            'error_rate': error_rate,
            'cache_hits': self.metrics['cache_hits'],
            'cache_misses': self.metrics['cache_misses'],
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time': self.metrics['avg_response_time'],
            'data_points_processed': self.metrics['data_points_processed'],
            'uptime_seconds': uptime.total_seconds(),
            'throughput_rps': throughput,
            'source_usage': self.metrics['source_usage'],
            'error_counts': self.metrics['error_counts'],
            'last_update': self.metrics['last_update']
        }
    
    def reset(self):
        """重置所有指标"""
        self.__init__()
        logger.info("性能监控指标已重置")
    
    def log_summary(self):
        """记录性能摘要到日志"""
        summary = self.get_summary()
        
        logger.info("=== 性能监控摘要 ===")
        logger.info(f"总请求数: {summary['total_requests']}")
        logger.info(f"成功率: {summary['success_rate']:.2%}")
        logger.info(f"缓存命中率: {summary['cache_hit_rate']:.2%}")
        logger.info(f"平均响应时间: {summary['avg_response_time']:.3f}s")
        logger.info(f"吞吐量: {summary['throughput_rps']:.2f} 请求/秒")
        logger.info(f"运行时长: {summary['uptime_seconds']:.0f}s")
        logger.info("===================")


def create_performance_report(metrics: Dict[str, Any]) -> str:
    """
    创建性能报告（通用工具函数）
    
    Args:
        metrics: 性能指标字典（可来自 PerformanceMonitor 或其他监控系统）
    
    Returns:
        格式化的性能报告字符串
    
    使用示例：
        >>> from core_bak_refactored.infrastructure.monitoring import PerformanceMonitor, create_performance_report
        >>> monitor = PerformanceMonitor()
        >>> # ... 记录一些请求 ...
        >>> summary = monitor.get_summary()
        >>> report = create_performance_report(summary)
        >>> print(report)
    """
    report_lines = [
        "=" * 60,
        "性能报告",
        "=" * 60,
        f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "请求统计:",
        f"  总请求数: {metrics.get('requests_total', 0)}",
        f"  成功请求: {metrics.get('requests_total', 0) - metrics.get('requests_failed', 0)}",
        f"  失败请求: {metrics.get('requests_failed', 0)}",
        f"  成功率: {(1 - metrics.get('requests_failed', 0) / max(metrics.get('requests_total', 1), 1)) * 100:.1f}%",
        "",
        "缓存统计:",
        f"  缓存命中: {metrics.get('cache_hits', 0)}",
        f"  缓存未命中: {metrics.get('cache_misses', 0)}",
        f"  命中率: {metrics.get('cache_hits', 0) / max(metrics.get('cache_hits', 0) + metrics.get('cache_misses', 0), 1) * 100:.1f}%",
        "",
        "性能指标:",
        f"  平均响应时间: {metrics.get('avg_response_time', 0):.3f}s",
        f"  数据点处理量: {metrics.get('data_points_processed', 0)}",
        "",
        "数据源使用:",
    ]
    
    # 添加数据源使用统计
    source_usage = metrics.get('source_usage', {})
    if source_usage:
        for source, count in sorted(source_usage.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {source}: {count} 次")
    else:
        report_lines.append("  无记录")
    
    report_lines.append("")
    report_lines.append("=" * 60)
    
    return "\n".join(report_lines)
