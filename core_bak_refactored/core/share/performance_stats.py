"""
共享性能统计管理器（业务层）

职责：
- 提供标准化的性能统计接口
- 支持性能指标的收集、计算和报告
- 提供性能数据的类型安全访问
- 集成底层 PerformanceMonitor 进行细粒度追踪

架构说明：
- 业务层指标：异常、告警、验证错误、吞吐量等
- 底层追踪：委托给 PerformanceMonitor（请求级别）
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger('DeepSeekQuant.Core.Share.PerformanceStats')


@dataclass
class PerformanceMetrics:
    """性能指标数据类（业务层）"""
    throughput: float = 0.0  # 吞吐量（数据点/秒）
    success_rate: float = 1.0  # 成功率 (0-1)
    reliability: float = 1.0  # 可靠性 (0-1)
    stability_score: float = 1.0  # 稳定性分数 (0-1)
    uptime_seconds: float = 0.0  # 运行时间（秒）
    uptime_human: str = "0 minutes"  # 人类可读的运行时间
    data_points_processed: int = 0  # 处理的数据点数
    anomalies_detected: int = 0  # 检测到的异常数
    alerts_triggered: int = 0  # 触发的告警数
    validation_errors: int = 0  # 验证错误数
    avg_processing_time: float = 0.0  # 平均处理时间（秒）
    monitoring_cycles: int = 0  # 监控周期数
    start_time: str = ""  # 启动时间
    
    # 底层请求追踪指标（来自 PerformanceMonitor）
    requests_total: int = 0  # 总请求数
    requests_failed: int = 0  # 失败请求数
    cache_hits: int = 0  # 缓存命中数
    cache_misses: int = 0  # 缓存未命中数
    cache_hit_rate: float = 0.0  # 缓存命中率
    avg_response_time: float = 0.0  # 平均响应时间（秒）
    source_usage: Dict[str, int] = None  # 数据源使用统计
    error_counts: Dict[str, int] = None  # 错误类型统计
    
    def __post_init__(self):
        """初始化字典类型字段"""
        if self.source_usage is None:
            self.source_usage = {}
        if self.error_counts is None:
            self.error_counts = {}


class PerformanceStatsManager:
    """
    性能统计管理器（业务层）
    
    职责：
    - 提供标准化的性能统计接口
    - 集成底层 PerformanceMonitor 进行细粒度追踪
    - 聚合业务层和请求层指标
    
    架构说明：
    - 使用 PerformanceMonitor 追踪请求级别性能
    - 自身管理业务层指标（异常、告警等）
    """
    
    def __init__(self, enable_request_tracking: bool = False):
        """
        初始化性能统计管理器
        
        Args:
            enable_request_tracking: 是否启用底层请求追踪（默认False以保持向后兼容）
        """
        self._stats = PerformanceMetrics()
        self._stats.start_time = datetime.now().isoformat()
        self._enable_request_tracking = enable_request_tracking
        self._request_tracker: Optional['PerformanceMonitor'] = None
        
        if enable_request_tracking:
            try:
                from core_bak_refactored.core.data.analytics.performance import PerformanceMonitor
                self._request_tracker = PerformanceMonitor()
                logger.debug("已启用底层请求追踪器")
            except ImportError as e:
                logger.warning(f"无法导入 PerformanceMonitor，请求追踪功能禁用: {e}")
                self._enable_request_tracking = False
    
    def get_stats(self) -> PerformanceMetrics:
        """获取性能统计"""
        return self._stats
    
    def get_stats_dict(self) -> Dict[str, Any]:
        """获取性能统计数据字典"""
        return asdict(self._stats)
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """增加计数器"""
        if hasattr(self._stats, counter_name):
            current_value = getattr(self._stats, counter_name)
            setattr(self._stats, counter_name, current_value + value)
    
    def update_metric(self, metric_name: str, value: Any):
        """更新指标"""
        if hasattr(self._stats, metric_name):
            setattr(self._stats, metric_name, value)
    
    def calculate_throughput(self) -> float:
        """计算吞吐量（数据点/秒）"""
        if self._stats.uptime_seconds > 0:
            return self._stats.data_points_processed / self._stats.uptime_seconds
        return 0.0
    
    def calculate_success_rate(self) -> float:
        """计算成功率"""
        total_attempts = self._stats.data_points_processed + self._stats.validation_errors
        if total_attempts > 0:
            return self._stats.data_points_processed / total_attempts
        return 1.0
    
    def calculate_uptime(self) -> float:
        """计算运行时间（秒）"""
        if self._stats.start_time:
            start_time = datetime.fromisoformat(self._stats.start_time)
            uptime = datetime.now() - start_time
            return uptime.total_seconds()
        return 0.0
    
    def format_uptime(self, seconds: float) -> str:
        """格式化运行时长为人类可读格式"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days} days {hours} hours"
        elif hours > 0:
            return f"{hours} hours {minutes} minutes"
        else:
            return f"{minutes} minutes"
    
    def update_performance_stats(self):
        """更新性能统计"""
        # 更新运行时间
        uptime_seconds = self.calculate_uptime()
        self._stats.uptime_seconds = uptime_seconds
        self._stats.uptime_human = self.format_uptime(uptime_seconds)
        
        # 更新吞吐量
        self._stats.throughput = self.calculate_throughput()
        
        # 更新成功率
        self._stats.success_rate = self.calculate_success_rate()
        
        # 更新可靠性（基于成功率和告警数）
        self._stats.reliability = max(0.0, min(1.0, self._stats.success_rate - (self._stats.alerts_triggered * 0.01)))
        
        # 更新稳定性分数（基于异常数和错误数）
        stability_penalty = (self._stats.anomalies_detected * 0.001) + (self._stats.validation_errors * 0.01)
        self._stats.stability_score = max(0.0, min(1.0, 1.0 - stability_penalty))
    
    def reset_stats(self):
        """重置统计"""
        self._stats = PerformanceMetrics()
        self._stats.start_time = datetime.now().isoformat()
    
    def record_request(self, symbol: str, success: bool, response_time: float, source: str = None):
        """
        记录API请求（如果启用了请求追踪）
        
        Args:
            symbol: 股票代码
            success: 是否成功
            response_time: 响应时间（秒）
            source: 数据源
        """
        if self._request_tracker:
            self._request_tracker.record_request(symbol, success, response_time, source)
    
    def record_cache_hit(self):
        """记录缓存命中"""
        if self._request_tracker:
            self._request_tracker.record_cache_hit()
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        if self._request_tracker:
            self._request_tracker.record_cache_miss()
    
    def record_error(self, error_type: str):
        """记录错误类型"""
        if self._request_tracker:
            self._request_tracker.record_error(error_type)
    
    def _sync_request_tracker_stats(self):
        """从底层追踪器同步请求级别统计"""
        if self._request_tracker:
            tracker_summary = self._request_tracker.get_summary()
            self._stats.requests_total = tracker_summary['total_requests']
            self._stats.requests_failed = tracker_summary['failed_requests']
            self._stats.cache_hits = tracker_summary['cache_hits']
            self._stats.cache_misses = tracker_summary['cache_misses']
            self._stats.cache_hit_rate = tracker_summary['cache_hit_rate']
            self._stats.avg_response_time = tracker_summary['avg_response_time']
            self._stats.source_usage = tracker_summary['source_usage'].copy()
            self._stats.error_counts = tracker_summary['error_counts'].copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要（包含请求追踪数据）"""
        self.update_performance_stats()
        
        # 同步底层追踪器数据
        if self._enable_request_tracking:
            self._sync_request_tracker_stats()
        
        summary = {
            'throughput': self._stats.throughput,
            'success_rate': self._stats.success_rate,
            'reliability': self._stats.reliability,
            'stability_score': self._stats.stability_score,
            'uptime_human': self._stats.uptime_human,
            'data_points_processed': self._stats.data_points_processed,
            'anomalies_detected': self._stats.anomalies_detected,
            'alerts_triggered': self._stats.alerts_triggered,
            'validation_errors': self._stats.validation_errors,
            'avg_processing_time': self._stats.avg_processing_time
        }
        
        # 如果启用了请求追踪，添加请求级别指标
        if self._enable_request_tracking:
            summary.update({
                'requests_total': self._stats.requests_total,
                'requests_failed': self._stats.requests_failed,
                'cache_hits': self._stats.cache_hits,
                'cache_misses': self._stats.cache_misses,
                'cache_hit_rate': self._stats.cache_hit_rate,
                'avg_response_time': self._stats.avg_response_time,
                'source_usage': self._stats.source_usage,
                'error_counts': self._stats.error_counts
            })
        
        return summary