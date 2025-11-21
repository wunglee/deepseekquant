"""
并行计算执行器 - Infrastructure层技术实现

职责：
1. 多进程/多线程并行计算框架
2. 动态任务分配和负载均衡
3. 资源监控和异常处理
4. 性能统计和监控

设计原则：
- CPU密集型: multiprocessing.Pool
- I/O密集型: concurrent.futures.ThreadPoolExecutor
- 避免小任务并行（开销>收益）

架构定位：技术基础设施层
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """并行计算配置"""
    # CPU密集型配置
    max_workers_cpu: int = field(default_factory=lambda: min(mp.cpu_count(), 8))
    min_items_for_parallel_cpu: int = 10  # 最小任务数
    
    # I/O密集型配置
    max_workers_io: int = field(default_factory=lambda: mp.cpu_count() * 2)
    min_items_for_parallel_io: int = 5
    
    # 通用配置
    timeout_seconds: int = 300  # 5分钟超时
    chunk_size: Optional[int] = None  # 自动计算
    enable_monitoring: bool = True


@dataclass
class ParallelMetrics:
    """并行计算性能指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_time_seconds: float = 0.0
    avg_task_time_seconds: float = 0.0
    speedup_ratio: float = 1.0  # 相对串行的加速比
    
    def to_dict(self) -> Dict:
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_time_seconds': round(self.total_time_seconds, 3),
            'avg_task_time_seconds': round(self.avg_task_time_seconds, 3),
            'speedup_ratio': round(self.speedup_ratio, 2)
        }


class ParallelExecutor:
    """
    并行计算执行器
    
    支持CPU密集型和I/O密集型任务的并行执行
    """
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        初始化并行执行器
        
        Parameters:
        config: 并行配置，默认使用标准配置
        """
        self.config = config or ParallelConfig()
        self.metrics = ParallelMetrics()
        
        logger.info(
            f"并行执行器初始化: CPU workers={self.config.max_workers_cpu}, "
            f"IO workers={self.config.max_workers_io}"
        )
    
    def should_parallelize(
        self,
        n_tasks: int,
        task_type: str = 'cpu'
    ) -> Tuple[bool, str]:
        """
        判断是否应该并行执行
        
        Parameters:
        n_tasks: 任务数量
        task_type: 'cpu' 或 'io'
        
        Returns:
        (should_parallel, reason): 是否并行和原因
        """
        if task_type == 'cpu':
            min_tasks = self.config.min_items_for_parallel_cpu
        else:
            min_tasks = self.config.min_items_for_parallel_io
        
        if n_tasks < min_tasks:
            return False, f"任务数{n_tasks}小于最小并行阈值{min_tasks}"
        
        return True, "满足并行条件"
    
    def map_cpu_intensive(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        CPU密集型任务并行映射
        
        Parameters:
        func: 要并行执行的函数
        items: 输入项列表
        chunk_size: 分块大小，None自动计算
        
        Returns:
        results: 结果列表
        """
        start_time = time.time()
        n_items = len(items)
        
        # 判断是否并行
        should_parallel, reason = self.should_parallelize(n_items, 'cpu')
        
        if not should_parallel:
            logger.info(f"使用串行执行: {reason}")
            results = [func(item) for item in items]
            serial_time = time.time() - start_time
            
            self._update_metrics(n_items, n_items, 0, serial_time, 1.0)
            return results
        
        # 并行执行
        logger.info(
            f"CPU密集型并行执行: {n_items}个任务, "
            f"{self.config.max_workers_cpu}个workers"
        )
        
        # 计算chunk_size
        if chunk_size is None:
            chunk_size = max(1, n_items // (self.config.max_workers_cpu * 4))
        
        try:
            with ProcessPoolExecutor(
                max_workers=self.config.max_workers_cpu
            ) as executor:
                # 使用map保证顺序
                results = list(executor.map(
                    func,
                    items,
                    chunksize=chunk_size,
                    timeout=self.config.timeout_seconds
                ))
                
                parallel_time = time.time() - start_time
                
                # 估算串行时间（用于计算加速比）
                if len(results) > 0:
                    estimated_serial_time = parallel_time * self.config.max_workers_cpu
                    speedup = estimated_serial_time / parallel_time if parallel_time > 0 else 1.0
                else:
                    speedup = 1.0
                
                self._update_metrics(n_items, len(results), 0, parallel_time, speedup)
                
                logger.info(
                    f"并行执行完成: {len(results)}/{n_items}成功, "
                    f"耗时{parallel_time:.2f}s, 加速比{speedup:.2f}x"
                )
                
                return results
                
        except Exception as e:
            logger.error(f"并行执行失败: {e}, 回退到串行")
            results = [func(item) for item in items]
            fallback_time = time.time() - start_time
            self._update_metrics(n_items, len(results), 0, fallback_time, 1.0)
            return results
    
    def map_io_intensive(
        self,
        func: Callable,
        items: List[Any]
    ) -> List[Any]:
        """
        I/O密集型任务并行映射
        
        Parameters:
        func: 要并行执行的函数
        items: 输入项列表
        
        Returns:
        results: 结果列表
        """
        start_time = time.time()
        n_items = len(items)
        
        # 判断是否并行
        should_parallel, reason = self.should_parallelize(n_items, 'io')
        
        if not should_parallel:
            logger.info(f"使用串行执行: {reason}")
            results = [func(item) for item in items]
            serial_time = time.time() - start_time
            self._update_metrics(n_items, n_items, 0, serial_time, 1.0)
            return results
        
        # 并行执行（使用线程池）
        logger.info(
            f"I/O密集型并行执行: {n_items}个任务, "
            f"{self.config.max_workers_io}个workers"
        )
        
        try:
            with ThreadPoolExecutor(
                max_workers=self.config.max_workers_io
            ) as executor:
                # 提交所有任务
                future_to_item = {
                    executor.submit(func, item): i
                    for i, item in enumerate(items)
                }
                
                # 按顺序收集结果
                results = [None] * n_items
                completed = 0
                failed = 0
                
                for future in as_completed(
                    future_to_item,
                    timeout=self.config.timeout_seconds
                ):
                    idx = future_to_item[future]
                    try:
                        results[idx] = future.result()
                        completed += 1
                    except Exception as e:
                        logger.error(f"任务{idx}失败: {e}")
                        results[idx] = None
                        failed += 1
                
                parallel_time = time.time() - start_time
                speedup = self.config.max_workers_io if completed > 0 else 1.0
                
                self._update_metrics(n_items, completed, failed, parallel_time, speedup)
                
                logger.info(
                    f"并行执行完成: {completed}/{n_items}成功, {failed}失败, "
                    f"耗时{parallel_time:.2f}s"
                )
                
                return results
                
        except Exception as e:
            logger.error(f"并行执行失败: {e}, 回退到串行")
            results = [func(item) for item in items]
            fallback_time = time.time() - start_time
            self._update_metrics(n_items, len(results), 0, fallback_time, 1.0)
            return results
    
    def map_with_progress(
        self,
        func: Callable,
        items: List[Any],
        task_type: str = 'cpu',
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        带进度回调的并行映射
        
        Parameters:
        func: 要并行执行的函数
        items: 输入项列表
        task_type: 'cpu' 或 'io'
        progress_callback: 进度回调函数 callback(completed, total)
        
        Returns:
        results: 结果列表
        """
        start_time = time.time()
        n_items = len(items)
        
        if task_type == 'cpu':
            max_workers = self.config.max_workers_cpu
            executor_class = ProcessPoolExecutor
        else:
            max_workers = self.config.max_workers_io
            executor_class = ThreadPoolExecutor
        
        logger.info(f"并行执行({task_type}): {n_items}个任务")
        
        try:
            with executor_class(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_item = {
                    executor.submit(func, item): i
                    for i, item in enumerate(items)
                }
                
                # 收集结果并更新进度
                results = [None] * n_items
                completed = 0
                
                for future in as_completed(future_to_item):
                    idx = future_to_item[future]
                    try:
                        results[idx] = future.result()
                        completed += 1
                        
                        # 进度回调
                        if progress_callback:
                            progress_callback(completed, n_items)
                            
                    except Exception as e:
                        logger.error(f"任务{idx}失败: {e}")
                        results[idx] = None
                        completed += 1
                
                elapsed = time.time() - start_time
                speedup = max_workers if completed > 0 else 1.0
                
                self._update_metrics(n_items, completed, n_items - completed, elapsed, speedup)
                
                return results
                
        except Exception as e:
            logger.error(f"并行执行失败: {e}")
            raise
    
    def _update_metrics(
        self,
        total: int,
        completed: int,
        failed: int,
        elapsed: float,
        speedup: float
    ):
        """更新性能指标"""
        self.metrics.total_tasks += total
        self.metrics.completed_tasks += completed
        self.metrics.failed_tasks += failed
        self.metrics.total_time_seconds += elapsed
        
        if self.metrics.completed_tasks > 0:
            self.metrics.avg_task_time_seconds = (
                self.metrics.total_time_seconds / self.metrics.completed_tasks
            )
        
        self.metrics.speedup_ratio = speedup
    
    def get_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            **self.metrics.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = ParallelMetrics()
        logger.info("并行执行器指标已重置")


# 全局并行执行器单例
_global_parallel_executor: Optional[ParallelExecutor] = None


def get_parallel_executor(config: Optional[ParallelConfig] = None) -> ParallelExecutor:
    """获取全局并行执行器实例（单例模式）"""
    global _global_parallel_executor
    
    if _global_parallel_executor is None:
        _global_parallel_executor = ParallelExecutor(config)
        logger.info("创建全局并行执行器实例")
    
    return _global_parallel_executor
