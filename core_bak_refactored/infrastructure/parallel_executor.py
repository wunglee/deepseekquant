"""
并行计算执行器 - Infrastructure层技术实现（迁移至 core_bak_refactored/infrastructure/）
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Callable, Any, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)

try:
    from common import RISK_MODEL_CONFIG as _RMC
except Exception:
    _RMC = {'parallel': {'min_tasks_for_parallel': 10, 'dynamic_chunking': True, 'memory_threshold_gb': 0.8}}

_DEFAULT_PARALLEL_CFG = _RMC.get('parallel', {})
logger = logging.getLogger(__name__)

try:
    from common import RISK_MODEL_CONFIG as _RMC
except Exception:
    _RMC = {'parallel': {'min_tasks_for_parallel': 10, 'dynamic_chunking': True, 'memory_threshold_gb': 0.8}}

_DEFAULT_PARALLEL_CFG = _RMC.get('parallel', {})


@dataclass
class ParallelConfig:
    """并行计算配置"""
    # CPU密集型配置
    max_workers_cpu: int = field(default_factory=lambda: min(mp.cpu_count(), 8))
    min_items_for_parallel_cpu: int = _DEFAULT_PARALLEL_CFG.get('min_tasks_for_parallel', 10)  # 最小任务数
    
    # I/O密集型配置
    max_workers_io: int = field(default_factory=lambda: mp.cpu_count() * 2)
    min_items_for_parallel_io: int = 5
    
    # 通用配置
    timeout_seconds: int = 300  # 5分钟超时
    chunk_size: Optional[int] = None  # 自动计算
    enable_monitoring: bool = True
    
    # P1-1: 动态分块参数
    enable_dynamic_chunking: bool = True  # 启用动态分块
    memory_threshold_gb: float = float(_DEFAULT_PARALLEL_CFG.get('memory_threshold_gb', 0.8))  # 内存阈值（占可用内存）


@dataclass
class ParallelMetrics:
    """并行计算性能指标"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_time_seconds: float = 0.0
    avg_task_time_seconds: float = 0.0
    speedup_ratio: float = 1.0  # 相对串行的加速比
    
    # P1-2: 性能监控增强
    peak_memory_mb: float = 0.0  # 峰值内存
    cpu_utilization_pct: float = 0.0  # CPU利用率
    task_distribution: Dict[str, int] = field(default_factory=dict)  # 任务分布
    
    # 评审建议补充指标
    task_times: List[float] = field(default_factory=list)  # 每批次平均任务耗时
    serialization_overhead: List[float] = field(default_factory=list)  # 序列化开销估计（占比）
    gil_contention: float = 0.0  # GIL争用估计（仅线程池场景）
    
    def to_dict(self) -> Dict:
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'total_time_seconds': round(self.total_time_seconds, 3),
            'avg_task_time_seconds': round(self.avg_task_time_seconds, 3),
            'speedup_ratio': round(self.speedup_ratio, 2),
            'peak_memory_mb': round(self.peak_memory_mb, 2),
            'cpu_utilization_pct': round(self.cpu_utilization_pct, 1),
            'task_distribution': self.task_distribution,
            'task_times': [round(t, 4) for t in self.task_times],
            'serialization_overhead': [round(s, 3) for s in self.serialization_overhead],
            'gil_contention': round(self.gil_contention, 3)
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
            data_size_mb = self._estimate_data_size_mb(items)
            if self.config.enable_dynamic_chunking:
                chunk_size = self._calculate_optimal_chunk_size(n_items, data_size_mb)
                logger.info(f"P1-1优化: 动态chunk_size={chunk_size} (data_size≈{data_size_mb:.1f}MB)")
            else:
                chunk_size = max(1, n_items // max(1, self.config.max_workers_cpu * 4))
        
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
                
                # 估算序列化开销占比（采样法）
                try:
                    import pickle
                    sample_items = items[:min(3, n_items)]
                    t0 = time.time()
                    for it in sample_items:
                        pickle.dumps(it)
                    serialize_time = time.time() - t0
                    if parallel_time > 0:
                        self.metrics.serialization_overhead.append(round(serialize_time / parallel_time, 3))
                except Exception:
                    pass
                
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
                
                # 线程池场景的GIL争用近似（CPU占用比）
                try:
                    import psutil
                    self.metrics.gil_contention = round(psutil.cpu_percent(interval=0.0) / 100.0, 3)
                except Exception:
                    pass
                
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
        
        # P1-2: 性能监控增强
        if self.config.enable_monitoring:
            self._collect_system_metrics()
        
        # 批次级任务耗时记录（专家建议）
        if completed > 0:
            self.metrics.task_times.append(elapsed / completed)
    
    def _collect_system_metrics(self):
        """P1-2: 收集系统性能指标"""
        try:
            import psutil
            
            # 内存使用
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
            
            # CPU利用率
            self.metrics.cpu_utilization_pct = psutil.cpu_percent(interval=0.1)
            
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"系统指标收集失败: {e}")
    
    def _calculate_optimal_chunk_size(
        self,
        n_tasks: int,
        data_size_mb: float = 10.0
    ) -> int:
        """
        P1-1: 动态计算最优分块大小
        """
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = self.config.max_workers_cpu
            max_concurrent_tasks = int(
                available_memory_gb * self.config.memory_threshold_gb * 1024 / max(1.0, data_size_mb)
            )
            memory_based_chunk = max(1, max_concurrent_tasks // max(1, cpu_count))
            cpu_based_chunk = max(1, n_tasks // max(1, cpu_count * 3))
            optimal_chunk = min(memory_based_chunk, cpu_based_chunk)
            return max(1, min(optimal_chunk, 20))
        except ImportError:
            return max(1, n_tasks // max(1, self.config.max_workers_cpu * 4))
        except Exception:
            return max(1, n_tasks // max(1, self.config.max_workers_cpu * 4))

    def _estimate_data_size_mb(self, items: List[Any]) -> float:
        """专家建议：动态估计任务数据大小（采样法）"""
        if not items:
            return 10.0
        import pickle
        sample_items = items[:min(5, len(items))]
        total_size = 0.0
        for item in sample_items:
            try:
                if hasattr(item, 'memory_usage'):
                    total_size += float(item.memory_usage(deep=True)) / 1e6
                else:
                    total_size += float(len(pickle.dumps(item))) / 1e6
            except Exception:
                total_size += 10.0
        avg = total_size / len(sample_items)
        return max(1.0, avg)

    def get_metrics(self) -> Dict:
        """获取当前并行执行器的性能指标（字典）"""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """重置性能指标"""
        self.metrics = ParallelMetrics()

_global_executor: Optional[ParallelExecutor] = None

def get_parallel_executor(config: Optional[ParallelConfig] = None) -> ParallelExecutor:
    """获取全局并行执行器单例"""
    global _global_executor
    if _global_executor is None:
        _global_executor = ParallelExecutor(config)
    elif config is not None:
        # 如果传入新配置，则更新现有单例的配置
        _global_executor.config = config
    return _global_executor
