"""
任务管理组件 - 负责任务调度和管理
"""

import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Callable, Optional
from .interfaces import ITaskManager
from concurrent.futures import ThreadPoolExecutor, Future
import threading


@dataclass
class TaskManagerConfig:
    """任务管理配置"""
    max_threads: int = 8
    max_queue_size: int = 10000
    processing_timeout: int = 30


@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    start_time: str
    args: Any
    kwargs: Any
    end_time: Optional[str] = None
    processing_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class TaskManager(ITaskManager):
    """任务管理器"""

    def __init__(self, config: TaskManagerConfig, processor_name: str):
        self.config = config
        self.processor_name = processor_name
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.task_queue: List[str] = []
        self.lock = threading.RLock()

    def initialize(self):
        """初始化任务管理器"""
        with self.lock:
            if self.thread_pool is None:
                self.thread_pool = ThreadPoolExecutor(
                    max_workers=self.config.max_threads,
                    thread_name_prefix=f"{self.processor_name}_Worker"
                )

    def generate_task_id(self) -> str:
        """生成任务ID"""
        return f"task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    def record_task_start(self, task_id: str, args: tuple, kwargs: dict):
        """记录任务开始"""
        with self.lock:
            task_info = TaskInfo(
                task_id=task_id,
                start_time=datetime.now().isoformat(),
                args=args,
                kwargs=kwargs
            )
            self.active_tasks[task_id] = task_info
            self.task_queue.append(task_id)

    def record_task_success(self, task_id: str, result: Any, processing_time: float):
        """记录任务成功"""
        with self.lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.end_time = datetime.now().isoformat()
                task_info.processing_time = processing_time
                task_info.result = result

    def record_task_failure(self, task_id: str, error: str, processing_time: float):
        """记录任务失败"""
        with self.lock:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                task_info.end_time = datetime.now().isoformat()
                task_info.processing_time = processing_time
                task_info.error = error

    def record_task_end(self, task_id: str):
        """记录任务结束（清理）"""
        with self.lock:
            if task_id in self.active_tasks:
                # 可选：将任务信息保存到历史记录
                del self.active_tasks[task_id]

            if task_id in self.task_queue:
                self.task_queue.remove(task_id)

    def submit_task(self, task_fn: Callable, *args, **kwargs) -> Future:
        """提交异步任务"""
        if self.thread_pool is None:
            raise RuntimeError("任务管理器未初始化")

        with self.lock:
            if len(self.task_queue) >= self.config.max_queue_size:
                raise RuntimeError(f"任务队列已满: {len(self.task_queue)}/{self.config.max_queue_size}")

            return self.thread_pool.submit(task_fn, *args, **kwargs)

    def batch_process(self, items: List[Any], batch_size: int,
                      process_fn: Callable, timeout: int) -> List[Any]:
        """批量处理项目"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]

            try:
                # 使用线程池并行处理批次
                futures = []
                for item in batch:
                    future = self.submit_task(process_fn, item)
                    futures.append(future)

                # 收集结果
                for future in futures:
                    try:
                        result = future.result(timeout=timeout)
                        results.append(result)
                    except Exception as e:
                        results.append({'error': str(e)})

            except Exception as e:
                # 为失败的批次添加错误结果
                results.extend([{'error': str(e)}] * len(batch))

        return results

    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        with self.lock:
            return {
                'active_tasks': len(self.active_tasks),
                'queue_size': len(self.task_queue),
                'max_queue_size': self.config.max_queue_size,
                'thread_pool_active': self.thread_pool is not None
            }

    def emergency_stop(self):
        """紧急停止"""
        with self.lock:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
                self.thread_pool = None

            self.active_tasks.clear()
            self.task_queue.clear()

    def cleanup(self):
        """清理资源"""
        with self.lock:
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None

            self.active_tasks.clear()
            self.task_queue.clear()

    def update_config(self, new_config: TaskManagerConfig):
        """更新配置"""
        with self.lock:
            self.config = new_config

            # 如果线程池已存在且线程数变更，需要重新创建
            if (self.thread_pool and
                    self.thread_pool._max_workers != new_config.max_threads):
                self.cleanup()
                self.initialize()