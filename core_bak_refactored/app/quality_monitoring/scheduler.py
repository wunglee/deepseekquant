"""数据质量监控调度器

[应用层] 定时任务调度
职责：
- 定时执行数据质量检查
- 管理监控循环生命周期
- 支持多种调度策略（APScheduler/Celery）

设计原则：
- 应用层责任：定时调度属于应用层关注点
- 职责分离：调度器仅负责调度，具体逻辑由QualityMonitoringService提供
- 可配置：支持自定义检查间隔和调度策略

使用示例：
    # 方案1：使用APScheduler（轻量级，适合单进程）
    scheduler = MonitoringScheduler(monitoring_service, strategy='apscheduler')
    scheduler.start()
    
    # 方案2：使用Celery（分布式，适合多进程）
    scheduler = MonitoringScheduler(monitoring_service, strategy='celery')
    scheduler.start()
"""

import logging
import threading
import time
from enum import Enum
from typing import Optional, Dict, Any

import pandas as pd

logger = logging.getLogger('DeepSeekQuant.Scheduler')


class ScheduleStrategy(str, Enum):
    """调度策略"""
    THREAD = 'thread'          # 简单线程（测试/开发）
    APSCHEDULER = 'apscheduler'  # APScheduler（生产推荐）
    CELERY = 'celery'          # Celery（分布式）


class MonitoringScheduler:
    """
    数据质量监控调度器
    
    功能：
    1. 定时执行数据质量检查
    2. 支持多种调度策略
    3. 管理调度器生命周期
    4. 错误处理和重试
    
    使用场景：
    - strategy='thread': 简单场景，单进程应用
    - strategy='apscheduler': 生产环境，需要灵活调度
    - strategy='celery': 分布式环境，多worker
    """
    
    def __init__(self,
                 monitoring_service,
                 strategy: str = 'apscheduler',
                 config: Optional[Dict[str, Any]] = None,
                 api_service=None):
        """
        初始化调度器
        
        Args:
            monitoring_service: QualityMonitoringService实例
            strategy: 调度策略（'thread'/'apscheduler'/'celery'）
            config: 调度配置
                - check_interval: 检查间隔（秒，默认300=5分钟）
                - max_retries: 最大重试次数（默认3）
                - retry_delay: 重试延迟（秒，默认60）
            api_service: API服务实例（用于广播数据更新）
        """
        self.monitoring_service = monitoring_service
        self.strategy = ScheduleStrategy(strategy)
        self.config = config or {}
        self.api_service = api_service  # API服务实例
        
        # 配置参数
        self.check_interval = self.config.get('check_interval', 300)  # 5分钟
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 60)
        
        # 状态管理
        self._running = False
        self._scheduler = None
        self._monitor_thread = None
        
        logger.info(f"MonitoringScheduler initialized with strategy={strategy}, interval={self.check_interval}s")
    
    def start(self):
        """启动调度器"""
        if self._running:
            logger.warning("调度器已在运行")
            return
        
        logger.info(f"启动监控调度器（策略：{self.strategy.value}）")
        self._running = True
        
        if self.strategy == ScheduleStrategy.THREAD:
            self._start_thread_scheduler()
        elif self.strategy == ScheduleStrategy.APSCHEDULER:
            self._start_apscheduler()
        elif self.strategy == ScheduleStrategy.CELERY:
            self._start_celery_scheduler()
    
    def stop(self):
        """停止调度器"""
        if not self._running:
            logger.warning("调度器未运行")
            return
        
        logger.info("停止监控调度器")
        self._running = False
        
        if self.strategy == ScheduleStrategy.APSCHEDULER and self._scheduler:
            self._scheduler.shutdown()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        
        logger.info("调度器已停止")
    
    def _start_thread_scheduler(self):
        """启动简单线程调度器（用于测试和简单场景）"""
        def monitoring_worker():
            logger.info("监控线程启动")
            while self._running:
                try:
                    self._execute_monitoring_cycle()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"监控循环异常: {e}", exc_info=True)
                    time.sleep(self.retry_delay)
            logger.info("监控线程退出")
        
        self._monitor_thread = threading.Thread(target=monitoring_worker, daemon=True, name='QualityMonitor')
        self._monitor_thread.start()
        logger.info("简单线程调度器已启动")
    
    def _start_apscheduler(self):
        """启动APScheduler（生产环境推荐）"""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.interval import IntervalTrigger
            
            self._scheduler = BackgroundScheduler()
            self._scheduler.add_job(
                func=self._execute_monitoring_cycle,
                trigger=IntervalTrigger(seconds=self.check_interval),
                id='quality_monitoring',
                name='数据质量监控',
                replace_existing=True,
                max_instances=1  # 确保同时只有一个实例运行
            )
            self._scheduler.start()
            logger.info(f"APScheduler已启动（间隔：{self.check_interval}秒）")
        
        except ImportError:
            logger.error("APScheduler未安装，请运行: pip install apscheduler")
            logger.info("回退到简单线程调度器")
            self._start_thread_scheduler()
    
    def _start_celery_scheduler(self):
        """启动Celery调度器（分布式环境）"""
        try:
            from celery import Celery
            from celery.schedules import crontab
            
            # 注意：Celery需要独立的配置和启动，这里仅注册任务
            logger.warning("Celery调度需要独立启动celery beat")
            logger.info("使用命令: celery -A app.data.scheduler beat --loglevel=info")
            
            # 创建任务（需要在Celery app配置中注册）
            # 实际项目中应该在专门的celery_config.py中配置
            logger.info("Celery任务已注册，等待celery beat调度")
        
        except ImportError:
            logger.error("Celery未安装，请运行: pip install celery")
            logger.info("回退到APScheduler")
            self._start_apscheduler()
    
    def _execute_monitoring_cycle(self):
        """
        执行一次监控循环
        
        功能：
        1. 调用QualityMonitoringService.run_check_cycle执行完整检查
        2. 更新监控周期计数
        3. 记录执行日志和统计
        """
        cycle_start = pd.Timestamp.now()
        logger.info("开始监控循环")
        
        try:
            # 调用监控服务执行完整检查周期
            summary = self.monitoring_service.run_check_cycle()
            # run_check_cycle 已经内部更新了 monitoring_cycles 计数
            
            # 记录日志
            cycle_duration = (pd.Timestamp.now() - cycle_start).total_seconds()
            logger.info(
                f"监控循环完成，耗时{cycle_duration:.2f}秒 | "
                f"质量得分: {summary.get('quality_score', 0):.2%} | "
                f"数据点: {summary.get('data_points_checked', 0)} | "
                f"异常: {summary.get('anomalies_detected', 0)} | "
                f"告警: {summary.get('alerts_triggered', 0)}"
            )
            
            # 广播数据更新到Socket.IO客户端
            if self.api_service and hasattr(self.api_service, 'broadcast_quality_update'):
                try:
                    self.api_service.broadcast_quality_update(summary)
                except Exception as broadcast_error:
                    logger.error(f"Socket.IO广播失败: {broadcast_error}")
        
        except Exception as e:
            logger.error(f"监控循环失败: {e}", exc_info=True)
            self.monitoring_service._performance_stats['validation_errors'] += 1
            
            # 重试逻辑
            for retry in range(self.max_retries):
                logger.info(f"重试监控循环（第{retry + 1}/{self.max_retries}次）")
                time.sleep(self.retry_delay)
                try:
                    # 简化的重试逻辑
                    summary = self.monitoring_service.run_check_cycle()
                    logger.info("重试成功")
                    break
                except Exception as retry_error:
                    logger.error(f"重试失败: {retry_error}")
                    if retry == self.max_retries - 1:
                        logger.critical("监控循环多次重试失败，请检查系统")
    
    def execute_now(self):
        """立即执行一次监控检查（手动触发）"""
        logger.info("手动触发监控检查")
        self._execute_monitoring_cycle()
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            'running': self._running,
            'strategy': self.strategy.value,
            'check_interval': self.check_interval,
            'next_run': self._get_next_run_time()
        }
    
    def _get_next_run_time(self) -> Optional[str]:
        """获取下次运行时间"""
        if not self._running:
            return None
        
        if self.strategy == ScheduleStrategy.APSCHEDULER and self._scheduler:
            job = self._scheduler.get_job('quality_monitoring')
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
        
        return "Unknown"


# Celery任务定义（如果使用Celery策略）
# 需要在独立的celery_config.py中配置
try:
    from celery import Celery
    
    # 示例Celery app配置
    celery_app = Celery('data_quality_monitor')
    
    @celery_app.task(name='quality_monitoring_task')
    def celery_monitoring_task():
        """Celery定时任务：数据质量监控"""
        # 注意：这里需要从配置或上下文获取monitoring_service实例
        # 实际实现需要依赖注入或单例模式
        logger.info("Celery任务执行中")
        # scheduler._execute_monitoring_cycle()

except ImportError:
    logger.debug("Celery未安装，跳过Celery任务定义")
