"""数据质量监控应用启动示例

展示如何集成监控调度器到应用中

使用方式：
    # 开发环境（简单线程）
    python -m core_bak_refactored.app.data.app_example --strategy thread
    
    # 生产环境（APScheduler）
    python -m core_bak_refactored.app.data.app_example --strategy apscheduler
    
    # 分布式环境（Celery）
    python -m core_bak_refactored.app.data.app_example --strategy celery
"""

import argparse
import logging
import signal
import sys
from datetime import datetime

from core_bak_refactored.app.data.monitoring_service import QualityMonitoringService
from core_bak_refactored.app.data.scheduler import MonitoringScheduler
from core_bak_refactored.app.data.api_service import DataQualityAPIService
from core_bak_refactored.core.monitoring.alert_manager import AlertConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DeepSeekQuant.App')


class DataQualityApplication:
    """数据质量监控应用
    
    集成组件：
    - QualityMonitoringService: 监控服务核心
    - MonitoringScheduler: 定时任务调度
    - DataQualityAPIService: REST API服务
    """
    
    def __init__(self, strategy: str = 'apscheduler', api_host: str = '0.0.0.0', api_port: int = 5001):
        """初始化应用
        
        Args:
            strategy: 调度策略（thread/apscheduler/celery）
            api_host: API 服务监听地址（默认：0.0.0.0）
            api_port: API 服务监听端口（默认：5001）
        
        Note:
            数据源类型通过配置文件 config/dev/data.yml 中的 primary_source 控制
            - mock: 使用模拟数据（不调用外部API）
            - yahoo: 使用Yahoo Finance真实数据
            - tushare: 使用Tushare数据
        """
        self.api_host = api_host
        self.api_port = api_port
        logger.info("初始化数据质量监控应用")
        
        # 1. 创建告警配置
        alert_config = AlertConfig(
            wechat_webhook_url=None,  # 生产环境需要配置
            dedup_window_minutes=10,
            max_alerts_per_hour=50
        )
        
        # 2. 创建监控服务（数据源由配置文件控制）
        self.monitoring_service = QualityMonitoringService(alert_config=alert_config)
        logger.info("监控服务已创建")
        
        # 3. 创建调度器
        scheduler_config = {
            'check_interval': 300,  # 5分钟检查一次
            'max_retries': 3,
            'retry_delay': 60
        }
        
        # 4. 创建API服务（先创建，以便调度器可以引用）
        self.api_service = DataQualityAPIService(self.monitoring_service, scheduler=None)
        logger.info("API服务已创建")
        
        # 5. 创建调度器（传入api_service以支持Socket.IO实时推送）
        self.scheduler = MonitoringScheduler(
            self.monitoring_service,
            strategy=strategy,
            config=scheduler_config,
            api_service=self.api_service  # 传递API服务实例
        )
        logger.info(f"调度器已创建（策略：{strategy}）")
        
        # 更新API服务的scheduler引用（用于手动触发）
        self.api_service.scheduler = self.scheduler
        
        # 5. 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def start(self, enable_api: bool = True):
        """启动应用
        
        Args:
            enable_api: 是否启动API服务（默认True）
        """
        logger.info("启动应用...")
        
        # 1. 启动调度器
        self.scheduler.start()
        logger.info("✅ 调度器已启动")
        
        # 2. 启动API服务（可选）
        if enable_api:
            logger.info(f"启动API服务在 http://{self.api_host}:{self.api_port}")
            # 使用socketio.run支持WebSocket
            self.api_service.socketio.run(
                self.api_service.app,
                host=self.api_host,
                port=self.api_port,
                debug=False,
                use_reloader=False,  # 重要：避免重复启动调度器
                allow_unsafe_werkzeug=True  # 开发环境允许
            )
        else:
            logger.info("API服务未启动（仅运行调度器）")
            # 保持主线程运行
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("收到中断信号")
    
    def stop(self):
        """停止应用"""
        logger.info("停止应用...")
        self.scheduler.stop()
        logger.info("✅ 应用已停止")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"收到信号 {signum}，准备退出...")
        self.stop()
        sys.exit(0)


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description='数据质量监控应用')
    parser.add_argument(
        '--strategy',
        choices=['thread', 'apscheduler', 'celery'],
        default='apscheduler',
        help='调度策略'
    )
    parser.add_argument(
        '--no-api',
        action='store_true',
        help='不启动API服务（仅运行调度器）'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='仅执行一次检查后退出'
    )
    
    args = parser.parse_args()
    
    # 创建应用（数据源由配置文件 config/dev/data.yml 控制）
    app = DataQualityApplication(strategy=args.strategy)
    
    if args.once:
        # 仅执行一次
        logger.info("执行单次监控检查")
        app.scheduler.execute_now()
        logger.info("检查完成，退出")
        return
    
    # 启动应用
    try:
        app.start(enable_api=not args.no_api)
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"应用异常: {e}", exc_info=True)
    finally:
        app.stop()


if __name__ == '__main__':
    main()
