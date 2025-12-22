"""应用主入口 - 数据质量监控服务

使用方式：
    # 默认端口（5000）
    python -m core_bak_refactored.app.main
    
    # 指定端口
    python -m core_bak_refactored.app.main --port 5001
    
    # 指定调度策略
    python -m core_bak_refactored.app.main --strategy apscheduler --port 5001
"""

import argparse
import signal
import sys
import traceback
import faulthandler
from core_bak_refactored.app.quality_monitoring.app_example import DataQualityApplication

# 启用 faulthandler，在段错误时打印调用栈
faulthandler.enable()


def main():
    """主入口函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数据质量监控应用')
    parser.add_argument(
        '--strategy',
        choices=['thread', 'apscheduler', 'celery'],
        default='apscheduler',
        help='调度策略（默认：apscheduler）'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5001,
        help='API 服务端口（默认：5001）'
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='API 服务地址（默认：0.0.0.0）'
    )
    
    args = parser.parse_args()
    
    # 创建应用（数据源由配置文件 config/dev/data_provider.yml 控制）
    app = DataQualityApplication(
        strategy=args.strategy,
        api_host=args.host,
        api_port=args.port
    )
    
    # 注册信号处理（优雅停止）
    def signal_handler(sig, frame):
        print('\n收到停止信号，正在优雅关闭...')
        app.stop()
        sys.exit(0)
    
    def sigsegv_handler(sig, frame):
        print('\n🚨🚨🚨 检测到段错误 (SIGSEGV)!!!')
        print('调用栈信息：')
        traceback.print_stack(frame)
        print('\n这通常是 C 扩展库（pandas/numpy）的问题')
        sys.exit(139)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGSEGV, sigsegv_handler)  # 捕获段错误
    
    # 启动应用
    try:
        app.start()
    except KeyboardInterrupt:
        print('\n收到键盘中断，正在优雅关闭...')
        app.stop()
    except Exception as e:
        print(f'应用启动失败: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()