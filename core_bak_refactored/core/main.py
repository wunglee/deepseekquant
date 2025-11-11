"""
DeepSeekQuant 系统入口
"""

import logging
from .system_config import SystemConfig
from .system_core import DeepSeekQuantSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepSeekQuant")


def main():
    """主函数"""
    # 加载配置
    config = SystemConfig().config

    # 创建系统实例
    system = DeepSeekQuantSystem(config)

    # 启动系统
    try:
        system.start()
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        system.stop()


if __name__ == "__main__":
    main()
