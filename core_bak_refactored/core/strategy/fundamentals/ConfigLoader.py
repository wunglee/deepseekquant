import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger('DeepSeekQuant.core.strategy.fundamentals.StrategyConfigLoader')


class StrategyConfigLoader:
    _instance = None
    _lock = None
    _observer = None  # 全局watchdog observer
    environment: str
    loaded = {}
    @staticmethod
    def _get_environment() -> str:
        """获取当前环境配置"""
        return os.getenv('DEEPSEEK_ENV', 'dev')

    def __new__(cls, config_file: Optional[str] = None, environment: Optional[str] = None):
        """单例模式实现"""
        if cls._instance is None:
            if cls._lock is None:
                import threading
                cls._lock = threading.Lock()

            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StrategyConfigLoader, cls).__new__(cls)
                    cls._instance._initialized = False

        return cls._instance

    def __init__(self, environment: Optional[str] = None):
        # 避免重复初始化，但允许environment参数变更时重新加载
        requested_env = environment or StrategyConfigLoader._get_environment()
        if self._initialized:
            # 如果environment参数与当前环境不同，重新加载配置
            if requested_env != self.environment:
                logger.info(f"检测到环境变更: {self.environment} -> {requested_env}，重新加载配置")
                self.environment = requested_env
                self.load_all_strategies()
            return
        self.environment = requested_env
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'config')
        env_dir = os.path.join(base_dir, self.environment, 'fundamentals')
        self.config_dir = Path(env_dir) if os.path.exists(env_dir) else Path(base_dir)
        self.load_all_strategies()

        # 启动热加载监听（仅在非测试环境且尚未启动）
        if self.environment != 'test' and StrategyConfigLoader._observer is None:
            self._start_hot_reload_watcher()

        self._initialized = True

    def load_all_strategies(self):
        """加载配置（优先从 core_bak_refactored/config/{env}/fundamentals/*.json 读取）"""
        try:
            import glob

            # 自动扫描所有 .json 文件
            json_files = glob.glob(os.path.join(self.config_dir, '*.json'))
            
            loaded = {}
            for json_path in json_files:
                strategy_name = os.path.splitext(os.path.basename(json_path))[0]
                try:
                    loaded[strategy_name] = self.load_strategy(strategy_name) or {}
                except Exception as e:
                    logger.warning(f"跳过无效配置文件 {json_path}: {e}")

            if loaded:
                self._config = loaded
                logger.info(f"从 JSON 加载配置 [{self.environment}]: {sorted(loaded.keys())}")
                return
        except Exception as e:
            logger.warning(f"JSON配置加载失败: {e}")
            raise RuntimeError(
                f"配置文件加载失败，请检查配置文件是否存在: "
            )

    def load_strategy(self, strategy_name: str = None) -> Dict[str, Any]:
        """
        加载指定策略配置文件

        Args:
            strategy_name: 策略文件名（不含.json后缀）

        Returns:
            策略配置字典（标准Python数据结构）

        Raises:
            FileNotFoundError: 策略文件不存在
            json.JSONDecodeError: JSON解析失败
        """
        config_file = os.path.join(self.config_dir, f"{strategy_name}.json")

        if not os.path.exists(config_file):
            available_strategies = self.list_available_strategies()
            raise FileNotFoundError(
                f"策略文件不存在: {config_file}\n"
                f"可用策略: {available_strategies if available_strategies else '（目录为空）'}"
            )

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 添加元数据，便于追踪
            config["_meta"] = {
                "strategy_name": strategy_name,
                "env": self.environment,
                "config_dir": str(self.config_dir),
                "file_path": str(config_file),
                "loaded_at": os.path.getmtime(config_file)
            }

            print(f"✓ 成功加载策略: {strategy_name}")
            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON解析失败 {config_file}: {e}")
        except Exception as e:
            raise RuntimeError(f"加载策略失败 {strategy_name}: {e}")

    def list_available_strategies(self) -> List[str]:
        """获取可用策略列表"""
        return [f.stem for f in list(self.config_dir.glob("*.json"))]

    def get_strategy_path(self, strategy_name: str) -> Path:
        """获取策略文件完整路径"""
        return self.config_dir / f"{strategy_name}.json"

    def get_config_summary(self) -> Dict[str, Any]:
        """获取当前配置环境摘要"""
        return {
            "environment": self.environment,
            "config_root": str(self.config_dir),
            "available_strategies": self.list_available_strategies(),
            "strategy_count": len(self.list_available_strategies())
        }
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """获取指定策略配置"""
        return self.loaded.get(strategy_name, {})

    def _start_hot_reload_watcher(self):
        """启动热加载监听器（全局单例）"""
        try:
            import threading
            import time
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class ConfigReloadHandler(FileSystemEventHandler):
                def __init__(self, config_manager):
                    self.config_manager = config_manager

                def on_modified(self, event):
                    if event.src_path.endswith(('.yml', '.yaml')):
                        logger.info(f"检测到配置文件变更: {event.src_path}")
                        time.sleep(0.1)
                        self.config_manager._load_config()

            def watch_thread():
                # core_bak_refactored/core/share/config_manager.py -> core_bak_refactored/config
                base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
                env_dir = os.path.join(base_dir, self.environment)
                watch_path = env_dir if os.path.exists(env_dir) else base_dir

                event_handler = ConfigReloadHandler(self)
                StrategyConfigLoader._observer = Observer()
                StrategyConfigLoader._observer.schedule(event_handler, watch_path, recursive=False)
                StrategyConfigLoader._observer.start()
                logger.info(f"启动配置热加载监听: {watch_path}")

                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    if StrategyConfigLoader._observer:
                        StrategyConfigLoader._observer.stop()
                if StrategyConfigLoader._observer:
                    StrategyConfigLoader._observer.join()

            watcher = threading.Thread(target=watch_thread, daemon=True, name='ConfigHotReload')
            watcher.start()
        except ImportError:
            logger.info("未安装watchdog，跳过配置热加载")
        except Exception as e:
            logger.warning(f"配置热加载启动失败: {e}")

# ----------------------------------------------------------------------
# 与Pandas集成（可选）
# ----------------------------------------------------------------------

def load_strategy_to_dataframe(strategy_name: str, env: Optional[str] = None, config_dir: Optional[Path] = None):
    """
    将策略配置转换为pandas DataFrame（如果安装了pandas）

    Args:
        strategy_name: 策略名称
        env: 环境名称（可选）
        config_dir: 项目根目录（可选）

    Returns:
        DataFrame或None（如果pandas未安装）
    """
    try:
        import pandas as pd
    except ImportError:
        print("警告: pandas未安装，返回字典格式")
        return None

    if env:
        os.environ["MUNGER_ENV"] = env

    loader = StrategyConfigLoader()
    config = loader.load_strategy(strategy_name)

    # 展平嵌套字典为表格
    records = []
    for category, metrics in config.items():
        if category.startswith("_"):  # 跳过元数据
            continue
        if isinstance(metrics, dict):
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    records.append({
                        "category": category,
                        "metric": metric_name,
                        **values  # 展开max, ideal, danger, weight等
                    })

    return pd.DataFrame(records)



# ----------------------------------------------------------------------
# 使用示例
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("芒格策略配置加载器 - 使用示例")
    print("=" * 70)

    # 示例1: 自动检测项目根目录
    print("\n示例1: 自动检测项目根目录")
    print("-" * 70)
    try:
        loader = StrategyConfigLoader()
        print(f"\n配置摘要: {loader.get_config_summary()}")
    except FileNotFoundError as e:
        print(f"初始化失败（预期，如果目录不存在）: {e}")
        print("创建示例目录结构用于演示...")