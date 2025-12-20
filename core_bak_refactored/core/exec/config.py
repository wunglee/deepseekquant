"""
交易配置管理器

职责：管理交易相关的配置参数（从 trade.yml 读取）
定位：交易执行模块配置管理
"""

from typing import Dict, Any
import logging
import yaml
import os

logger = logging.getLogger(__name__)


class TradeConfig:
    """交易配置管理器（从 trade.yml 加载）
    
    职责：
    - 从 core_bak_refactored/config/{env}/trade.yml 读取交易配置
    - 提供交易成本参数查询接口
    """
    
    def __init__(self, environment: str = None):
        """
        初始化交易配置管理器
        
        Args:
            environment: 环境名称（dev/test/prod），默认从环境变量读取
        """
        self.environment = environment or os.getenv('DEEPSEEK_ENV', 'dev')
        self._config_dir = self._get_config_dir()
        
        # 加载交易配置文件
        self._trade_config = self._load_yaml_config('trade.yml')
        
        # 解析交易配置
        self.cost = self._trade_config.get('cost', {})
    
    def _get_config_dir(self) -> str:
        """获取配置文件目录"""
        # core/exec/config.py -> core_bak_refactored
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_dir = os.path.join(current_dir, 'config', self.environment)
        if not os.path.exists(config_dir):
            config_dir = os.path.join(current_dir, 'config')
        return config_dir
    
    def _load_yaml_config(self, filename: str) -> Dict[str, Any]:
        """加载YAML配置文件
        
        Args:
            filename: 配置文件名
        
        Returns:
            配置字典，如果文件不存在则返回空字典
        """
        try:
            config_path = os.path.join(self._config_dir, filename)
            
            if not os.path.exists(config_path):
                logger.warning(f"未找到配置文件: {config_path}，使用空配置")
                return {}
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            logger.info(f"✅ {filename} 加载成功: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"加载{filename}失败: {e}")
            return {}
    
    def get_trade_cost(self, market_code: str) -> Dict[str, Any]:
        """获取指定市场的交易成本参数
        
        Args:
            market_code: 市场代码（CN, US, HK, JP, EU, SG）
        
        Returns:
            交易成本参数字典
        """
        return self.cost.get(market_code, {})
