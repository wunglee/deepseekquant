"""
风险配置管理器

职责：管理风险相关的配置参数（从 risk.yml 读取）
定位：风险模块配置管理
"""

from typing import Dict, Any
import logging
import yaml
import os

logger = logging.getLogger(__name__)


class RiskConfig:
    """风险配置管理器（从 risk.yml 加载）
    
    职责：
    - 从 core_bak_refactored/config/{env}/risk.yml 读取风险配置
    - 提供风险参数查询接口
    - 提供市场监控参数查询接口
    """
    
    def __init__(self, environment: str = None):
        """
        初始化风险配置管理器
        
        Args:
            environment: 环境名称（dev/test/prod），默认从环境变量读取
        """
        self.environment = environment or os.getenv('DEEPSEEK_ENV', 'dev')
        self._config_dir = self._get_config_dir()
        
        # 加载风险配置文件
        self._risk_config = self._load_yaml_config('risk.yml')
        
        # 解析风险配置
        self.risk_parameters = self._risk_config.get('risk_parameters', {})
        self.market_monitoring = self._risk_config.get('market_monitoring', {})
        self.limits = self._risk_config.get('limits', {})
    
    def _get_config_dir(self) -> str:
        """获取配置文件目录"""
        # core/risk/config.py -> core_bak_refactored
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
    
    def get_risk_parameters(self, market_code: str) -> Dict[str, Any]:
        """获取指定市场的风险参数
        
        Args:
            market_code: 市场代码（CN, US, HK, JP, EU, SG）
        
        Returns:
            风险参数字典
        """
        return self.risk_parameters.get(market_code, {})
    
    def get_market_monitoring(self, market_code: str) -> Dict[str, Any]:
        """获取指定市场的监控参数
        
        Args:
            market_code: 市场代码
        
        Returns:
            监控参数字典
        """
        return self.market_monitoring.get(market_code, {})
