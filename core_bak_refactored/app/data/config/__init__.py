"""
配置加载工具模块
"""

import json
import os
from typing import Dict, Any


def load_config(config_file: str = 'default_config.json') -> Dict[str, Any]:
    """加载配置文件"""
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, config_file)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dashboard_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取仪表板配置"""
    return config.get('dashboard', {})


def get_api_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """获取API服务配置"""
    return config.get('api_service', {})
