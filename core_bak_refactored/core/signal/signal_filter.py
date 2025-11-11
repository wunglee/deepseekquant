"""
信号过滤器 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 过滤和筛选信号
"""

from typing import Dict, List
import logging

from .signal_models import TradingSignal

logger = logging.getLogger('DeepSeekQuant.SignalFilter')


class SignalFilter:
    """信号过滤器"""
    
    def __init__(self, config: Dict):
        self.config = config
    
