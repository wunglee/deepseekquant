"""
信号处理器核心 - 业务层
从 core_bak/signal_engine.py 拆分
职责: 信号处理流程协调
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from .signal_models import TradingSignal
from .signal_generator import SignalGenerator
from .signal_aggregator import SignalAggregator
from .signal_validator import SignalValidator
from .signal_filter import SignalFilter

logger = logging.getLogger('DeepSeekQuant.SignalProcessor')


class SignalProcessor:
    """信号处理器 - 协调信号生成、聚合、验证、过滤"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化各个组件
        self.generator = SignalGenerator(config)
        self.aggregator = SignalAggregator(config)
        self.validator = SignalValidator(config)
        self.filter = SignalFilter(config)
        
        logger.info("信号处理器初始化完成")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理信号生成请求
        
        Args:
            data: 市场数据
            
        Returns:
            处理结果
        """
        try:
            # 1. 生成原始信号
            raw_signals = self.generator.generate_signals(data)
            
            # 2. 聚合信号
            aggregated_signals = self.aggregator.aggregate(raw_signals)
            
            # 3. 验证信号
            valid_signals = self.validator.validate(aggregated_signals)
            
            # 4. 过滤信号
            final_signals = self.filter.filter_signals(valid_signals)
            
            return {
                'success': True,
                'signals': final_signals,
                'count': len(final_signals),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"信号处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'signals': []
            }
    
    def get_active_signals(self) -> List[TradingSignal]:
        """获取当前活跃信号"""
        # 从存储中获取活跃信号
        return []
    
    def clear_expired_signals(self):
        """清理过期信号"""
        pass
