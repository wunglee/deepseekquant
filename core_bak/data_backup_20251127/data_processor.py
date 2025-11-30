"""
数据处理器核心 - 业务层
从 core_bak/data_fetcher.py 拆分
职责: 协调数据获取、验证、质量监控
"""

from typing import Dict, Any
from datetime import datetime
import logging
import asyncio

from .data_fetcher import DataFetcher
from .data_validator import DataValidator
from .data_quality_monitor import DataQualityMonitor

logger = logging.getLogger('DeepSeekQuant.DataProcessor')


class DataProcessor:
    """数据处理器 - 协调数据相关组件"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # 初始化各个组件
        self.fetcher = DataFetcher(config)
        self.validator = DataValidator(config)
        self.quality_monitor = DataQualityMonitor(config)
        
        logger.info("数据处理器初始化完成")
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理数据请求
        
        Args:
            data: 输入数据
            
        Returns:
            处理结果
        """
        try:
            action = data.get('action')
            
            if action == 'fetch':
                return self._handle_fetch(data)
            elif action == 'validate':
                return self._handle_validate(data)
            elif action == 'monitor':
                return self._handle_monitor(data)
            else:
                return {'success': False, 'error': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"数据处理失败: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_fetch(self, data: Dict) -> Dict:
        """处理数据获取"""
        symbols = data.get('symbols', [])
        period = data.get('period', '1y')
        interval = data.get('interval', '1d')
        data_type = data.get('data_type', 'ohlcv')
        adjustments = data.get('adjustments', True)
        result = asyncio.get_event_loop().run_until_complete(
            self.fetcher.get_historical_data(symbols, period, interval, data_type, adjustments)
        )
        return {'success': True, 'data': result}
    
    def _handle_validate(self, data: Dict) -> Dict:
        """处理数据验证"""
        validation = self.validator.validate_market_data(data.get('market_data', []))
        return {'success': True, 'validation': validation}
    
    def _handle_monitor(self, data: Dict) -> Dict:
        """处理质量监控"""
        report = self.quality_monitor.generate_report(data)
        return {'success': True, 'report': report}
