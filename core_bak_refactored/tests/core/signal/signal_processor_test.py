"""
信号处理器测试
"""

import unittest
from datetime import datetime

from core.signal.signal_processor import SignalProcessor


class TestSignalProcessor(unittest.TestCase):
    """测试信号处理器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {
            'technical': {},
            'quantitative': {},
            'ml': {'enabled': False}
        }
        self.processor = SignalProcessor(self.config)
    
    def test_processor_initialization(self):
        """测试处理器初始化"""
        self.assertIsNotNone(self.processor)
        self.assertIsNotNone(self.processor.generator)
        self.assertIsNotNone(self.processor.aggregator)
        self.assertIsNotNone(self.processor.validator)
        self.assertIsNotNone(self.processor.filter)
    
    def test_get_active_signals(self):
        """测试获取活跃信号"""
        signals = self.processor.get_active_signals()
        self.assertIsInstance(signals, list)
        self.assertEqual(len(signals), 0)  # 默认无活跃信号
    
    def test_clear_expired_signals(self):
        """测试清理过期信号"""
        # 应该不抛异常
        self.processor.clear_expired_signals()


if __name__ == '__main__':
    unittest.main()
