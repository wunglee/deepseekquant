"""
信号过滤器测试
"""

import unittest
from datetime import datetime

from core.signal.signal_models import (
    TradingSignal, SignalType, SignalMetadata
)
from core.signal.signal_filter import SignalFilter


class TestSignalFilter(unittest.TestCase):
    """测试信号过滤器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {}
        self.filter = SignalFilter(self.config)
    
    def test_filter_initialization(self):
        """测试过滤器初始化"""
        self.assertIsNotNone(self.filter)
        self.assertEqual(self.filter.config, {})
    
    def test_filter_signals_empty_dict(self):
        """测试空字典过滤"""
        result = self.filter.filter_signals({})
        self.assertEqual(result, {})
    
    def test_filter_signals_returns_all(self):
        """测试默认返回所有信号"""
        signals = {
            '000001.SZ': [
                TradingSignal(
                    id='test_001',
                    symbol='000001.SZ',
                    signal_type=SignalType.BUY,
                    price=10.0,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(confidence=0.7)
                ),
                TradingSignal(
                    id='test_002',
                    symbol='000001.SZ',
                    signal_type=SignalType.SELL,
                    price=10.5,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(confidence=0.6)
                )
            ],
            '000002.SZ': [
                TradingSignal(
                    id='test_003',
                    symbol='000002.SZ',
                    signal_type=SignalType.BUY,
                    price=20.0,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(confidence=0.8)
                )
            ]
        }
        
        result = self.filter.filter_signals(signals)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result['000001.SZ']), 2)
        self.assertEqual(len(result['000002.SZ']), 1)


if __name__ == '__main__':
    unittest.main()
