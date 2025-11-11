"""
信号聚合器测试
"""

import unittest
from datetime import datetime

from core.signal.signal_models import (
    TradingSignal, SignalType, SignalStrength, SignalSource, SignalMetadata
)
from core.signal.signal_aggregator import SignalAggregator


class TestSignalAggregator(unittest.TestCase):
    """测试信号聚合器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = {}
        self.aggregator = SignalAggregator(self.config)
    
    def test_aggregator_initialization(self):
        """测试聚合器初始化"""
        self.assertIsNotNone(self.aggregator)
        self.assertEqual(self.aggregator.config, {})
    
    def test_calculate_weighted_strength_empty_list(self):
        """测试空列表的加权强度"""
        strength = self.aggregator.calculate_weighted_strength([])
        self.assertEqual(strength, SignalStrength.WEAK)
    
    def test_calculate_weighted_strength_single_signal(self):
        """测试单个信号的加权强度"""
        signal = TradingSignal(
            id='test_001',
            symbol='000001.SZ',
            signal_type=SignalType.BUY,
            price=10.0,
            timestamp=datetime.now().isoformat(),
            metadata=SignalMetadata(confidence=0.85),
            weight=1.0
        )
        
        strength = self.aggregator.calculate_weighted_strength([signal])
        self.assertEqual(strength, SignalStrength.VERY_STRONG)  # >= 0.8
    
    def test_calculate_weighted_strength_multiple_signals(self):
        """测试多个信号的加权强度"""
        signals = [
            TradingSignal(
                id='test_001',
                symbol='000001.SZ',
                signal_type=SignalType.BUY,
                price=10.0,
                timestamp=datetime.now().isoformat(),
                metadata=SignalMetadata(confidence=0.6),
                weight=1.0
            ),
            TradingSignal(
                id='test_002',
                symbol='000001.SZ',
                signal_type=SignalType.BUY,
                price=10.0,
                timestamp=datetime.now().isoformat(),
                metadata=SignalMetadata(confidence=0.8),
                weight=1.0
            )
        ]
        
        strength = self.aggregator.calculate_weighted_strength(signals)
        # 加权平均 = (0.6 + 0.8) / 2 = 0.7, 对应 STRONG
        self.assertEqual(strength, SignalStrength.STRONG)
    
    def test_calculate_weighted_strength_with_different_weights(self):
        """测试不同权重的加权强度"""
        signals = [
            TradingSignal(
                id='test_001',
                symbol='000001.SZ',
                signal_type=SignalType.BUY,
                price=10.0,
                timestamp=datetime.now().isoformat(),
                metadata=SignalMetadata(confidence=0.9),
                weight=2.0  # 高权重
            ),
            TradingSignal(
                id='test_002',
                symbol='000001.SZ',
                signal_type=SignalType.BUY,
                price=10.0,
                timestamp=datetime.now().isoformat(),
                metadata=SignalMetadata(confidence=0.3),
                weight=1.0  # 低权重
            )
        ]
        
        strength = self.aggregator.calculate_weighted_strength(signals)
        # 加权平均 = (0.9*2 + 0.3*1) / (2+1) = 2.1/3 = 0.7
        self.assertEqual(strength, SignalStrength.STRONG)
    
    def test_calculate_weighted_strength_zero_weights(self):
        """测试零权重的情况"""
        signals = [
            TradingSignal(
                id='test_001',
                symbol='000001.SZ',
                signal_type=SignalType.BUY,
                price=10.0,
                timestamp=datetime.now().isoformat(),
                metadata=SignalMetadata(confidence=0.9),
                weight=0.0
            )
        ]
        
        strength = self.aggregator.calculate_weighted_strength(signals)
        self.assertEqual(strength, SignalStrength.WEAK)  # 零权重返回WEAK
    
    def test_aggregate_signals(self):
        """测试信号聚合"""
        signals = {
            '000001.SZ': [
                TradingSignal(
                    id='test_001',
                    symbol='000001.SZ',
                    signal_type=SignalType.BUY,
                    price=10.0,
                    timestamp=datetime.now().isoformat(),
                    metadata=SignalMetadata(confidence=0.7)
                )
            ]
        }
        
        result = self.aggregator.aggregate(signals)
        self.assertEqual(len(result), 1)
        self.assertIn('000001.SZ', result)


if __name__ == '__main__':
    unittest.main()
