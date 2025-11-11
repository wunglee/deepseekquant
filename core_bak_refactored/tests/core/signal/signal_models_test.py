"""
信号数据模型测试
"""

import unittest
from datetime import datetime

from core.signal.signal_models import (
    SignalType, SignalStrength, SignalSource, SignalStatus,
    SignalMetadata, TradingSignal
)


class TestSignalEnums(unittest.TestCase):
    """测试信号枚举类"""
    
    def test_signal_type_values(self):
        """测试SignalType枚举值"""
        self.assertEqual(SignalType.BUY.value, "buy")
        self.assertEqual(SignalType.SELL.value, "sell")
        self.assertEqual(SignalType.HOLD.value, "hold")
        self.assertEqual(SignalType.CLOSE.value, "close")
    
    def test_signal_strength_values(self):
        """测试SignalStrength枚举值"""
        self.assertEqual(SignalStrength.WEAK.value, "weak")
        self.assertEqual(SignalStrength.MILD.value, "mild")
        self.assertEqual(SignalStrength.STRONG.value, "strong")
        self.assertEqual(SignalStrength.VERY_STRONG.value, "very_strong")
    
    def test_signal_source_values(self):
        """测试SignalSource枚举值"""
        self.assertEqual(SignalSource.TECHNICAL.value, "technical")
        self.assertEqual(SignalSource.QUANTITATIVE.value, "quantitative")
        self.assertEqual(SignalSource.MACHINE_LEARNING.value, "machine_learning")
    
    def test_signal_status_values(self):
        """测试SignalStatus枚举值"""
        self.assertEqual(SignalStatus.GENERATED.value, "generated")
        self.assertEqual(SignalStatus.VALIDATED.value, "validated")
        self.assertEqual(SignalStatus.REJECTED.value, "rejected")
        self.assertEqual(SignalStatus.EXECUTED.value, "executed")


class TestSignalMetadata(unittest.TestCase):
    """测试信号元数据"""
    
    def test_default_metadata_creation(self):
        """测试默认元数据创建"""
        metadata = SignalMetadata()
        
        self.assertIsNotNone(metadata.generated_at)
        self.assertEqual(metadata.source, SignalSource.TECHNICAL)
        self.assertEqual(metadata.confidence, 0.0)
        self.assertEqual(metadata.strength, SignalStrength.MILD)
        self.assertEqual(metadata.priority, 1)
        self.assertEqual(metadata.tags, [])
        self.assertEqual(metadata.parameters, {})
    
    def test_custom_metadata_creation(self):
        """测试自定义元数据创建"""
        metadata = SignalMetadata(
            source=SignalSource.MACHINE_LEARNING,
            confidence=0.85,
            strength=SignalStrength.STRONG,
            priority=5,
            tags=['ml', 'high_confidence'],
            parameters={'model': 'lstm'},
            model_version='2.0.0',
            strategy_name='ML_Strategy_1'
        )
        
        self.assertEqual(metadata.source, SignalSource.MACHINE_LEARNING)
        self.assertEqual(metadata.confidence, 0.85)
        self.assertEqual(metadata.strength, SignalStrength.STRONG)
        self.assertEqual(metadata.priority, 5)
        self.assertIn('ml', metadata.tags)
        self.assertEqual(metadata.parameters['model'], 'lstm')
        self.assertEqual(metadata.model_version, '2.0.0')
        self.assertEqual(metadata.strategy_name, 'ML_Strategy_1')


class TestTradingSignal(unittest.TestCase):
    """测试交易信号"""
    
    def setUp(self):
        """测试前准备"""
        self.metadata = SignalMetadata(
            source=SignalSource.TECHNICAL,
            confidence=0.75,
            strength=SignalStrength.STRONG
        )
        
        self.signal = TradingSignal(
            id='test_signal_001',
            symbol='000001.SZ',
            signal_type=SignalType.BUY,
            price=10.50,
            timestamp=datetime.now().isoformat(),
            metadata=self.metadata,
            quantity=1000,
            stop_loss=10.00,
            take_profit=11.00,
            timeframe='1d'
        )
    
    def test_signal_creation(self):
        """测试信号创建"""
        self.assertEqual(self.signal.id, 'test_signal_001')
        self.assertEqual(self.signal.symbol, '000001.SZ')
        self.assertEqual(self.signal.signal_type, SignalType.BUY)
        self.assertEqual(self.signal.price, 10.50)
        self.assertEqual(self.signal.quantity, 1000)
        self.assertEqual(self.signal.stop_loss, 10.00)
        self.assertEqual(self.signal.take_profit, 11.00)
    
    def test_signal_default_values(self):
        """测试信号默认值"""
        minimal_signal = TradingSignal(
            id='minimal_001',
            symbol='000002.SZ',
            signal_type=SignalType.SELL,
            price=20.0,
            timestamp=datetime.now().isoformat(),
            metadata=self.metadata
        )
        
        self.assertIsNone(minimal_signal.quantity)
        self.assertIsNone(minimal_signal.stop_loss)
        self.assertIsNone(minimal_signal.take_profit)
        self.assertEqual(minimal_signal.timeframe, '1d')
        self.assertEqual(minimal_signal.status, SignalStatus.GENERATED)
        self.assertEqual(minimal_signal.weight, 1.0)
        self.assertEqual(minimal_signal.risk_score, 0.0)
    
    def test_signal_to_dict(self):
        """测试信号转字典"""
        signal_dict = self.signal.to_dict()
        
        self.assertEqual(signal_dict['id'], 'test_signal_001')
        self.assertEqual(signal_dict['symbol'], '000001.SZ')
        self.assertEqual(signal_dict['signal_type'], 'buy')
        self.assertEqual(signal_dict['price'], 10.50)
        self.assertEqual(signal_dict['quantity'], 1000)
        self.assertEqual(signal_dict['stop_loss'], 10.00)
        self.assertEqual(signal_dict['take_profit'], 11.00)
        self.assertIn('metadata', signal_dict)
        self.assertEqual(signal_dict['metadata']['confidence'], 0.75)
    
    def test_signal_from_dict(self):
        """测试从字典创建信号"""
        signal_dict = {
            'id': 'dict_signal_001',
            'symbol': '000003.SZ',
            'signal_type': 'sell',
            'price': 15.0,
            'timestamp': datetime.now().isoformat(),
            'metadata': {
                'source': 'quantitative',
                'confidence': 0.60,
                'strength': 'mild',
                'priority': 3
            },
            'quantity': 500,
            'timeframe': '1h',
            'weight': 0.8,
            'risk_score': 0.3
        }
        
        signal = TradingSignal.from_dict(signal_dict)
        
        self.assertEqual(signal.id, 'dict_signal_001')
        self.assertEqual(signal.symbol, '000003.SZ')
        self.assertEqual(signal.signal_type, SignalType.SELL)
        self.assertEqual(signal.price, 15.0)
        self.assertEqual(signal.quantity, 500)
        self.assertEqual(signal.timeframe, '1h')
        self.assertEqual(signal.weight, 0.8)
        self.assertEqual(signal.risk_score, 0.3)
        self.assertEqual(signal.metadata.confidence, 0.60)
    
    def test_signal_round_trip_conversion(self):
        """测试信号往返转换"""
        # to_dict -> from_dict -> to_dict
        dict1 = self.signal.to_dict()
        signal2 = TradingSignal.from_dict(dict1)
        dict2 = signal2.to_dict()
        
        self.assertEqual(dict1['id'], dict2['id'])
        self.assertEqual(dict1['symbol'], dict2['symbol'])
        self.assertEqual(dict1['signal_type'], dict2['signal_type'])
        self.assertEqual(dict1['price'], dict2['price'])
        self.assertEqual(dict1['metadata']['confidence'], dict2['metadata']['confidence'])
    
    def test_signal_with_correlation(self):
        """测试带相关性的信号"""
        signal = TradingSignal(
            id='corr_signal_001',
            symbol='000004.SZ',
            signal_type=SignalType.BUY,
            price=12.0,
            timestamp=datetime.now().isoformat(),
            metadata=self.metadata,
            correlation={'000005.SZ': 0.85, '000006.SZ': 0.72}
        )
        
        self.assertEqual(len(signal.correlation), 2)
        self.assertEqual(signal.correlation['000005.SZ'], 0.85)
        self.assertEqual(signal.correlation['000006.SZ'], 0.72)
    
    def test_signal_with_risk_metrics(self):
        """测试带风险指标的信号"""
        signal = TradingSignal(
            id='risk_signal_001',
            symbol='000007.SZ',
            signal_type=SignalType.BUY,
            price=8.0,
            timestamp=datetime.now().isoformat(),
            metadata=self.metadata,
            risk_score=0.45,
            expected_return=0.15,
            expected_hold_period=120,
            volatility=0.25,
            liquidity_score=0.90
        )
        
        self.assertEqual(signal.risk_score, 0.45)
        self.assertEqual(signal.expected_return, 0.15)
        self.assertEqual(signal.expected_hold_period, 120)
        self.assertEqual(signal.volatility, 0.25)
        self.assertEqual(signal.liquidity_score, 0.90)


if __name__ == '__main__':
    unittest.main()
