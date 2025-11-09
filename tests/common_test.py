#!/usr/bin/env python3
# common_test.py - 适配 unittest 发现机制的测试文件

import unittest
import sys
import os
import json
from datetime import datetime
from enum import Enum
from dataclasses import asdict

# 处理路径中的空格和特殊字符
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"运行测试的目录: {current_dir}")
print(f"Python 路径: {sys.path}")

# 导入 common 模块（直接导入，简化逻辑）
import common
from common import (
    SignalType, SignalStrength, SignalSource, SignalStatus, RiskLevel,
    PortfolioObjective, AllocationMethod, ExecutionStrategy, TradeDirection,
    OrderType, OrderStatus, PositionStatus, DataFrequency, DataSourceType,
    ProcessorState, TradingMode, ConfigFormat, ConfigSource,
    AcquisitionFunctionType, OptimizationObjective,
    SignalMetadata, TradingSignal, RiskAssessment, PerformanceMetrics, SystemStatus,
    enum_to_dict, validate_enum_value, get_enum_values, get_enum_names,
    serialize_dict, deserialize_dict, DeepSeekQuantEncoder,
    DEFAULT_CONFIG_PATH, DEFAULT_LOG_LEVEL, DEFAULT_INITIAL_CAPITAL,
    ERROR_CONFIG_LOAD, SUCCESS_CONFIG_LOAD, WARNING_CONFIG_DEFAULT,
    INFO_SYSTEM_START, MILLISECONDS_PER_SECOND, TRADING_DAYS_PER_YEAR, EPSILON
)

print("✓ 成功导入 common 模块")


class TestCommonEnums(unittest.TestCase):
    """测试所有枚举类型"""

    def test_signal_type_enum(self):
        """测试信号类型枚举"""
        self.assertEqual(SignalType.BUY.value, "buy")
        self.assertEqual(SignalType.SELL.value, "sell")
        self.assertEqual(SignalType.HOLD.value, "hold")
        self.assertTrue(validate_enum_value(SignalType, "buy"))
        self.assertFalse(validate_enum_value(SignalType, "invalid"))

    def test_signal_strength_enum(self):
        """测试信号强度枚举"""
        self.assertEqual(SignalStrength.WEAK.value, "weak")
        self.assertEqual(SignalStrength.EXTREME.value, "extreme")

    def test_signal_source_enum(self):
        """测试信号来源枚举"""
        self.assertEqual(SignalSource.TECHNICAL.value, "technical")
        self.assertEqual(SignalSource.MACHINE_LEARNING.value, "machine_learning")

    def test_risk_level_enum(self):
        """测试风险等级枚举"""
        self.assertEqual(RiskLevel.VERY_LOW.value, "very_low")
        self.assertEqual(RiskLevel.BLACK_SWAN.value, "black_swan")

    def test_trading_mode_enum(self):
        """测试交易模式枚举"""
        self.assertEqual(TradingMode.PAPER_TRADING.value, "paper_trading")
        self.assertEqual(TradingMode.LIVE_TRADING.value, "live_trading")


class TestSignalMetadata(unittest.TestCase):
    """测试信号元数据类"""

    def test_default_values(self):
        """测试默认值"""
        metadata = SignalMetadata()
        self.assertEqual(metadata.source, SignalSource.TECHNICAL)
        self.assertEqual(metadata.confidence, 0.0)
        self.assertEqual(metadata.strength, SignalStrength.MILD)
        self.assertEqual(metadata.priority, 1)
        self.assertIsNotNone(metadata.generated_at)
        self.assertEqual(metadata.tags, [])
        self.assertEqual(metadata.parameters, {})

    def test_custom_values(self):
        """测试自定义值"""
        metadata = SignalMetadata(
            source=SignalSource.MACHINE_LEARNING,
            confidence=0.95,
            strength=SignalStrength.STRONG,
            priority=5,
            tags=["trend", "momentum"],
            parameters={"window": 20, "threshold": 0.5}
        )
        self.assertEqual(metadata.source, SignalSource.MACHINE_LEARNING)
        self.assertEqual(metadata.confidence, 0.95)
        self.assertEqual(metadata.tags, ["trend", "momentum"])

    def test_to_dict(self):
        """测试转换为字典"""
        metadata = SignalMetadata(
            source=SignalSource.TECHNICAL,
            confidence=0.8
        )
        metadata_dict = asdict(metadata)
        # asdict() 不会自动转换枚举为字符串,而是保持枚举对象
        self.assertEqual(metadata_dict['source'], SignalSource.TECHNICAL)
        self.assertEqual(metadata_dict['confidence'], 0.8)


class TestTradingSignal(unittest.TestCase):
    """测试交易信号类"""

    def setUp(self):
        """设置测试数据"""
        self.metadata = SignalMetadata(
            source=SignalSource.TECHNICAL,
            confidence=0.9
        )
        self.signal_data = {
            'id': 'test_signal_001',
            'symbol': 'AAPL',
            'signal_type': SignalType.BUY,
            'price': 150.0,
            'timestamp': '2024-01-15T10:30:00',
            'metadata': self.metadata,
            'quantity': 100,
            'stop_loss': 145.0,
            'take_profit': 160.0,
            'timeframe': '1d',
            'status': SignalStatus.GENERATED,
            'reason': 'Breakout above resistance',
            'weight': 0.8,
            'risk_score': 0.2,
            'expected_return': 0.15
        }

    def test_signal_creation(self):
        """测试信号创建"""
        signal = TradingSignal(**self.signal_data)
        self.assertEqual(signal.symbol, 'AAPL')
        self.assertEqual(signal.signal_type, SignalType.BUY)
        self.assertEqual(signal.price, 150.0)
        self.assertEqual(signal.quantity, 100)

    def test_to_dict_method(self):
        """测试to_dict方法"""
        signal = TradingSignal(**self.signal_data)
        signal_dict = signal.to_dict()

        self.assertEqual(signal_dict['symbol'], 'AAPL')
        self.assertEqual(signal_dict['signal_type'], 'buy')
        self.assertEqual(signal_dict['price'], 150.0)
        # TradingSignal.to_dict() 会将枚举转换为字符串值
        # 但metadata是通过asdict()转换的,所以枚举保持为对象
        self.assertEqual(signal_dict['metadata']['source'], SignalSource.TECHNICAL)
        self.assertEqual(signal_dict['metadata']['confidence'], 0.9)

    def test_from_dict_method(self):
        """测试from_dict方法"""
        signal = TradingSignal(**self.signal_data)
        signal_dict = signal.to_dict()

        # 从字典重建信号
        reconstructed_signal = TradingSignal.from_dict(signal_dict)
        self.assertEqual(reconstructed_signal.symbol, 'AAPL')
        self.assertEqual(reconstructed_signal.signal_type, SignalType.BUY)
        self.assertEqual(reconstructed_signal.metadata.source, SignalSource.TECHNICAL)

    def test_default_values(self):
        """测试默认值"""
        signal = TradingSignal(
            id='test_001',
            symbol='TSLA',
            signal_type=SignalType.BUY,
            price=200.0,
            timestamp='2024-01-15T10:30:00',
            metadata=self.metadata
        )
        self.assertEqual(signal.timeframe, '1d')
        self.assertEqual(signal.status, SignalStatus.GENERATED)
        self.assertEqual(signal.weight, 1.0)
        self.assertEqual(signal.risk_score, 0.0)


class TestRiskAssessment(unittest.TestCase):
    """测试风险评估类"""

    def test_risk_assessment_creation(self):
        """测试风险评估创建"""
        risk = RiskAssessment(
            approved=True,
            reason="Within risk limits",
            risk_level=RiskLevel.MODERATE,
            warnings=["High volatility"],
            max_position_size=50000.0,
            suggested_allocation=0.05,
            risk_score=0.4
        )
        self.assertTrue(risk.approved)
        self.assertEqual(risk.risk_level, RiskLevel.MODERATE)
        self.assertEqual(risk.warnings, ["High volatility"])

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        risk = RiskAssessment(
            approved=False,
            risk_level=RiskLevel.HIGH,
            risk_score=0.7
        )
        risk_dict = risk.to_dict()
        reconstructed_risk = RiskAssessment.from_dict(risk_dict)

        self.assertFalse(reconstructed_risk.approved)
        self.assertEqual(reconstructed_risk.risk_level, RiskLevel.HIGH)
        self.assertEqual(reconstructed_risk.risk_score, 0.7)


class TestPerformanceMetrics(unittest.TestCase):
    """测试性能指标类"""

    def test_initial_state(self):
        """测试初始状态"""
        metrics = PerformanceMetrics()
        self.assertEqual(metrics.total_operations, 0)
        self.assertEqual(metrics.successful_operations, 0)
        self.assertEqual(metrics.failed_operations, 0)
        self.assertEqual(metrics.avg_processing_time, 0.0)
        self.assertEqual(metrics.min_processing_time, float('inf'))

    def test_update_method(self):
        """测试更新方法"""
        metrics = PerformanceMetrics()

        # 第一次更新
        metrics.update(True, 0.1)
        self.assertEqual(metrics.total_operations, 1)
        self.assertEqual(metrics.successful_operations, 1)
        self.assertEqual(metrics.failed_operations, 0)
        self.assertEqual(metrics.avg_processing_time, 0.1)
        self.assertEqual(metrics.min_processing_time, 0.1)
        self.assertEqual(metrics.max_processing_time, 0.1)

        # 第二次更新
        metrics.update(False, 0.3)
        self.assertEqual(metrics.total_operations, 2)
        self.assertEqual(metrics.successful_operations, 1)
        self.assertEqual(metrics.failed_operations, 1)
        self.assertEqual(metrics.avg_processing_time, 0.2)
        self.assertEqual(metrics.min_processing_time, 0.1)
        self.assertEqual(metrics.max_processing_time, 0.3)
        self.assertEqual(metrics.error_rate, 0.5)

    def test_to_dict_method(self):
        """测试转换为字典"""
        metrics = PerformanceMetrics()
        metrics.update(True, 0.15)
        metrics.update(True, 0.25)

        metrics_dict = metrics.to_dict()
        self.assertEqual(metrics_dict['total_operations'], 2)
        self.assertEqual(metrics_dict['successful_operations'], 2)
        self.assertEqual(metrics_dict['avg_processing_time'], 0.2)


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def test_enum_to_dict(self):
        """测试枚举转字典"""
        result = enum_to_dict(SignalType)
        self.assertEqual(result['BUY'], 'buy')
        self.assertEqual(result['SELL'], 'sell')
        self.assertIn('HOLD', result)

    def test_validate_enum_value(self):
        """测试枚举值验证"""
        self.assertTrue(validate_enum_value(SignalType, 'buy'))
        self.assertTrue(validate_enum_value(SignalType, SignalType.BUY))
        self.assertFalse(validate_enum_value(SignalType, 'invalid_value'))

    def test_get_enum_values(self):
        """测试获取枚举值列表"""
        values = get_enum_values(SignalType)
        self.assertIn('buy', values)
        self.assertIn('sell', values)
        self.assertNotIn('BUY', values)  # 应该是值，不是名称

    def test_get_enum_names(self):
        """测试获取枚举名称列表"""
        names = get_enum_names(SignalType)
        self.assertIn('BUY', names)
        self.assertIn('SELL', names)
        self.assertNotIn('buy', names)  # 应该是名称，不是值

    def test_serialize_deserialize_dict(self):
        """测试字典序列化和反序列化"""
        test_dict = {
            'symbol': 'AAPL',
            'price': 150.0,
            'active': True,
            'tags': ['tech', 'large_cap']
        }

        # 序列化
        json_str = serialize_dict(test_dict)
        self.assertIsInstance(json_str, str)
        self.assertIn('AAPL', json_str)

        # 反序列化
        reconstructed_dict = deserialize_dict(json_str)
        self.assertEqual(reconstructed_dict['symbol'], 'AAPL')
        self.assertEqual(reconstructed_dict['price'], 150.0)


class TestConstants(unittest.TestCase):
    """测试常量"""

    def test_config_constants(self):
        """测试配置常量"""
        self.assertEqual(DEFAULT_CONFIG_PATH, "config/deepseekquant.json")
        self.assertEqual(DEFAULT_LOG_LEVEL, "INFO")
        self.assertEqual(DEFAULT_INITIAL_CAPITAL, 1000000.0)

    def test_message_constants(self):
        """测试消息常量"""
        self.assertEqual(ERROR_CONFIG_LOAD, "配置加载失败")
        self.assertEqual(SUCCESS_CONFIG_LOAD, "配置加载成功")
        self.assertEqual(WARNING_CONFIG_DEFAULT, "使用默认配置")
        self.assertEqual(INFO_SYSTEM_START, "系统启动")

    def test_numeric_constants(self):
        """测试数值常量"""
        self.assertEqual(MILLISECONDS_PER_SECOND, 1000)
        self.assertEqual(TRADING_DAYS_PER_YEAR, 252)
        self.assertEqual(EPSILON, 1e-10)


# 允许直接运行此文件
if __name__ == '__main__':
    # 设置测试加载器
    loader = unittest.TestLoader()

    # 添加所有测试类
    test_classes = [
        TestCommonEnums,
        TestSignalMetadata,
        TestTradingSignal,
        TestRiskAssessment,
        TestPerformanceMetrics,
        TestUtilityFunctions,
        TestConstants
    ]

    # 创建测试套件
    suites = [loader.loadTestsFromTestCase(test_class) for test_class in test_classes]
    suite = unittest.TestSuite(suites)

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 退出代码
    exit(0 if result.wasSuccessful() else 1)