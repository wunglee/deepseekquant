"""
风险数据模型测试
"""

import unittest
from datetime import datetime

from core.risk.risk_models import (
    RiskLevel, RiskType, RiskMetric, RiskControlAction,
    RiskLimit, PositionLimit, RiskAssessment, RiskEvent, StressTestScenario
)


class TestRiskEnums(unittest.TestCase):
    """测试风险枚举类"""
    
    def test_risk_level_values(self):
        """测试RiskLevel枚举值"""
        self.assertEqual(RiskLevel.VERY_LOW.value, "very_low")
        self.assertEqual(RiskLevel.LOW.value, "low")
        self.assertEqual(RiskLevel.MODERATE.value, "moderate")
        self.assertEqual(RiskLevel.HIGH.value, "high")
        self.assertEqual(RiskLevel.VERY_HIGH.value, "very_high")
        self.assertEqual(RiskLevel.EXTREME.value, "extreme")
    
    def test_risk_type_values(self):
        """测试RiskType枚举值"""
        self.assertEqual(RiskType.MARKET_RISK.value, "market_risk")
        self.assertEqual(RiskType.CREDIT_RISK.value, "credit_risk")
        self.assertEqual(RiskType.LIQUIDITY_RISK.value, "liquidity_risk")
        self.assertEqual(RiskType.OPERATIONAL_RISK.value, "operational_risk")
    
    def test_risk_metric_values(self):
        """测试RiskMetric枚举值"""
        self.assertEqual(RiskMetric.VOLATILITY.value, "volatility")
        self.assertEqual(RiskMetric.VALUE_AT_RISK.value, "value_at_risk")
        self.assertEqual(RiskMetric.EXPECTED_SHORTFALL.value, "expected_shortfall")
    
    def test_risk_control_action_values(self):
        """测试RiskControlAction枚举值"""
        self.assertEqual(RiskControlAction.ALLOW.value, "allow")
        self.assertEqual(RiskControlAction.WARN.value, "warn")
        self.assertEqual(RiskControlAction.REJECT.value, "reject")


class TestRiskLimit(unittest.TestCase):
    """测试风险限额配置"""
    
    def test_risk_limit_creation(self):
        """测试风险限额创建"""
        limit = RiskLimit(
            risk_type=RiskType.MARKET_RISK,
            metric=RiskMetric.VALUE_AT_RISK,
            threshold=0.05,
            action=RiskControlAction.WARN
        )
        
        self.assertEqual(limit.risk_type, RiskType.MARKET_RISK)
        self.assertEqual(limit.metric, RiskMetric.VALUE_AT_RISK)
        self.assertEqual(limit.threshold, 0.05)
        self.assertEqual(limit.action, RiskControlAction.WARN)
    
    def test_risk_limit_to_dict(self):
        """测试风险限额转字典"""
        limit = RiskLimit(
            risk_type=RiskType.MARKET_RISK,
            metric=RiskMetric.VALUE_AT_RISK,
            threshold=0.05
        )
        
        limit_dict = limit.to_dict()
        self.assertIn('risk_type', limit_dict)
        self.assertIn('metric', limit_dict)
        self.assertIn('threshold', limit_dict)
    
    def test_risk_limit_from_dict_with_enum_objects(self):
        """测试从字典创建（枚举对象）"""
        data = {
            'risk_type': RiskType.MARKET_RISK,
            'metric': RiskMetric.VALUE_AT_RISK,
            'threshold': 0.05,
            'action': RiskControlAction.WARN
        }
        
        limit = RiskLimit.from_dict(data)
        self.assertEqual(limit.risk_type, RiskType.MARKET_RISK)
        self.assertEqual(limit.metric, RiskMetric.VALUE_AT_RISK)
    
    def test_risk_limit_from_dict_with_strings(self):
        """测试从字典创建（字符串）"""
        data = {
            'risk_type': 'market_risk',
            'metric': 'value_at_risk',
            'threshold': 0.05,
            'action': 'warn'
        }
        
        limit = RiskLimit.from_dict(data)
        self.assertEqual(limit.risk_type, RiskType.MARKET_RISK)
        self.assertEqual(limit.metric, RiskMetric.VALUE_AT_RISK)
        self.assertEqual(limit.action, RiskControlAction.WARN)
    
    def test_risk_limit_from_dict_with_dict_enums(self):
        """测试从字典创建（字典格式枚举）"""
        data = {
            'risk_type': {'value': 'market_risk'},
            'metric': {'value': 'value_at_risk'},
            'threshold': 0.05,
            'action': {'value': 'warn'}
        }
        
        limit = RiskLimit.from_dict(data)
        self.assertEqual(limit.risk_type, RiskType.MARKET_RISK)
        self.assertEqual(limit.metric, RiskMetric.VALUE_AT_RISK)
        self.assertEqual(limit.action, RiskControlAction.WARN)


class TestPositionLimit(unittest.TestCase):
    """测试头寸限额配置"""
    
    def test_position_limit_creation(self):
        """测试头寸限额创建"""
        limit = PositionLimit(
            symbol='000001.SZ',
            max_notional=1000000.0,
            max_quantity=10000.0,
            max_weight=0.1
        )
        
        self.assertEqual(limit.symbol, '000001.SZ')
        self.assertEqual(limit.max_notional, 1000000.0)
        self.assertEqual(limit.max_quantity, 10000.0)
        self.assertEqual(limit.max_weight, 0.1)
    
    def test_position_limit_defaults(self):
        """测试头寸限额默认值"""
        limit = PositionLimit(
            symbol='000002.SZ',
            max_notional=500000.0,
            max_quantity=5000.0,
            max_weight=0.05
        )
        
        self.assertEqual(limit.min_liquidity_ratio, 0.1)
        self.assertEqual(limit.max_leverage, 1.0)
        self.assertEqual(limit.concentration_limit, 0.2)
    
    def test_position_limit_to_from_dict(self):
        """测试头寸限额字典转换"""
        limit = PositionLimit(
            symbol='000003.SZ',
            max_notional=2000000.0,
            max_quantity=20000.0,
            max_weight=0.15
        )
        
        limit_dict = limit.to_dict()
        limit2 = PositionLimit.from_dict(limit_dict)
        
        self.assertEqual(limit.symbol, limit2.symbol)
        self.assertEqual(limit.max_notional, limit2.max_notional)


class TestRiskAssessment(unittest.TestCase):
    """测试风险评估结果"""
    
    def test_risk_assessment_creation(self):
        """测试风险评估创建"""
        assessment = RiskAssessment(
            timestamp=datetime.now().isoformat(),
            portfolio_id='portfolio_001',
            overall_risk_level=RiskLevel.MODERATE,
            risk_score=50.0,
            value_at_risk=0.05,
            expected_shortfall=0.08,
            max_drawdown=0.15,
            liquidity_risk=0.03,
            concentration_risk=0.04,
            leverage_risk=0.02,
            stress_test_results={'market_crash': -0.20},
            scenario_analysis={'recession': -0.15},
            risk_contributions={'stock_a': 0.30},
            limit_breaches=[],
            recommendations=[]
        )
        
        self.assertEqual(assessment.overall_risk_level, RiskLevel.MODERATE)
        self.assertEqual(assessment.risk_score, 50.0)
        self.assertEqual(assessment.value_at_risk, 0.05)
    
    def test_risk_assessment_from_dict(self):
        """测试从字典创建风险评估"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_id': 'portfolio_002',
            'overall_risk_level': 'high',
            'risk_score': 75.0,
            'value_at_risk': 0.10,
            'expected_shortfall': 0.15,
            'max_drawdown': 0.25,
            'liquidity_risk': 0.05,
            'concentration_risk': 0.08,
            'leverage_risk': 0.06,
            'stress_test_results': {},
            'scenario_analysis': {},
            'risk_contributions': {},
            'limit_breaches': [],
            'recommendations': []
        }
        
        assessment = RiskAssessment.from_dict(data)
        self.assertEqual(assessment.overall_risk_level, RiskLevel.HIGH)
        self.assertEqual(assessment.risk_score, 75.0)


class TestRiskEvent(unittest.TestCase):
    """测试风险事件记录"""
    
    def test_risk_event_creation(self):
        """测试风险事件创建"""
        event = RiskEvent(
            event_id='event_001',
            event_type=RiskType.MARKET_RISK,
            severity=RiskLevel.HIGH,
            timestamp=datetime.now().isoformat(),
            description='Market volatility spike',
            triggered_by='market_data',
            impact_assessment={'portfolio_loss': 0.08},
            action_taken=RiskControlAction.REDUCE
        )
        
        self.assertEqual(event.event_id, 'event_001')
        self.assertEqual(event.event_type, RiskType.MARKET_RISK)
        self.assertEqual(event.severity, RiskLevel.HIGH)
        self.assertEqual(event.action_taken, RiskControlAction.REDUCE)
    
    def test_risk_event_from_dict(self):
        """测试从字典创建风险事件"""
        data = {
            'event_id': 'event_002',
            'event_type': 'liquidity_risk',
            'severity': 'moderate',
            'timestamp': datetime.now().isoformat(),
            'description': 'Low liquidity detected',
            'triggered_by': 'liquidity_monitor',
            'impact_assessment': {},
            'action_taken': 'warn'
        }
        
        event = RiskEvent.from_dict(data)
        self.assertEqual(event.event_type, RiskType.LIQUIDITY_RISK)
        self.assertEqual(event.severity, RiskLevel.MODERATE)
        self.assertEqual(event.action_taken, RiskControlAction.WARN)


class TestStressTestScenario(unittest.TestCase):
    """测试压力测试场景"""
    
    def test_stress_test_scenario_creation(self):
        """测试压力测试场景创建"""
        scenario = StressTestScenario(
            scenario_id='scenario_001',
            name='Market Crash',
            description='Severe market downturn',
            parameters={'market_drop': -0.30},
            probability=0.05,
            impact_level=RiskLevel.EXTREME,
            duration='1w',
            triggers=['global_crisis'],
            mitigation_strategies=['diversification']
        )
        
        self.assertEqual(scenario.scenario_id, 'scenario_001')
        self.assertEqual(scenario.name, 'Market Crash')
        self.assertEqual(scenario.impact_level, RiskLevel.EXTREME)
        self.assertEqual(scenario.probability, 0.05)
    
    def test_stress_test_scenario_from_dict(self):
        """测试从字典创建压力测试场景"""
        data = {
            'scenario_id': 'scenario_002',
            'name': 'Interest Rate Shock',
            'description': 'Sudden rate hike',
            'parameters': {'rate_increase': 0.02},
            'probability': 0.10,
            'impact_level': 'high',
            'duration': '3d',
            'triggers': ['central_bank_decision'],
            'mitigation_strategies': ['hedge_with_bonds']
        }
        
        scenario = StressTestScenario.from_dict(data)
        self.assertEqual(scenario.impact_level, RiskLevel.HIGH)
        self.assertEqual(scenario.name, 'Interest Rate Shock')


if __name__ == '__main__':
    unittest.main()
