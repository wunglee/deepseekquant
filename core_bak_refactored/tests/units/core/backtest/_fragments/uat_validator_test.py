"""
UATValidator tests: exception handling (3-level alerts) and UAT report.
"""


import pandas as pd

from core_bak_refactored.core.backtest._fragments.uat_validator import (
    UATValidator,
    AlertLevel,
    UATResult,
    BusinessExemption,
)


class TestAbnormalHandlingAlerts:
    """异常处理三级告警测试"""
    
    def test_level1_alert_15_to_20_percent(self):
        """测试Level 1告警：15%-20%误差"""
        validator = UATValidator()
        alert = validator.handle_exception(
            prediction_error=0.17,
            event_id='test_event_level1'
        )
        assert alert is not None
        assert alert.level == AlertLevel.LEVEL_1
        assert alert.error_range == "15%-20%"
        assert alert.action == "内部记录，下周复核"
        assert alert.report_deadline == "3个工作日内"
    
    def test_level2_alert_20_to_25_percent(self):
        """测试Level 2告警：20%-25%误差"""
        validator = UATValidator()
        alert = validator.handle_exception(
            prediction_error=0.22,
            event_id='test_event_level2'
        )
        assert alert is not None
        assert alert.level == AlertLevel.LEVEL_2
        assert alert.error_range == "20%-25%"
        assert alert.action == "预警，人工复核"
        assert alert.report_deadline == "24小时内"
    
    def test_level3_alert_above_25_percent(self):
        """测试Level 3告警：>25%误差"""
        validator = UATValidator()
        alert = validator.handle_exception(
            prediction_error=0.30,
            event_id='test_event_level3'
        )
        assert alert is not None
        assert alert.level == AlertLevel.LEVEL_3
        assert alert.error_range == ">25%"
        assert alert.action == "暂停自动报送，立即干预"
        assert alert.report_deadline == "立即"
    
    def test_no_alert_below_15_percent(self):
        """测试无告警：<15%误差"""
        validator = UATValidator()
        alert = validator.handle_exception(
            prediction_error=0.10,
            event_id='test_event_no_alert'
        )
        assert alert is None
    
    def test_alert_history_tracking(self):
        """测试告警历史记录"""
        validator = UATValidator()
        validator.handle_exception(0.16, 'event1')
        validator.handle_exception(0.22, 'event2')
        validator.handle_exception(0.30, 'event3')
        
        all_alerts = validator.get_alert_history()
        assert len(all_alerts) == 3
        
        critical_alerts = validator.get_alert_history(level=AlertLevel.LEVEL_3)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].metadata['event_id'] == 'event3'


class TestUATReportGeneration:
    """UAT报告生成测试"""
    
    def test_generate_basic_uat_report(self):
        """测试基础UAT报告生成"""
        validator = UATValidator()
        
        test_results = {
            'weighted_average_error': UATResult(
                test_item='weighted_average_error',
                passed=True,
                actual_value=0.12,
                threshold=0.15
            ),
            'cross_market_consistency': UATResult(
                test_item='cross_market_consistency',
                passed=True,
                actual_value=0.87,
                threshold=0.85
            ),
            'data_quality': UATResult(
                test_item='data_quality',
                passed=False,
                actual_value=0.88,
                threshold=0.90
            )
        }
        
        report = validator.generate_uat_report(test_results)
        
        assert 'uat_version' in report
        assert 'overall_status' in report
        assert 'core_metrics' in report
        assert 'abnormal_handling' in report
        assert 'cross_validation' in report
        assert 'recommendations' in report
        
        assert report['core_metrics']['total_tests'] == 3
        assert report['core_metrics']['passed_tests'] == 2
        assert report['core_metrics']['failed_tests'] == 1
        assert abs(report['core_metrics']['pass_rate'] - 0.667) < 0.01
        
        assert report['overall_status']['passed'] is False
        assert report['overall_status']['core_tests_passed'] is False
    
    def test_uat_report_with_abnormal_handling(self):
        """测试包含异常处置记录的UAT报告"""
        validator = UATValidator()
        
        validator.handle_exception(0.17, 'event1')
        validator.handle_exception(0.22, 'event2')
        validator.handle_exception(0.30, 'event3')
        
        test_results = {
            'test1': UATResult('test1', True, 0.10, 0.15)
        }
        
        report = validator.generate_uat_report(test_results)
        
        assert report['abnormal_handling']['total_alerts'] == 3
        assert report['abnormal_handling']['level_breakdown']['LEVEL_1'] == 1
        assert report['abnormal_handling']['level_breakdown']['LEVEL_2'] == 1
        assert report['abnormal_handling']['level_breakdown']['LEVEL_3'] == 1
        
        assert len(report['abnormal_handling']['alert_details']) == 3
        assert report['abnormal_handling']['alert_details'][0]['level'] == 'LEVEL_1'
        assert report['abnormal_handling']['alert_details'][1]['level'] == 'LEVEL_2'
        assert report['abnormal_handling']['alert_details'][2]['level'] == 'LEVEL_3'
        
        assert report['overall_status']['passed'] is False
        assert report['overall_status']['no_critical_alerts'] is False
    
    def test_uat_report_with_cross_validation(self):
        """测试包含交叉验证结果的UAT报告"""
        validator = UATValidator()
        
        test_results = {
            'test1': UATResult('test1', True, 0.10, 0.15)
        }
        
        cross_validation = {
            'enabled': True,
            'passed': True,
            'sources_compared': ['yahoo', 'mock'],
            'comparisons': [{
                'source_a': 'yahoo',
                'source_b': 'mock',
                'passed': True,
                'overlap_days': 100
            }]
        }
        
        report = validator.generate_uat_report(test_results, cross_validation_results=cross_validation)
        
        assert report['cross_validation']['enabled'] is True
        assert report['cross_validation']['passed'] is True
        assert len(report['cross_validation']['sources_compared']) == 2
        assert report['overall_status']['cross_validation_passed'] is True
    
    def test_uat_report_recommendations(self):
        """测试UAT报告建议生成"""
        validator = UATValidator()
        
        test_results_pass = {
            'test1': UATResult('test1', True, 0.10, 0.15)
        }
        report = validator.generate_uat_report(test_results_pass)
        assert any('正常' in rec for rec in report['recommendations'])
        
        validator2 = UATValidator()
        test_results_fail = {
            'data_quality': UATResult('data_quality', False, 0.88, 0.90)
        }
        report2 = validator2.generate_uat_report(test_results_fail)
        assert any('data_quality' in rec for rec in report2['recommendations'])
        
        validator3 = UATValidator()
        validator3.handle_exception(0.30, 'critical_event')
        report3 = validator3.generate_uat_report(test_results_pass)
        assert any('Level 3' in rec or '严重' in rec for rec in report3['recommendations'])


class TestBusinessExemption:
    """业务豁免机制测试（专家answer.md第3轮4节）"""
    
    def test_business_exemption_creation(self):
        """测试业务豁免创建"""
        exemption = BusinessExemption(
            category='extreme_market',
            reason='2024年X月X日全球市场因XX事件出现系统性波动',
            approvers=['张三（风控总监）', '李四（业务负责人）'],
            period=(datetime(2024, 1, 1), datetime(2024, 1, 31)),
            audit_trail={
                'decision_number': '2024-001',
                'meeting_minutes': 'risk_committee_2024_001.pdf'
            }
        )
        
        assert exemption.category == 'extreme_market'
        assert len(exemption.approvers) == 2
        assert exemption.period[0] < exemption.period[1]
        assert 'decision_number' in exemption.audit_trail
    
    def test_uat_result_with_exemption(self):
        """测试包含业务豁免的UAT结果"""
        exemption = BusinessExemption(
            category='data_anomaly',
            reason='交易所数据源临时故障',
            approvers=['数据质量负责人'],
            period=(datetime(2024, 1, 1), datetime(2024, 1, 2)),
            audit_trail={'incident_id': 'INC-2024-001'}
        )
        
        result = UATResult(
            test_item='data_quality',
            passed=False,
            actual_value=0.85,
            threshold=0.90,
            business_exemption=exemption,
            risk_statement='数据质量暂时低于标准，但已获豁免'
        )
        
        assert result.business_exemption is not None
        assert result.business_exemption.category == 'data_anomaly'
        assert result.risk_statement is not None
    
    def test_exemption_categories(self):
        """测试所有豁免类别（专家answer.md第3轮4节）"""
        categories = [
            'extreme_market',      # 极端行情豁免
            'system_maintenance',  # 系统维护豁免
            'data_anomaly',        # 数据异常豁免
            'model_transition'     # 模型过渡豁免
        ]
        
        for category in categories:
            exemption = BusinessExemption(
                category=category,
                reason=f'{category}测试理由',
                approvers=['测试审批人'],
                period=(pd.Timestamp.now(), pd.Timestamp.now() + pd.Timedelta(days=7)),
                audit_trail={'test': True}
            )
            assert exemption.category == category
