"""
压力测试验证器测试
测试范围：
1. 历史事件加载
2. 场景验证功能
3. 验证报告生成
4. Mock依赖注入

注意：使用Mock数据提供者，真实数据集成后需补充集成测试
"""

import sys
from pathlib import Path
# 添加core_bak_refactored到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from core_bak_refactored.core.risk.stress_test_validator import (
    StressTestValidator,
    HistoricalEvent,
    ValidationResult
)
from core_bak_refactored.tests.core.risk.mocks import (
    MockHistoricalDataSource,
    MockPortfolioBuilder
)


class TestStressTestValidator:
    """测试压力测试验证器"""
    
    def test_load_validation_events(self):
        """测试事件加载"""
        validator = StressTestValidator(
            data_source=MockHistoricalDataSource(),
            portfolio_builder=MockPortfolioBuilder()
        )
        
        assert len(validator.events) == 3, "应加载3个验证事件"
        
        event_ids = [e.event_id for e in validator.events]
        assert '2015_china_market_crash' in event_ids
        assert 'covid_19_pandemic' in event_ids
        assert '2008_financial_crisis' in event_ids
    
    def test_validate_scenario_basic(self):
        """测试基本场景验证"""
        validator = StressTestValidator(
            data_source=MockHistoricalDataSource(),
            portfolio_builder=MockPortfolioBuilder()
        )
        
        result = validator.validate_scenario(
            scenario_id='2015_china_market_crash',
            stress_tester=None  # 简化版本不需要
        )
        
        assert isinstance(result, ValidationResult)
        assert result.event_id == '2015_china_market_crash'
        assert result.actual_loss != 0.0, "应有实际损失数据"
        assert result.predicted_loss != 0.0, "应有预测损失数据"
        assert result.prediction_error >= 0.0, "误差应为非负"
    
    def test_validate_all_scenarios(self):
        """测试批量场景验证"""
        validator = StressTestValidator(
            data_source=MockHistoricalDataSource(),
            portfolio_builder=MockPortfolioBuilder()
        )
        
        results = validator.validate_all_scenarios(stress_tester=None)
        
        assert len(results) == 3, "应返回3个验证结果"
        
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.actual_loss != 0.0
            assert result.predicted_loss != 0.0
    
    def test_validation_result_acceptability(self):
        """测试验证结果可接受性判断"""
        # 可接受的结果（误差≤20%）
        good_result = ValidationResult(
            event_id='test1',
            scenario_id='test_scenario',
            predicted_loss=-0.30,
            actual_loss=-0.35,
            prediction_error=0.14  # 14%误差
        )
        assert good_result.is_acceptable(threshold=0.20) is True
        
        # 不可接受的结果（误差>20%）
        bad_result = ValidationResult(
            event_id='test2',
            scenario_id='test_scenario',
            predicted_loss=-0.30,
            actual_loss=-0.50,
            prediction_error=0.40  # 40%误差
        )
        assert bad_result.is_acceptable(threshold=0.20) is False
    
    def test_generate_validation_report(self):
        """测试验证报告生成"""
        validator = StressTestValidator(
            data_source=MockHistoricalDataSource(),
            portfolio_builder=MockPortfolioBuilder()
        )
        
        results = validator.validate_all_scenarios(stress_tester=None)
        report = validator.generate_validation_report(results)
        
        assert 'total_validations' in report
        assert report['total_validations'] == 3
        assert 'acceptable_count' in report
        assert 'acceptable_rate' in report
        assert 'avg_error' in report
        assert 'results' in report
        assert len(report['results']) == 3
    
    def test_generate_validation_report_empty(self):
        """测试空结果报告生成"""
        validator = StressTestValidator(
            data_source=MockHistoricalDataSource(),
            portfolio_builder=MockPortfolioBuilder()
        )
        
        report = validator.generate_validation_report([])
        assert report['status'] == 'no_results'
    
    def test_invalid_scenario_id(self):
        """测试无效场景ID"""
        validator = StressTestValidator(
            data_source=MockHistoricalDataSource(),
            portfolio_builder=MockPortfolioBuilder()
        )
        
        with pytest.raises(ValueError, match="未找到事件"):
            validator.validate_scenario(
                scenario_id='invalid_scenario',
                stress_tester=None
            )


class TestMockDataSource:
    """测试Mock数据源"""
    
    def test_get_event_returns(self):
        """测试事件收益率获取"""
        data_source = MockHistoricalDataSource()
        
        event = HistoricalEvent(
            event_id='test_event',
            name='测试事件',
            period=('2020-01-01', '2020-01-31'),
            expected_decline=-0.30,
            scenario_params={}
        )
        
        returns = data_source.get_event_returns(event, '000300.SH')
        
        assert isinstance(returns, float)
        assert returns < 0, "应为负收益（损失）"
        # 应接近expected_decline（允许随机扰动）
        assert -0.40 < returns < -0.20


class TestMockPortfolioBuilder:
    """测试Mock组合构造器"""
    
    def test_build_test_portfolio(self):
        """测试组合构造"""
        builder = MockPortfolioBuilder()
        
        # 测试沪深300组合
        portfolio = builder.build_test_portfolio('csi300')
        assert '000300.SH' in portfolio
        assert portfolio['000300.SH'] == 1.0
        
        # 测试行业轮动组合
        portfolio = builder.build_test_portfolio('sector_rotation')
        assert len(portfolio) == 4
        assert abs(sum(portfolio.values()) - 1.0) < 1e-6  # 权重总和为1
        
        # 测试A+H混合组合
        portfolio = builder.build_test_portfolio('ah_hybrid')
        assert '000300.SH' in portfolio
        assert 'HSI' in portfolio


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
