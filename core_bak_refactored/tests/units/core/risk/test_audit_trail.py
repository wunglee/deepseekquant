"""
测试风险处理器审计跟踪功能

P0任务测试：验证RiskProcessor在process流程中正确记录审计跟踪，
包括步骤名称、时间戳、状态、输入hash、输出摘要
"""

import pytest
import hashlib
import json
from datetime import datetime
from typing import Dict, Any

from core_bak_refactored.core.risk.risk_processor import RiskProcessor
from core_bak_refactored.core.risk.risk_models import RiskLevel


@pytest.fixture
def risk_config():
    """风险处理器配置"""
    return {
        'market_type': 'CN',
        'trading_days_per_year': 252,
        'enable_audit_trail': True,
        'market_configs': {
            'CN': {
                'trading_days': 252,
                'risk_free_rate': 0.03,
                'base_currency': 'CNY'
            }
        }
    }


@pytest.fixture
def sample_risk_data():
    """样本风险数据"""
    import numpy as np
    np.random.seed(42)
    
    returns = np.random.normal(0.001, 0.02, 100)
    
    return {
        'portfolio_state': {
            'allocations': {
                'AAPL': {'weight': 0.6},
                'MSFT': {'weight': 0.4}
            }
        },
        'market_data': {
            'prices': {
                'AAPL': {
                    'close': (100 * np.cumprod(1 + returns)).tolist(),
                    'currency': 'USD'
                },
                'MSFT': {
                    'close': (150 * np.cumprod(1 + returns * 0.9)).tolist(),
                    'currency': 'USD'
                }
            },
            'risk_free_rate': 0.03
        }
    }


class TestAuditTrail:
    """审计跟踪测试"""
    
    def test_audit_trail_enabled(self, risk_config, sample_risk_data):
        """测试审计跟踪启用"""
        processor = RiskProcessor(risk_config)
        
        result = processor.process(sample_risk_data)
        
        assert result['success'], "处理应成功"
        assert 'audit_trail' in result, "结果应包含audit_trail"
        assert isinstance(result['audit_trail'], list), "audit_trail应为列表"
        assert len(result['audit_trail']) > 0, "应有审计记录"
    
    def test_audit_trail_disabled(self, risk_config, sample_risk_data):
        """测试审计跟踪禁用"""
        config = risk_config.copy()
        config['enable_audit_trail'] = False
        
        processor = RiskProcessor(config)
        result = processor.process(sample_risk_data)
        
        assert result['success']
        assert result['audit_trail'] == [], "禁用时应返回空列表"
    
    def test_audit_trail_structure(self, risk_config, sample_risk_data):
        """测试审计跟踪记录结构"""
        processor = RiskProcessor(risk_config)
        result = processor.process(sample_risk_data)
        
        audit_trail = result['audit_trail']
        
        # 验证每条记录的结构
        for entry in audit_trail:
            assert 'step' in entry, "应包含step字段"
            assert 'timestamp' in entry, "应包含timestamp字段"
            assert 'status' in entry, "应包含status字段"
            
            # 验证timestamp格式
            try:
                datetime.fromisoformat(entry['timestamp'])
            except ValueError:
                pytest.fail(f"timestamp格式无效: {entry['timestamp']}")
            
            # 验证status值
            assert entry['status'] in ['SUCCESS', 'ERROR', 'SKIP'], \
                f"status应为有效值，实际: {entry['status']}"
    
    def test_audit_trail_expected_steps(self, risk_config, sample_risk_data):
        """测试审计跟踪包含预期步骤"""
        processor = RiskProcessor(risk_config)
        result = processor.process(sample_risk_data)
        
        audit_trail = result['audit_trail']
        steps = [entry['step'] for entry in audit_trail]
        
        # 验证关键步骤存在
        expected_steps = [
            'calculator_start',
            'calculator_complete',
            'limits_check_start',
            'limits_check_complete',
            'stress_test_start',
            'stress_test_complete',
            'portfolio_analysis_start',
            'portfolio_analysis_complete',
            'assessment_generation_start',
            'assessment_generation_complete'
        ]
        
        for expected in expected_steps:
            assert expected in steps, f"应包含步骤: {expected}"
    
    def test_audit_trail_input_hash(self, risk_config, sample_risk_data):
        """测试审计跟踪输入hash"""
        processor = RiskProcessor(risk_config)
        result = processor.process(sample_risk_data)
        
        audit_trail = result['audit_trail']
        
        # calculator_start步骤应有input_hash
        calc_start = [e for e in audit_trail if e['step'] == 'calculator_start'][0]
        
        # 应包含input_hash字段
        assert 'input_hash' in calc_start
        
        # 如果有input_params，应计算hash
        if calc_start.get('input_params'):
            assert calc_start['input_hash'] is not None
            assert len(calc_start['input_hash']) == 16, "hash应为16字符"
    
    def test_audit_trail_output_summary(self, risk_config, sample_risk_data):
        """测试审计跟踪输出摘要"""
        processor = RiskProcessor(risk_config)
        result = processor.process(sample_risk_data)
        
        audit_trail = result['audit_trail']
        
        # calculator_complete步骤应有output_summary
        calc_complete = [e for e in audit_trail if e['step'] == 'calculator_complete'][0]
        
        assert 'output_summary' in calc_complete
        
        if calc_complete['output_summary']:
            summary = calc_complete['output_summary']
            assert 'type' in summary, "摘要应包含type字段"
    
    def test_audit_trail_error_handling(self, risk_config):
        """测试审计跟踪错误处理"""
        processor = RiskProcessor(risk_config)
        
        # 提供无效数据触发错误
        invalid_data = {
            'portfolio_state': None,
            'market_data': None
        }
        
        result = processor.process(invalid_data)
        
        # 即使失败也应有审计跟踪
        assert 'audit_trail' in result
        
        # 应包含error步骤
        steps = [e['step'] for e in result['audit_trail']]
        if not result['success']:
            assert any('error' in step.lower() or e['status'] == 'ERROR' 
                      for e, step in zip(result['audit_trail'], steps)), \
                "失败时应记录error步骤"
    
    def test_summarize_output_dict(self, risk_config):
        """测试输出摘要 - 字典类型"""
        processor = RiskProcessor(risk_config)
        
        test_dict = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        summary = processor._summarize_output(test_dict)
        
        assert summary['type'] == 'dict'
        assert 'keys' in summary
        assert set(summary['keys']) == {'key1', 'key2', 'key3'}
        assert summary['size'] == 3
    
    def test_summarize_output_list(self, risk_config):
        """测试输出摘要 - 列表类型"""
        processor = RiskProcessor(risk_config)
        
        # 短列表
        short_list = [1, 2, 3]
        summary = processor._summarize_output(short_list)
        
        assert summary['type'] == 'list'
        assert summary['length'] == 3
        assert summary['sample'] == [1, 2, 3]
        
        # 长列表
        long_list = list(range(100))
        summary = processor._summarize_output(long_list)
        
        assert summary['type'] == 'list'
        assert summary['length'] == 100
        assert '100 items' in str(summary['sample'])
    
    def test_summarize_output_other_types(self, risk_config):
        """测试输出摘要 - 其他类型"""
        processor = RiskProcessor(risk_config)
        
        # 字符串
        test_str = "test_string_value"
        summary = processor._summarize_output(test_str)
        
        assert summary['type'] == 'str'
        assert summary['value'] == test_str
        
        # 数值
        test_num = 42.5
        summary = processor._summarize_output(test_num)
        
        assert summary['type'] == 'float'
        assert '42.5' in summary['value']
    
    def test_audit_step_manual_call(self, risk_config):
        """测试手动调用_audit_step"""
        processor = RiskProcessor(risk_config)
        
        # 重置审计跟踪
        processor.audit_trail = []
        
        # 手动记录步骤
        processor._audit_step(
            step_name='test_step',
            input_params={'param1': 'value1'},
            output_data={'result': 'success'},
            status='SUCCESS'
        )
        
        assert len(processor.audit_trail) == 1
        entry = processor.audit_trail[0]
        
        assert entry['step'] == 'test_step'
        assert entry['status'] == 'SUCCESS'
        assert entry['input_params'] == {'param1': 'value1'}
        assert entry['output_summary'] is not None
    
    def test_audit_trail_chronological_order(self, risk_config, sample_risk_data):
        """测试审计跟踪按时间顺序"""
        processor = RiskProcessor(risk_config)
        result = processor.process(sample_risk_data)
        
        audit_trail = result['audit_trail']
        
        # 提取所有时间戳
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in audit_trail]
        
        # 验证时间戳递增
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], \
                f"时间戳应递增: {timestamps[i]} > {timestamps[i+1]}"
