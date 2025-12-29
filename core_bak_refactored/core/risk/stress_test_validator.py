"""
压力测试验证器 - 风险模块专属
从backtest_framework.py重构，仅保留风险模块职责部分

职责范围：
- 压力测试场景的历史有效性验证
- 场景参数准确性评估
- 损失预测误差统计

非职责（已拆分到其他模块）：
- 数据获取 → core/data/_fragments/historical_data_provider.py
- 组合构造 → core/portfolio/_fragments/synthetic_portfolio_builder.py
- 回测引擎 → core/backtest/_fragments/event_window_backtester.py
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Protocol

import pandas as pd

logger = logging.getLogger('DeepSeekQuant.StressTestValidator')


# =============================================================================
# 数据模型（风险模块专属）
# =============================================================================

@dataclass
class HistoricalEvent:
    """
    历史压力事件定义（风险场景验证专用）
    
    用途：定义需要验证的历史极端事件
    示例：2015中国股灾、2008金融危机等
    """
    event_id: str
    name: str
    period: tuple  # (start_date, end_date)
    expected_decline: float
    scenario_params: Dict[str, Any]  # 对应StressTester的场景参数
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    场景验证结果（风险模块专属）
    
    用途：记录压力测试场景的历史验证结果
    """
    event_id: str
    scenario_id: str
    predicted_loss: float  # StressTester预测的损失
    actual_loss: float  # 历史实际损失
    prediction_error: float  # abs(predicted - actual) / abs(actual)
    validation_date: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_acceptable(self, threshold: float = 0.20) -> bool:
        """判断预测误差是否在可接受范围内（默认≤20%）"""
        return self.prediction_error <= threshold


# =============================================================================
# 协议接口（定义外部依赖，待未来模块实现）
# =============================================================================

class HistoricalDataSource(Protocol):
    """
    历史数据源协议（待core/data模块实现）
    
    状态：功能碎片接口
    实现位置：core/data/_fragments/historical_data_provider.py
    """
    
    def get_event_returns(self, event: HistoricalEvent, asset_id: str) -> float:
        """
        获取事件期间的实际收益率
        
        Args:
            event: 历史事件定义
            asset_id: 资产标识（如'000300.SH'）
        
        Returns:
            float: 事件期间的总收益率
        """
        ...


class PortfolioBuilder(Protocol):
    """
    组合构造器协议（待core/portfolio模块实现）
    
    状态：功能碎片接口
    实现位置：core/portfolio/_fragments/synthetic_portfolio_builder.py
    """
    
    def build_test_portfolio(self, portfolio_type: str) -> Dict[str, float]:
        """
        构造测试组合
        
        Args:
            portfolio_type: 组合类型（'csi300'/'sector_rotation'/'ah_hybrid'）
        
        Returns:
            Dict[str, float]: {asset_id: weight}
        """
        ...


# =============================================================================
# 压力测试验证器（风险模块核心实现）
# =============================================================================

class StressTestValidator:
    """
    压力测试验证器
    
    职责：
    1. 验证压力测试场景的历史准确性
    2. 计算预测误差并生成验证报告
    3. 为压力测试器提供参数校准建议
    
    非职责（已拆分）：
    - 历史数据获取（依赖HistoricalDataSource接口）
    - 组合构造（依赖PortfolioBuilder接口）
    - 通用回测引擎（属于core/backtest模块）
    
    设计模式：
    - 依赖注入：通过构造函数注入外部依赖
    - 接口抽象：通过Protocol定义依赖接口，降低耦合
    """
    
    def __init__(self, 
                 data_source: HistoricalDataSource,
                 portfolio_builder: PortfolioBuilder):
        """
        初始化验证器
        
        Args:
            data_source: 历史数据源（依赖注入，支持Mock和真实实现）
            portfolio_builder: 组合构造器（依赖注入）
        """
        self.data_source = data_source
        self.portfolio_builder = portfolio_builder
        self.events = self._load_validation_events()
    
    def _load_validation_events(self) -> List[HistoricalEvent]:
        """
        加载需要验证的历史事件清单
        
        基于专家answer.md第5轮1.3节，选择3个核心事件作为MVP
        """
        return [
            HistoricalEvent(
                event_id='2015_china_market_crash',
                name='2015中国股灾',
                period=('2015-06-15', '2015-08-26'),
                expected_decline=-0.43,
                scenario_params={'decline': -0.30, 'liquidity_dry_up': 0.8}
            ),
            HistoricalEvent(
                event_id='covid_19_pandemic',
                name='COVID-19疫情',
                period=('2020-02-20', '2020-03-23'),
                expected_decline=-0.34,
                scenario_params={'decline': -0.20, 'volatility_spike': 2.0}
            ),
            HistoricalEvent(
                event_id='2008_financial_crisis',
                name='2008金融危机',
                period=('2008-09-15', '2008-11-20'),
                expected_decline=-0.40,
                scenario_params={'decline': -0.40, 'volatility_spike': 3.5}
            )
        ]
    
    def validate_scenario(self, 
                         scenario_id: str,
                         stress_tester,
                         benchmark_asset: str = '000300.SH') -> ValidationResult:
        """
        验证单个压力场景的历史准确性
        
        Args:
            scenario_id: 场景标识（必须匹配HistoricalEvent.event_id）
            stress_tester: 压力测试器实例（用于获取预测损失）
            benchmark_asset: 基准资产（用于计算实际损失）
        
        Returns:
            ValidationResult: 验证结果
        
        Raises:
            ValueError: 如果scenario_id未找到
        """
        # 查找对应事件
        event = next((e for e in self.events if e.event_id == scenario_id), None)
        if not event:
            raise ValueError(f"未找到事件: {scenario_id}")
        
        try:
            # 1. 获取实际历史损失（通过数据源）
            actual_loss = self.data_source.get_event_returns(event, benchmark_asset)
            
            # 2. 获取压力测试器预测损失
            # 注：当前简化版本，未来需要构造完整的组合状态和市场数据
            predicted_loss = event.scenario_params.get('decline', 0.0)
            
            # 3. 计算预测误差
            if actual_loss != 0:
                prediction_error = abs(predicted_loss - actual_loss) / abs(actual_loss)
            else:
                prediction_error = 0.0
            
            result = ValidationResult(
                event_id=event.event_id,
                scenario_id=scenario_id,
                predicted_loss=predicted_loss,
                actual_loss=actual_loss,
                prediction_error=prediction_error,
                metadata={
                    'event_name': event.name,
                    'period': event.period,
                    'benchmark_asset': benchmark_asset
                }
            )
            
            logger.info(
                f"场景验证完成: {event.name}, "
                f"预测={predicted_loss:.2%}, 实际={actual_loss:.2%}, "
                f"误差={prediction_error:.2%}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"场景验证失败: {scenario_id}, 错误: {e}", exc_info=True)
            raise
    
    def validate_all_scenarios(self, stress_tester) -> List[ValidationResult]:
        """
        验证所有历史事件场景
        
        Args:
            stress_tester: 压力测试器实例
        
        Returns:
            List[ValidationResult]: 所有验证结果
        """
        results = []
        for event in self.events:
            try:
                result = self.validate_scenario(
                    scenario_id=event.event_id,
                    stress_tester=stress_tester
                )
                results.append(result)
            except Exception as e:
                logger.error(f"验证事件 {event.name} 失败: {e}")
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        生成验证报告（统计摘要）
        
        Args:
            results: 验证结果列表
        
        Returns:
            Dict: 包含统计指标的报告
        """
        if not results:
            return {'status': 'no_results'}
        
        errors = [r.prediction_error for r in results]
        acceptable_count = sum(1 for r in results if r.is_acceptable())
        
        report = {
            'total_validations': len(results),
            'acceptable_count': acceptable_count,
            'acceptable_rate': acceptable_count / len(results),
            'avg_error': sum(errors) / len(errors),
            'max_error': max(errors),
            'min_error': min(errors),
            'results': [
                {
                    'event_id': r.event_id,
                    'predicted': f"{r.predicted_loss:.2%}",
                    'actual': f"{r.actual_loss:.2%}",
                    'error': f"{r.prediction_error:.2%}",
                    'acceptable': r.is_acceptable()
                }
                for r in results
            ]
        }
        
        return report


# =============================================================================
# 功能碎片占位符（待未来模块实现）
# =============================================================================

# Deleted:class MockHistoricalDataSource:
# Deleted:    """
# Deleted:    模拟历史数据源（临时实现）
# Deleted:    
# Deleted:    状态：功能碎片
# Deleted:    迁移目标：core/data/_fragments/historical_data_provider.py
# Deleted:    
# Deleted:    警告：仅用于框架验证，生产环境需替换为真实数据源
# Deleted:    """
# Deleted:    
# Deleted:    def get_event_returns(self, event: HistoricalEvent, asset_id: str) -> float:
# Deleted:        """返回模拟的事件收益率（基于expected_decline）"""
# Deleted:        # 简化：直接返回预期下跌，实际应从真实数据计算
# Deleted:        import random
# Deleted:        # 添加随机扰动模拟真实数据偏差
# Deleted:        noise = random.uniform(-0.1, 0.1) * event.expected_decline
# Deleted:        return event.expected_decline + noise


# Deleted:class MockPortfolioBuilder:
# Deleted:    """
# Deleted:    模拟组合构造器（临时实现）
# Deleted:    
# Deleted:    状态：功能碎片
# Deleted:    迁移目标：core/portfolio/_fragments/synthetic_portfolio_builder.py
# Deleted:    """
# Deleted:    
# Deleted:    def build_test_portfolio(self, portfolio_type: str) -> Dict[str, float]:
# Deleted:        """返回简化的测试组合"""
# Deleted:        if portfolio_type == 'csi300':
# Deleted:            return {'000300.SH': 1.0}
# Deleted:        elif portfolio_type == 'sector_rotation':
# Deleted:            return {'finance': 0.30, 'consumer': 0.25, 'tech': 0.20, 'other': 0.25}
# Deleted:        elif portfolio_type == 'ah_hybrid':
# Deleted:            return {'000300.SH': 0.70, 'HSI': 0.30}
# Deleted:        else:
# Deleted:            return {'000300.SH': 1.0}
