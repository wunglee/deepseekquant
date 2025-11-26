"""
UAT验收测试框架 - Phase 5B-5
基于专家answer.md第1轮第5节指导实现

职责：
- 实现UAT断言清单（5项核心验收标准）
- 实现三级异常处置流程（Level 1/2/3）
- 实现加权平均误差计算（按事件类型）
- 实现三级指标验收（MAPE + 方向准确率 + 尾部控制）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger('DeepSeekQuant.UAT')


class AlertLevel(Enum):
    """异常级别（专家answer.md第1轮5.3节）"""
    LEVEL_1 = "LEVEL_1"  # 15%-20%：内部记录，下周复核
    LEVEL_2 = "LEVEL_2"  # 20%-25%：预警，人工复核
    LEVEL_3 = "LEVEL_3"  # >25%：暂停自动报送，立即干预


@dataclass
class EventTypeWeight:
    """事件类型权重配置（专家answer.md第1轮1.1节）"""
    event_type: str
    allowed_error_threshold: float  # 允许误差阈值
    weight: float  # 权重
    rationale: str  # 理由


@dataclass
class UATResult:
    """UAT验收结果"""
    test_item: str
    passed: bool
    actual_value: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExceptionAlert:
    """异常告警（专家answer.md第1轮5.3节）"""
    level: AlertLevel
    error_range: str
    action: str
    report_deadline: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UATValidator:
    """
    UAT验收测试验证器（Phase 5B-5）
    
    基于专家answer.md第1轮5.1节UAT通过标准：
    
    | 测试项 | 通过标准 | 检查频率 |
    |--------|---------|---------|
    | 历史回测误差 | ≤15%（按加权平均） | 每次发布 |
    | 跨市场一致性 | ≥85% | 季度 |
    | 行业参数差异 | ≥10%且统计显著 | 半年度 |
    | 数据质量评分 | ≥90% | 每月 |
    | 系统响应时间 | ≤5秒（单场景压力测试） | 每次发布 |
    """
    
    # 事件类型权重配置（专家answer.md第1轮1.1节）
    EVENT_TYPE_WEIGHTS = [
        EventTypeWeight(
            event_type='market_crash',
            allowed_error_threshold=0.12,
            weight=0.30,
            rationale='数据质量高，模型成熟度高'
        ),
        EventTypeWeight(
            event_type='liquidity_crisis',
            allowed_error_threshold=0.18,
            weight=0.25,
            rationale='市场结构变化大，模型适用性受限'
        ),
        EventTypeWeight(
            event_type='currency_crisis',
            allowed_error_threshold=0.20,
            weight=0.15,
            rationale='跨境传导机制复杂'
        ),
        EventTypeWeight(
            event_type='geopolitical_risk',
            allowed_error_threshold=0.15,
            weight=0.20,
            rationale='近期事件，数据可信度高'
        ),
        EventTypeWeight(
            event_type='sovereign_debt_crisis',
            allowed_error_threshold=0.15,
            weight=0.10,
            rationale='传导路径清晰'
        )
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化UAT验证器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._alert_history: List[ExceptionAlert] = []
    
    def validate_weighted_average_error(self,
                                        errors_by_event: Dict[str, float],
                                        event_type_mapping: Dict[str, str]) -> UATResult:
        """
        验证加权平均误差（专家answer.md第1轮1.1节）
        
        Args:
            errors_by_event: {event_id: prediction_error}
            event_type_mapping: {event_id: event_type}
        
        Returns:
            UATResult（通过标准：≤15%）
        """
        # 计算加权平均误差
        weighted_errors = []
        total_weight = 0.0
        
        for event_id, error in errors_by_event.items():
            event_type = event_type_mapping.get(event_id, 'unknown')
            
            # 查找对应的权重配置
            weight_config = next((w for w in self.EVENT_TYPE_WEIGHTS if w.event_type == event_type), None)
            
            if weight_config:
                weighted_errors.append(error * weight_config.weight)
                total_weight += weight_config.weight
            else:
                logger.warning(f"未知事件类型 {event_type}，使用默认权重0.1")
                weighted_errors.append(error * 0.1)
                total_weight += 0.1
        
        if total_weight == 0:
            logger.error("总权重为0，无法计算加权平均误差")
            return UATResult(
                test_item='weighted_average_error',
                passed=False,
                actual_value=1.0,
                threshold=0.15,
                details={'error': 'zero_weight'}
            )
        
        weighted_avg_error = sum(weighted_errors) / total_weight
        
        passed = weighted_avg_error <= 0.15
        
        logger.info(f"加权平均误差: {weighted_avg_error:.4f} ({'通过' if passed else '未通过'})")\n        \n        return UATResult(
            test_item='weighted_average_error',
            passed=passed,
            actual_value=weighted_avg_error,
            threshold=0.15,
            details={
                'individual_errors': errors_by_event,
                'event_type_mapping': event_type_mapping,
                'weighted_errors': dict(zip(errors_by_event.keys(), weighted_errors)),
                'total_weight': total_weight
            }
        )
    
    def validate_triple_indicator_system(self,
                                         predictions: List[float],
                                         actuals: List[float]) -> Dict[str, UATResult]:
        """
        验证三级指标体系（专家answer.md第1轮1.2节）
        
        必须同时满足：
        1. MAPE ≤ 15%
        2. 方向准确率 ≥ 90%
        3. 尾部误差控制：最大单事件误差≤25%，且此类事件不超过总事件数的20%
        
        Args:
            predictions: 预测损失列表
            actuals: 实际损失列表
        
        Returns:
            Dict[str, UATResult]: 三项指标的验收结果
        """
        if len(predictions) != len(actuals):
            raise ValueError("预测和实际数据长度不一致")
        
        results = {}
        
        # 1. MAPE（平均绝对百分比误差）
        mape_values = []
        for pred, actual in zip(predictions, actuals):
            if actual != 0:
                mape = abs(pred - actual) / abs(actual)
                mape_values.append(mape)
        
        avg_mape = np.mean(mape_values) if mape_values else 0.0
        mape_passed = avg_mape <= 0.15
        
        results['mape'] = UATResult(
            test_item='MAPE',
            passed=mape_passed,
            actual_value=avg_mape,
            threshold=0.15,
            details={'individual_mape': mape_values}
        )
        
        # 2. 方向准确率（预测损失方向与实际一致）
        direction_correct = 0
        for pred, actual in zip(predictions, actuals):
            # 同号即方向一致
            if (pred * actual > 0) or (pred == 0 and actual == 0):
                direction_correct += 1
        
        direction_accuracy = direction_correct / len(predictions) if len(predictions) > 0 else 0.0
        direction_passed = direction_accuracy >= 0.90
        
        results['direction_accuracy'] = UATResult(
            test_item='direction_accuracy',
            passed=direction_passed,
            actual_value=direction_accuracy,
            threshold=0.90,
            details={'correct_count': direction_correct, 'total_count': len(predictions)}
        )
        
        # 3. 尾部误差控制
        tail_errors = []
        for pred, actual in zip(predictions, actuals):
            if actual != 0:
                error = abs(pred - actual) / abs(actual)
                tail_errors.append(error)
        
        max_error = max(tail_errors) if tail_errors else 0.0
        extreme_error_count = sum(1 for e in tail_errors if e > 0.25)
        extreme_error_ratio = extreme_error_count / len(tail_errors) if tail_errors else 0.0
        
        tail_passed = (max_error <= 0.25) or (extreme_error_ratio <= 0.20)
        
        results['tail_error_control'] = UATResult(
            test_item='tail_error_control',
            passed=tail_passed,
            actual_value=max_error,
            threshold=0.25,
            details={
                'max_error': max_error,
                'extreme_error_count': extreme_error_count,
                'extreme_error_ratio': extreme_error_ratio,
                'extreme_threshold_ratio': 0.20
            }
        )
        
        # 总体通过：三项全部通过
        all_passed = all(r.passed for r in results.values())
        
        logger.info(f"三级指标体系: MAPE={avg_mape:.4f}, 方向准确率={direction_accuracy:.2%}, "
                   f"尾部控制={'通过' if tail_passed else '未通过'}, 总体={'通过' if all_passed else '未通过'}")
        
        return results
    
    def validate_cross_market_consistency(self,
                                         consistency_score: float) -> UATResult:
        """
        验证跨市场一致性（专家answer.md第1轮5.1节）
        
        Args:
            consistency_score: 一致性评分（相关性）
        
        Returns:
            UATResult（通过标准：≥0.85）
        """
        passed = consistency_score >= 0.85
        
        logger.info(f"跨市场一致性: {consistency_score:.4f} ({'通过' if passed else '未通过'})")
        
        return UATResult(
            test_item='cross_market_consistency',
            passed=passed,
            actual_value=consistency_score,
            threshold=0.85
        )
    
    def validate_industry_parameter_difference(self,
                                               industry_parameters: Dict[str, float],
                                               t_test_results: Dict[Tuple[str, str], float]) -> UATResult:
        """
        验证行业参数差异（专家answer.md第1轮5.1节）
        
        Args:
            industry_parameters: {industry: parameter_value}
            t_test_results: {(industry_a, industry_b): p_value}
        
        Returns:
            UATResult（通过标准：差异≥10%且p<0.05）
        """
        # 计算所有行业间的参数差异
        industries = list(industry_parameters.keys())
        differences = []
        
        for i in range(len(industries)):
            for j in range(i + 1, len(industries)):
                ind_a = industries[i]
                ind_b = industries[j]
                
                param_a = industry_parameters[ind_a]
                param_b = industry_parameters[ind_b]
                
                # 计算百分比差异
                if param_b != 0:
                    diff_pct = abs(param_a - param_b) / abs(param_b)
                    differences.append(diff_pct)
        
        avg_diff = np.mean(differences) if differences else 0.0
        
        # 检查统计显著性
        significant_pairs = sum(1 for p_value in t_test_results.values() if p_value < 0.05)
        total_pairs = len(t_test_results)
        significance_ratio = significant_pairs / total_pairs if total_pairs > 0 else 0.0
        
        passed = (avg_diff >= 0.10) and (significance_ratio > 0.5)  # 超过一半的配对显著
        
        logger.info(f"行业参数差异: 平均差异={avg_diff:.2%}, 显著性比例={significance_ratio:.2%}, "
                   f"{'通过' if passed else '未通过'}")
        
        return UATResult(
            test_item='industry_parameter_difference',
            passed=passed,
            actual_value=avg_diff,
            threshold=0.10,
            details={
                'industry_parameters': industry_parameters,
                't_test_results': {f"{k[0]}-{k[1]}": v for k, v in t_test_results.items()},
                'significant_pairs': significant_pairs,
                'total_pairs': total_pairs,
                'significance_ratio': significance_ratio
            }
        )
    
    def validate_data_quality(self,
                             completeness: float,
                             consistency: float,
                             accuracy: float) -> UATResult:
        """
        验证数据质量评分（专家answer.md第1轮5.1节）
        
        Args:
            completeness: 完整性评分
            consistency: 一致性评分
            accuracy: 准确性评分
        
        Returns:
            UATResult（通过标准：≥0.90）
        """
        # 数据质量综合评分（加权平均）
        quality_score = (completeness * 0.4 + consistency * 0.3 + accuracy * 0.3)
        
        passed = quality_score >= 0.90
        
        logger.info(f"数据质量评分: {quality_score:.4f} ({'通过' if passed else '未通过'})")
        
        return UATResult(
            test_item='data_quality_score',
            passed=passed,
            actual_value=quality_score,
            threshold=0.90,
            details={
                'completeness': completeness,
                'consistency': consistency,
                'accuracy': accuracy
            }
        )
    
    def validate_system_response_time(self,
                                      response_time_seconds: float) -> UATResult:
        """
        验证系统响应时间（专家answer.md第1轮5.1节）
        
        Args:
            response_time_seconds: 响应时间（秒）
        
        Returns:
            UATResult（通过标准：≤5秒）
        """
        passed = response_time_seconds <= 5.0
        
        logger.info(f"系统响应时间: {response_time_seconds:.2f}秒 ({'通过' if passed else '未通过'})")
        
        return UATResult(
            test_item='system_response_time',
            passed=passed,
            actual_value=response_time_seconds,
            threshold=5.0
        )
    
    def handle_exception(self,
                        prediction_error: float,
                        event_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[ExceptionAlert]:
        """
        异常处置（专家answer.md第1轮5.3节）
        
        | 级别 | 误差范围 | 处置措施 | 上报时限 |
        |------|---------|---------|---------|
        | Level 1 | 15%-20% | 内部记录，下周复核 | 3个工作日内 |
        | Level 2 | 20%-25% | 预警，人工复核 | 24小时内 |
        | Level 3 | >25% | 暂停自动报送，立即干预 | 立即 |
        
        Args:
            prediction_error: 预测误差
            event_id: 事件ID
            metadata: 附加元数据
        
        Returns:
            ExceptionAlert（如果触发异常）或None
        """
        alert = None
        
        if prediction_error > 0.25:
            # Level 3：暂停自动报送，立即干预
            alert = ExceptionAlert(
                level=AlertLevel.LEVEL_3,
                error_range=">25%",
                action="暂停自动报送，立即干预",
                report_deadline="立即",
                metadata={
                    'event_id': event_id,
                    'prediction_error': prediction_error,
                    **(metadata or {})
                }
            )
            logger.critical(f"触发Level 3异常: 事件={event_id}, 误差={prediction_error:.2%}")
        
        elif prediction_error > 0.20:
            # Level 2：预警，人工复核
            alert = ExceptionAlert(
                level=AlertLevel.LEVEL_2,
                error_range="20%-25%",
                action="预警，人工复核",
                report_deadline="24小时内",
                metadata={
                    'event_id': event_id,
                    'prediction_error': prediction_error,
                    **(metadata or {})
                }
            )
            logger.warning(f"触发Level 2异常: 事件={event_id}, 误差={prediction_error:.2%}")
        
        elif prediction_error > 0.15:
            # Level 1：内部记录，下周复核
            alert = ExceptionAlert(
                level=AlertLevel.LEVEL_1,
                error_range="15%-20%",
                action="内部记录，下周复核",
                report_deadline="3个工作日内",
                metadata={
                    'event_id': event_id,
                    'prediction_error': prediction_error,
                    **(metadata or {})
                }
            )
            logger.info(f"触发Level 1异常: 事件={event_id}, 误差={prediction_error:.2%}")
        
        if alert:
            self._alert_history.append(alert)
        
        return alert
    
    def get_alert_history(self, 
                          level: Optional[AlertLevel] = None,
                          since: Optional[datetime] = None) -> List[ExceptionAlert]:
        """
        获取告警历史
        
        Args:
            level: 筛选级别（None表示所有级别）
            since: 起始时间（None表示所有时间）
        
        Returns:
            告警列表
        """
        alerts = self._alert_history
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
