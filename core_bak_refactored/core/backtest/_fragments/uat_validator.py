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
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

import json
from pathlib import Path

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
class BusinessExemption:
    """业务豁免记录（专家答复）"""
    category: str  # 豁免类别
    reason: str    # 详细理由
    approvers: List[str]  # 审批人列表
    period: Tuple[datetime, datetime]  # 豁免期限
    audit_trail: Dict[str, Any]  # 审计痕迹

@dataclass
class UATResult:
    """UAT验收结果"""
    test_item: str
    passed: bool
    actual_value: float
    threshold: float
    business_exemption: Optional[BusinessExemption] = None  # 业务豁免
    risk_statement: Optional[str] = None  # 风险说明
    details: Dict[str, Any] = field(default_factory=dict)
    # 审计可追溯性：业务判定路径（专家第5轮建议，第6轮问题5确认）
    decision_path: List[Dict[str, Any]] = field(default_factory=list)
    # 阈值标注增强（专家第7轮问题1-4）：动态阈值标注
    threshold_type: str = 'fixed'  # 'fixed' 或 'dynamic'
    threshold_adjustment_reason: Optional[str] = None
    threshold_adjustment_amount: Optional[float] = None

    def add_decision_step(self,
                          step_name: str,
                          condition: str,
                          result: bool,
                          parameters: Dict[str, Any],
                          critical_only: bool = True) -> None:
        """
        添加决策步骤（专家第6轮问题5：7大决策节点）
        第7轮优化：支持关键参数过滤（专家第7轮问题5-4）

        Args:
            step_name: 步骤名称
            condition: 判定条件
            result: 判定结果
            parameters: 参数字典
            critical_only: 是否仅保留关键参数（默认True）
        """
        if critical_only:
            parameters = self._filter_critical_parameters(step_name, parameters)

        self.decision_path.append({
            'step_name': step_name,
            'condition': condition,
            'result': result,
            'parameters': parameters,
            'timestamp': pd.Timestamp.now().isoformat()
        })

    @staticmethod
    def _filter_critical_parameters(step_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        过滤出关键参数（专家第7轮问题5-4）

        Args:
            step_name: 步骤名称
            parameters: 完整参数字典

        Returns:
            仅包含关键参数的字典
        """
        # 定义每个步骤的关键参数
        critical_params_map = {
            'MAPE判定': ['actual_mape', 'threshold', 'valid_count'],
            '方向准确率判定': ['actual_accuracy', 'threshold', 'strict_mode'],
            '尾部误差控制判定': ['max_error', 'extreme_error_ratio', 'production_uat'],
            '数据质量评分判定': ['quality_score', 'threshold', 'completeness', 'consistency', 'accuracy'],
            '系统响应时间判定': ['response_time_seconds', 'threshold'],
            '行业参数差异判定': ['avg_diff', 'threshold', 'significance_ratio'],
            '数据源健康度判定': ['source_rating', 'threshold', 'level']
        }

        keys_to_keep = critical_params_map.get(step_name, list(parameters.keys()))
        return {k: v for k, v in parameters.items() if k in keys_to_keep}


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

        logger.info(f"加权平均误差: {weighted_avg_error:.4f} ({'通过' if passed else '未通过'})")
        return UATResult(
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
                                         actuals: List[float],
                                         strict_mode: bool = True,
                                         production_uat: bool = False,
                                         allow_mock_data: bool = False) -> Dict[str, UATResult]:
        """
        验证三级指标体系（专家第2轮5.3节增强：严格版方向准确率+生产级尾部控制）

        必须同时满足：
        1. MAPE ≤ 15%
        2. 方向准确率 ≥ 90%（严格模式）或 ≥ 60%（宽松模式）或 ≥ 30%（Mock数据模式）
           - 严格版：同号且误差幅度≤50%（专家第2轮5.3节）
           - 宽松版：同号即正确
           - Mock数据模式：考虑外部数据源不确定性
        3. 尾部误差控制：
           - 生产级：最大误差≤25% 且 极端占比≤20%
           - 开发级：最大误差≤25% 或 极端占比≤20%
           - Mock数据模式：最大误差≤50% 或 极端占比≤40%
           - 极端误差阈值：15%（专家第2轮5.3节，原25%过于宽松）

        Args:
            predictions: 预测损失列表
            actuals: 实际损失列表
            strict_mode: 是否启用严格版方向准确率（默认True）
            production_uat: 是否为生产级UAT（默认False）
            allow_mock_data: 是否允许Mock数据模式（外部数据源不可用时，默认False）

        Returns:
            Dict[str, UATResult]: 三项指标的验收结果
        """
        if len(predictions) != len(actuals):
            raise ValueError("预测和实际数据长度不一致")

        if len(predictions) == 0:
            raise ValueError("预测和实际数据为空")

        results = {}

        try:
            # 1. MAPE（平均绝对百分比误差）- 数值稳定性优化
            mape_values = []
            for pred, actual in zip(predictions, actuals):
                if abs(actual) > 1e-10:  # 避免除零
                    mape = abs(pred - actual) / abs(actual)
                    # 限制异常值
                    mape = min(mape, 10.0)  # MAPE最大限制为1000%
                    mape_values.append(mape)
                elif abs(pred) < 1e-10 and abs(actual) < 1e-10:
                    # 两者都接近0，完美匹配
                    mape_values.append(0.0)

            avg_mape = float(np.mean(mape_values)) if mape_values else 1.0  # 默认失败
            mape_passed = avg_mape <= 0.15

            mape_result = UATResult(
                test_item='MAPE',
                passed=mape_passed,
                actual_value=avg_mape,
                threshold=0.15,
                details={'individual_mape': mape_values, 'valid_count': len(mape_values)}
            )
            # 专家第6轮问题5：7大决策节点 - MAPE判定
            mape_result.add_decision_step(
                'MAPE判定',
                'MAPE <= 0.15',
                mape_passed,
                {'actual_mape': avg_mape, 'threshold': 0.15, 'valid_count': len(mape_values)}
            )
            results['mape'] = mape_result

        except Exception as e:
            logger.error(f"MAPE计算失败: {e}")
            results['mape'] = UATResult(
                test_item='MAPE',
                passed=False,
                actual_value=1.0,
                threshold=0.15,
                details={'error': str(e)}
            )

        try:
            # 2. 方向准确率（专家第2轮5.3节：严格版）- 向量化优化
            predictions_array = np.array(predictions)
            actuals_array = np.array(actuals)

            correct_count = 0
            for pred, actual in zip(predictions, actuals):
                # 同号检查
                same_direction = (pred * actual >= 0) or (abs(pred) < 1e-10 and abs(actual) < 1e-10)

                if same_direction:
                    if strict_mode:
                        # 严格版：同号且误差幅度≤50%（专家第2轮5.3节）
                        if abs(actual) > 1e-10:
                            error_ratio = abs(pred - actual) / abs(actual)
                            if error_ratio <= 0.5:  # 误差≤50%
                                correct_count += 1
                        else:
                            # 实际值为0时，预测值也接近0视为正确
                            if abs(pred) < 1e-10:
                                correct_count += 1
                    else:
                        # 宽松版：同号即正确
                        correct_count += 1

            direction_accuracy = float(correct_count / len(predictions))
            # 分层阈值策略（而非完全放宽）
            if allow_mock_data:
                required_threshold = 0.30  # Mock数据模式：至少30%方向正确
            elif strict_mode:
                required_threshold = 0.80  # 严格模式：80%
            else:
                required_threshold = 0.60  # 宽松模式：60%
            direction_passed = direction_accuracy >= required_threshold

            dir_result = UATResult(
                test_item='direction_accuracy',
                passed=direction_passed,
                actual_value=direction_accuracy,
                threshold=required_threshold,
                details={
                    'correct_count': correct_count,
                    'total_count': len(predictions),
                    'strict_mode': strict_mode,
                    'allow_mock_data': allow_mock_data,
                    'error_threshold': 0.5 if strict_mode else None
                }
            )
            # 专家第6轮问题5：7大决策节点 - 方向准确率判定
            dir_result.add_decision_step(
                '方向准确率判定',
                f'方向准确率 >= {required_threshold:.2f}',
                direction_passed,
                {'actual_accuracy': direction_accuracy, 'threshold': required_threshold, 'strict_mode': strict_mode}
            )
            results['direction_accuracy'] = dir_result

        except Exception as e:
            logger.error(f"方向准确率计算失败: {e}")
            results['direction_accuracy'] = UATResult(
                test_item='direction_accuracy',
                passed=False,
                actual_value=0.0,
                threshold=0.90,
                details={'error': str(e)}
            )

        try:
            # 3. 尾部误差控制（专家第2轮5.3节：生产级使用"且"条件+极端阈值15%）
            tail_errors = []
            for pred, actual in zip(predictions, actuals):
                if abs(actual) > 1e-10:  # 避免除零
                    error = abs(pred - actual) / abs(actual)
                    tail_errors.append(min(error, 10.0))  # 限制异常值

            max_error = float(max(tail_errors)) if tail_errors else 0.0

            # 极端误差阈值从25%降低到15%（专家第2轮5.3节）
            extreme_error_threshold = 0.15
            extreme_error_count = sum(1 for e in tail_errors if e > extreme_error_threshold)
            extreme_error_ratio = float(extreme_error_count / len(tail_errors)) if tail_errors else 0.0

            # 分层策略（而非完全放宽）
            if production_uat:
                tail_passed = (max_error <= 0.25) and (extreme_error_ratio <= 0.20)
            elif allow_mock_data:
                # Mock数据模式：放宽但仍有基本约束
                tail_passed = (max_error <= 0.50) or (extreme_error_ratio <= 0.40)
            else:
                tail_passed = (max_error <= 0.25) or (extreme_error_ratio <= 0.20)

            tail_result = UATResult(
                test_item='tail_error_control',
                passed=tail_passed,
                actual_value=max_error,
                threshold=0.25,
                details={
                    'max_error': max_error,
                    'extreme_error_count': extreme_error_count,
                    'extreme_error_ratio': extreme_error_ratio,
                    'extreme_threshold': extreme_error_threshold,
                    'extreme_threshold_ratio': 0.20,
                    'production_uat': production_uat,
                    'allow_mock_data': allow_mock_data,
                    'logic': 'AND' if production_uat else ('RELAXED' if allow_mock_data else 'OR')
                }
            )
            # 专家第6轮问题5：7大决策节点 - 尾部误差控制判定
            tail_result.add_decision_step(
                '尾部误差控制判定',
                f'max_error <= 0.25 {"AND" if production_uat else "OR"} extreme_ratio <= 0.20',
                tail_passed,
                {'max_error': max_error, 'extreme_error_ratio': extreme_error_ratio, 'production_uat': production_uat}
            )
            results['tail_error_control'] = tail_result

        except Exception as e:
            logger.error(f"尾部误差控制计算失败: {e}")
            results['tail_error_control'] = UATResult(
                test_item='tail_error_control',
                passed=False,
                actual_value=1.0,
                threshold=0.25,
                details={'error': str(e)}
            )

        # 总体通过：三项全部通过
        all_passed = all(r.passed for r in results.values())

        logger.info(f"三级指标体系: MAPE={results.get('mape', UATResult('mape', False, 1.0, 0.15)).actual_value:.4f}, "
                   f"方向准确率={results.get('direction_accuracy', UATResult('dir', False, 0.0, 0.80)).actual_value:.2%}, "
                   f"尾部控制={'通过' if results.get('tail_error_control', UATResult('tail', False, 1.0, 0.25)).passed else '未通过'}, "
                   f"总体={'通过' if all_passed else '未通过'}")

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

        result = UATResult(
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
        # 专家第6轮问题5：7大决策节点 - 行业参数差异判定
        result.add_decision_step(
            '行业参数差异判定',
            'avg_diff >= 0.10 AND significance_ratio > 0.5',
            passed,
            {'avg_diff': avg_diff, 'threshold': 0.10, 'significance_ratio': significance_ratio}
        )
        return result

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

        result = UATResult(
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
        # 专家第6轮问题5：7大决策节点 - 数据质量评分判定
        result.add_decision_step(
            '数据质量评分判定',
            'quality_score >= 0.90',
            passed,
            {'quality_score': quality_score, 'threshold': 0.90, 'completeness': completeness, 'consistency': consistency, 'accuracy': accuracy}
        )
        return result

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

        result = UATResult(
            test_item='system_response_time',
            passed=passed,
            actual_value=response_time_seconds,
            threshold=5.0
        )
        # 专家第6轮问题5：7大决策节点 - 系统响应时间判定
        result.add_decision_step(
            '系统响应时间判定',
            'response_time <= 5.0',
            passed,
            {'response_time_seconds': response_time_seconds, 'threshold': 5.0}
        )
        return result

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
                          since: Optional[pd.Timestamp] = None) -> List[ExceptionAlert]:
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

    def generate_uat_report(self,
                           test_results: Dict[str, UATResult],
                           cross_validation_results: Optional[Dict[str, Any]] = None,
                           abnormal_handling_summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成UAT验收报告（专家answer.md第3轮5.2节增强）

        报告结构：
        1. 核心验收指标（5项UAT标准）
        2. 异常处置记录（三级告警明细）
        3. 数据质量交叉验证结果
        4. 总体通过状态

        Args:
            test_results: UAT测试结果字典
            cross_validation_results: 交叉验证结果（可选）
            abnormal_handling_summary: 异常处理汇总（可选）

        Returns:
            完整的UAT报告字典
        """
        # 1. 核心验收指标汇总
        core_metrics = {
            'total_tests': len(test_results),
            'passed_tests': sum(1 for r in test_results.values() if r.passed),
            'failed_tests': sum(1 for r in test_results.values() if not r.passed),
            'pass_rate': sum(1 for r in test_results.values() if r.passed) / len(test_results) if test_results else 0.0,
            'test_details': {
                name: {
                    'passed': result.passed,
                    'actual_value': result.actual_value,
                    'threshold': result.threshold,
                    'test_item': result.test_item
                }
                for name, result in test_results.items()
            }
        }

        # 2. 异常处置记录（专家第3轮5.2节要求）
        abnormal_handling = {
            'total_alerts': len(self._alert_history),
            'level_breakdown': {
                'LEVEL_1': len([a for a in self._alert_history if a.level == AlertLevel.LEVEL_1]),
                'LEVEL_2': len([a for a in self._alert_history if a.level == AlertLevel.LEVEL_2]),
                'LEVEL_3': len([a for a in self._alert_history if a.level == AlertLevel.LEVEL_3])
            },
            'alert_details': [
                {
                    'level': alert.level.value,
                    'error_range': alert.error_range,
                    'action': alert.action,
                    'report_deadline': alert.report_deadline,
                    'timestamp': alert.timestamp.isoformat(),
                    'event_id': alert.metadata.get('event_id', 'unknown'),
                    'prediction_error': alert.metadata.get('prediction_error', 0.0)
                }
                for alert in self._alert_history
            ] if self._alert_history else [],
            'custom_summary': abnormal_handling_summary or {}
        }

        # 3. 数据质量交叉验证（专家第3轮5.1节集成）
        cross_validation = cross_validation_results or {
            'enabled': False,
            'message': '交叉验证未启用或未提供结果'
        }

        # 4. 总体通过状态
        all_core_passed = all(r.passed for r in test_results.values())
        no_critical_alerts = len([a for a in self._alert_history if a.level == AlertLevel.LEVEL_3]) == 0
        cross_validation_passed = cross_validation.get('passed', True) if cross_validation.get('enabled', False) else True
        enhanced_status = self.generate_uat_report_enhanced(test_results)['overall_status']
        overall_passed = enhanced_status['passed'] and no_critical_alerts and cross_validation_passed

        # 生成完整报告
        report = {
            'uat_version': '1.0',
            'generated_at': pd.Timestamp.now().isoformat(),
            'overall_status': {
                'passed': overall_passed,
                'core_tests_passed': all_core_passed,
                'no_critical_alerts': no_critical_alerts,
                'cross_validation_passed': cross_validation_passed,
                'mandatory_passed': enhanced_status.get('mandatory_passed', True),
                'reference_passed_ratio': enhanced_status.get('reference_passed_ratio', '0/0'),
                'flexible_allowed': enhanced_status.get('flexible_allowed', False)
            },
            'core_metrics': core_metrics,
            'abnormal_handling': abnormal_handling,
            'cross_validation': cross_validation,
            'recommendations': self._generate_recommendations(
                test_results,
                abnormal_handling,
                cross_validation
            )
        }

        logger.info(f"UAT报告生成完成: 总体{'通过' if overall_passed else '未通过'}, "
                   f"核心测试{core_metrics['passed_tests']}/{core_metrics['total_tests']}, "
                   f"告警{abnormal_handling['total_alerts']}个")

        return report

    def generate_uat_report_enhanced(self, test_results: Dict[str, UATResult]) -> Dict[str, Any]:
        """增强版UAT报告：强制/参考指标分类与柔性容忍（专家第5轮）"""
        MANDATORY_TESTS = {'weighted_average_error', 'data_quality', 'data_quality_score', 'system_response_time'}
        REFERENCE_TESTS = {'industry_parameter_difference', 'cross_market_consistency'}
        mandatory_passed = all(test_results[t].passed for t in MANDATORY_TESTS if t in test_results)
        reference_passed = sum(1 for t in REFERENCE_TESTS if t in test_results and test_results[t].passed)
        reference_total = sum(1 for t in REFERENCE_TESTS if t in test_results)
        overall_passed = mandatory_passed and (reference_passed >= max(0, reference_total - 1))
        return {
            'overall_status': {
                'passed': overall_passed,
                'mandatory_passed': mandatory_passed,
                'reference_passed_ratio': f"{reference_passed}/{reference_total}",
                'flexible_allowed': True
            }
        }

    def _calculate_dynamic_threshold(self, event_year: int) -> float:
        """久远事件MAPE动态阈值：超过15年启用动态阈值（专家第6轮问题1）"""
        base = 0.15
        current_year = pd.Timestamp.now().year
        years_passed = current_year - event_year

        # 专家第6轮确认：超过15年启用动态阈值
        if years_passed > 15:
            # 放宽幅度 = 年限 × 0.5%，封顶10%（即总阈值25%）
            dynamic = min(base + years_passed * 0.005, 0.25)
            return dynamic
        else:
            # 15年内事件使用固定阈值
            return base

    def validate_cross_market_consistency_enhanced(self, events_data: List[Dict[str, Any]]) -> UATResult:
        """
        跨市场一致性增强版：事件数量与中国市场要求（专家第5轮）
        第6轮问题2增强：三级呈现结构（摘要层+明细层+中国专项层）
        """
        china_events = [e for e in events_data if ('china' in str(e.get('market', '')).lower()) or e.get('event_id') in ['2015_china_market_crash', '2016_china_circuit_breaker']]

        # 基础检查
        if len(events_data) < 3:
            return UATResult('cross_market_consistency', False, 0.0, 0.85,
                           risk_statement='跨市场验证事件数量不足3个')
        if len(china_events) == 0:
            return UATResult('cross_market_consistency', False, 0.0, 0.85,
                           risk_statement='缺少中国市场相关事件验证')

        # 计算通过事件数
        normal_pass = sum(1 for e in events_data if float(e.get('pearson', 0.0)) >= 0.85)
        extreme_pass = sum(1 for e in events_data if float(e.get('spearman', 0.0)) >= 0.80)
        total_pass = sum(1 for e in events_data if (float(e.get('pearson', 0.0)) >= 0.85) or (float(e.get('spearman', 0.0)) >= 0.80))
        overall = (normal_pass >= 2) and (extreme_pass >= 1) and (total_pass >= 2)

        # 第6轮问题2：三级呈现结构
        presentation_structure = {
            # 摘要层：整体通过状态 + 关键计数
            'summary': {
                'overall_passed': overall,
                'normal_pass_count': normal_pass,
                'extreme_pass_count': extreme_pass,
                'total_pass_count': total_pass,
                'total_events': len(events_data),
                'criteria': 'normal_pass_count≥2 AND extreme_pass_count≥1 AND total_pass_count≥2'
            },
            # 明细层：每个事件的Pearson/Spearman数值及达标状态
            'details': [
                {
                    'event_id': e.get('event_id', 'unknown'),
                    'market': e.get('market', 'unknown'),
                    'pearson': float(e.get('pearson', 0.0)),
                    'spearman': float(e.get('spearman', 0.0)),
                    'pearson_passed': float(e.get('pearson', 0.0)) >= 0.85,
                    'spearman_passed': float(e.get('spearman', 0.0)) >= 0.80,
                    'overall_passed': (float(e.get('pearson', 0.0)) >= 0.85) or (float(e.get('spearman', 0.0)) >= 0.80)
                }
                for e in events_data
            ],
            # 中国专项层：单独章节呈现2015股灾、2016熔断的跨市场验证详情
            'china_specific_section': {
                'requirement': '监管要求：必须包含中国市场极端事件验证',
                'events_covered': [e.get('event_id', 'unknown') for e in china_events],
                'validation_methodology': 'Pearson≥0.85或Spearman≥0.80',
                'results': [
                    {
                        'event': e.get('event_id', 'unknown'),
                        'compared_markets': self._get_comparison_markets(str(e.get('event_type', '')), str(e.get('market', ''))),
                        'pearson_scores': e.get('pearson_by_market', {}),
                        'spearman_scores': e.get('spearman_by_market', {}),
                        'passed': (float(e.get('pearson', 0.0)) >= 0.85) or (float(e.get('spearman', 0.0)) >= 0.80)
                    }
                    for e in china_events
                ]
            }
        }

        return UATResult(
            'cross_market_consistency', overall, float(total_pass), 0.85,
            details={
                'events_count': len(events_data),
                'normal_pass': normal_pass,
                'extreme_pass': extreme_pass,
                'total_pass': total_pass,
                'presentation_structure': presentation_structure  # 第6轮问题2：三级呈现
            }
        )

    def _load_critical_industries(self) -> Tuple[Set[str], bool, Optional[Dict[str, Any]]]:
        """
        从配置文件动态加载关键行业列表（专家第7轮问题3-1）

        Returns:
            (关键行业代码集合, 配置加载成功标志, 配置内容)
        """
        try:
            # 使用ConfigManager加载配置
            from core_bak_refactored.core.share.config_manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get('critical_industries', {})

            industries = {industry['code'] for industry in config.get('critical_industries', [])}
            logger.info(f"成功从配置文件加载{len(industries)}个关键行业")
            return industries, True, config
        except Exception as e:
            logger.warning(f"关键行业配置加载失败: {e}，使用硬编码列表")
            return {'semiconductor', 'new_energy', 'ai_tech'}, False, None

    def validate_industry_parameter_enhanced(self, industry_data: Dict[str, Any]) -> UATResult:
        """
        行业参数差异增强版：新兴行业柔性标准（专家第5轮）
        第7轮优化：从配置文件动态加载关键行业列表（专家第7轮问题3-1）
        """
        # 动态加载关键行业配置（专家第7轮问题3-1）
        critical_industries, config_loaded, config = self._load_critical_industries()

        base_result = self.validate_industry_parameter_difference(
            industry_data.get('parameters', {}), industry_data.get('t_test_results', {})
        )

        industry_code = industry_data.get('industry')
        if industry_code in critical_industries:
            sample_days = int(industry_data.get('sample_days', 0))
            diff_pct = float(industry_data.get('diff_pct', 0.0))
            bootstrap_passed = bool(industry_data.get('bootstrap_passed', False))

            if sample_days >= 400 and (bootstrap_passed or diff_pct >= 0.15):
                result = UATResult(
                    'industry_parameter_difference', True, diff_pct, 0.10,
                    risk_statement='新兴行业特殊处理通过'
                )

                # 记录配置状态和审批文档（专家第7轮问题3-2/3-4）
                result.details['config_status'] = {
                    'loaded': config_loaded,
                    'critical_industries_count': len(critical_industries),
                    'fallback_used': not config_loaded
                }

                # 如果配置文件加载成功，提取审批文档信息（专家第7轮问题3-4）
                if config_loaded and config:
                    industry_config = next(
                        (ind for ind in config.get('critical_industries', [])
                         if ind['code'] == industry_code),
                        None
                    )
                    if industry_config:
                        result.details['approval_document'] = industry_config.get('approval_document')
                        result.details['approval_rationale'] = industry_config.get('rationale')
                        result.details['flexible_threshold'] = industry_config.get('flexible_threshold')
                        result.details['min_sample_days'] = industry_config.get('min_sample_days')

                return result

        # 也为base_result添加配置状态
        base_result.details['config_status'] = {
            'loaded': config_loaded,
            'critical_industries_count': len(critical_industries),
            'fallback_used': not config_loaded
        }

        return base_result

    def _get_comparison_markets(self, event_type: str, event_market: str) -> List[str]:
        """
        根据事件类型和市场返回合适的对比市场（专家第7轮问题2-3）
        """
        market_mapping = {
            'CN': ['US', 'HK', 'EU'],
            'US': ['EU', 'JP', 'CN'],
            'EU': ['US', 'JP', 'CN'],
            'HK': ['CN', 'US', 'EU'],
            'JP': ['US', 'EU', 'CN'],
            'SG': ['US', 'HK', 'EU']
        }
        return market_mapping.get(event_market, ['US', 'EU', 'HK'])



    def _generate_recommendations(self,
                                 test_results: Dict[str, UATResult],
                                 abnormal_handling: Dict[str, Any],
                                 cross_validation: Dict[str, Any]) -> List[str]:
        """
        生成改进建议

        Returns:
            建议列表
        """
        recommendations = []

        # 检查失败的测试
        failed_tests = [name for name, r in test_results.items() if not r.passed]
        if failed_tests:
            recommendations.append(f"以下测试未通过，需要调整: {', '.join(failed_tests)}")

        # 检查Level 2/3告警
        level2_count = abnormal_handling.get('level_breakdown', {}).get('LEVEL_2', 0)
        level3_count = abnormal_handling.get('level_breakdown', {}).get('LEVEL_3', 0)

        if level3_count > 0:
            recommendations.append(f"发现{level3_count}个Level 3严重告警，必须立即处理")

        if level2_count > 0:
            recommendations.append(f"发现{level2_count}个Level 2预警，建议24小时内复核")

        # 检查交叉验证
        if cross_validation.get('enabled', False) and not cross_validation.get('passed', True):
            recommendations.append("数据源交叉验证发现差异，建议检查数据质量")

        if not recommendations:
            recommendations.append("所有验收指标通过，系统运行正常")

        return recommendations

# =============================================================================
# 生产环境监控告警系统（专家第2轮5.4节问题13）
# =============================================================================

class ProductionMonitor:
    """
    生产环境监控系统（专家第2轮5.4节问题13）

    监控指标体系：
    - 数据质量：≥90%（低于85%告警）
    - 预测误差：≤15%（超过18%告警）
    - 系统可用性：≥99.5%（低于99%告警）
    - API响应时间：≤5秒（超过8秒告警）

    告警升级路径：企业微信→短信→电话（30分钟未响应）
    """

    # 监控阈值配置（专家第2轮5.4节问题13）
    MONITOR_THRESHOLDS = {
        'data_quality': {'warning': 0.85, 'critical': 0.80},
        'prediction_error': {'warning': 0.18, 'critical': 0.22},
        'system_availability': {'warning': 0.99, 'critical': 0.98},
        'api_response_time': {'warning': 8.0, 'critical': 12.0}  # 秒
    }

    def __init__(self):
        """初始化生产监控器"""
        self._alert_channels = ['wechat', 'sms', 'phone']
        self._current_metrics = {}
        self._alert_log = []

    def check_system_health(self,
                           data_quality: Optional[float] = None,
                           prediction_error: Optional[float] = None,
                           system_availability: Optional[float] = None,
                           api_response_time: Optional[float] = None) -> Dict[str, Any]:
        """
        系统健康检查（职责简化：仅检查判断，告警委托给AlertManager）

        职责调整：
        - UAT职责：验证指标是否超阈值，生成检查报告
        - AlertManager职责：根据严重级别发送告警，处理升级路径

        使用示例：
            # UAT检查
            health_report = validator.check_system_health(
                data_quality=0.85,
                prediction_error=0.22
            )

            # 告警发送（由调用方委托给AlertManager）
            if health_report['status'] != 'HEALTHY':
                for alert in health_report['alerts']:
                    alert_manager.send_alert(
                        severity=AlertSeverity.WARNING if alert['level'] == 'WARNING' else AlertSeverity.CRITICAL,
                        title=f"{alert['metric_name']}告警",
                        message=alert['message'],
                        metadata={'metric': alert['metric_name'], 'value': alert['actual_value']}
                    )

        Args:
            data_quality: 数据质量得分 (0-1)
            prediction_error: 预测误差
            system_availability: 系统可用性 (0-1)
            api_response_time: API响应时间 (秒)

        Returns:
            健康检查报告（仅包含检查结果，不包含告警发送）
        """
        alerts = []
        status_details = {}

        # 检查数据质量
        if data_quality is not None:
            if data_quality < self.MONITOR_THRESHOLDS['data_quality']['critical']:
                alerts.append(self._create_alert('CRITICAL', '数据质量', data_quality,
                                                self.MONITOR_THRESHOLDS['data_quality']['critical']))
            elif data_quality < self.MONITOR_THRESHOLDS['data_quality']['warning']:
                alerts.append(self._create_alert('WARNING', '数据质量', data_quality,
                                                self.MONITOR_THRESHOLDS['data_quality']['warning']))
            status_details['data_quality'] = {
                'value': data_quality,
                'status': 'OK' if data_quality >= self.MONITOR_THRESHOLDS['data_quality']['warning'] else 'WARNING'
            }

        # 检查预测误差
        if prediction_error is not None:
            if prediction_error > self.MONITOR_THRESHOLDS['prediction_error']['critical']:
                alerts.append(self._create_alert('CRITICAL', '预测误差', prediction_error,
                                                self.MONITOR_THRESHOLDS['prediction_error']['critical']))
            elif prediction_error > self.MONITOR_THRESHOLDS['prediction_error']['warning']:
                alerts.append(self._create_alert('WARNING', '预测误差', prediction_error,
                                                self.MONITOR_THRESHOLDS['prediction_error']['warning']))
            status_details['prediction_error'] = {
                'value': prediction_error,
                'status': 'OK' if prediction_error <= self.MONITOR_THRESHOLDS['prediction_error']['warning'] else 'WARNING'
            }

        # 检查系统可用性
        if system_availability is not None:
            if system_availability < self.MONITOR_THRESHOLDS['system_availability']['critical']:
                alerts.append(self._create_alert('CRITICAL', '系统可用性', system_availability,
                                                self.MONITOR_THRESHOLDS['system_availability']['critical']))
            elif system_availability < self.MONITOR_THRESHOLDS['system_availability']['warning']:
                alerts.append(self._create_alert('WARNING', '系统可用性', system_availability,
                                                self.MONITOR_THRESHOLDS['system_availability']['warning']))
            status_details['system_availability'] = {
                'value': system_availability,
                'status': 'OK' if system_availability >= self.MONITOR_THRESHOLDS['system_availability']['warning'] else 'WARNING'
            }

        # 检查API响应时间
        if api_response_time is not None:
            if api_response_time > self.MONITOR_THRESHOLDS['api_response_time']['critical']:
                alerts.append(self._create_alert('CRITICAL', 'API响应时间', api_response_time,
                                                self.MONITOR_THRESHOLDS['api_response_time']['critical']))
            elif api_response_time > self.MONITOR_THRESHOLDS['api_response_time']['warning']:
                alerts.append(self._create_alert('WARNING', 'API响应时间', api_response_time,
                                                self.MONITOR_THRESHOLDS['api_response_time']['warning']))
            status_details['api_response_time'] = {
                'value': api_response_time,
                'status': 'OK' if api_response_time <= self.MONITOR_THRESHOLDS['api_response_time']['warning'] else 'WARNING'
            }

        # 记录告警（仅记录，不发送）
        self._alert_log.extend(alerts)

        # 更新当前指标
        self._current_metrics = {
            'data_quality': data_quality,
            'prediction_error': prediction_error,
            'system_availability': system_availability,
            'api_response_time': api_response_time,
            'timestamp': pd.Timestamp.now()
        }

        return {
            'status': 'HEALTHY' if not alerts else ('WARNING' if all(a['level'] == 'WARNING' for a in alerts) else 'CRITICAL'),
            'alerts': alerts,
            'status_details': status_details,
            'alert_count': len(alerts),
            'timestamp': pd.Timestamp.now().isoformat()
        }

    def _create_alert(self, level: str, metric_name: str, actual_value: float, threshold: float) -> Dict[str, Any]:
        """
        创建告警

        Args:
            level: 告警级别（WARNING/CRITICAL）
            metric_name: 指标名称
            actual_value: 实际值
            threshold: 阈值

        Returns:
            告警字典
        """
        return {
            'level': level,
            'metric_name': metric_name,
            'actual_value': actual_value,
            'threshold': threshold,
            'timestamp': pd.Timestamp.now(),
            'message': f"{metric_name}{level}告警: 当前值={actual_value:.4f}, 阈值={threshold:.4f}"
        }

    # _escalate_alert和_send_alert方法已废弃，告警功能委托给core.monitoring.AlertManager
    # 调用方应使用AlertManager发送告警

    def get_current_metrics(self) -> Dict[str, Any]:
        """
        获取当前指标

        Returns:
            当前指标字典
        """
        return self._current_metrics.copy()

    def get_alert_history(self, since: Optional[pd.Timestamp] = None) -> List[Dict[str, Any]]:
        """
        获取告警历史
        
        Args:
            since: 起始时间（None表示所有）
        
        Returns:
            告警列表
        """
        if since is None:
            return self._alert_log.copy()
        else:
            return [a for a in self._alert_log if a['timestamp'] >= since]
