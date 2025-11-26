"""
数据质量检查器（职责归位重构）

职责：
- 数据质量多维度验证（完整性/一致性/连续性/合理性）
- 数据源交叉验证（双维度比对）
- 数据质量报告生成

设计原则：
- 单一职责：仅负责数据质量检查，不涉及数据获取
- 可复用：供历史数据提供者、实时数据流等多处使用
- 可扩展：支持自定义检查规则
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger('DeepSeekQuant.DataQuality')


@dataclass
class DataQualityReport:
    """数据质量报告"""
    overall_score: float  # 总分 0-1
    completeness: float  # 完整性得分
    consistency: float  # 一致性得分
    continuity: float  # 连续性得分
    reasonableness: float  # 合理性得分
    issues: List[str] = field(default_factory=list)  # 问题列表
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    
    @property
    def passed(self) -> bool:
        """是否通过（总分≥0.9）"""
        return self.overall_score >= 0.9


@dataclass
class CrossValidationResult:
    """交叉验证结果"""
    source_a: str
    source_b: str
    passed: bool
    overlap_days: int
    daily_divergence: Dict[str, Any]
    mean_divergence: Dict[str, Any]
    std_divergence: Dict[str, Any]
    details: Dict[str, Any] = field(default_factory=dict)


class DataQualityChecker:
    """
    数据质量检查器
    
    功能：
    1. 单源数据质量检查（完整性/一致性/连续性/合理性）
    2. 多源交叉验证（逐日差异/窗口统计量）
    3. 质量报告生成
    
    使用示例：
        checker = DataQualityChecker()
        
        # 单源质量检查
        report = checker.check_quality(data, index_id='000300.SH')
        if not report.passed:
            logger.warning(f"数据质量问题: {report.issues}")
        
        # 交叉验证
        result = checker.cross_validate(data_a, data_b, 'yahoo', 'mock')
        if not result.passed:
            logger.warning(f"数据源差异: {result.daily_divergence}")
    """
    
    # 合理性检查阈值
    REASONABLENESS_THRESHOLDS = {
        'max_daily_return': 0.12,  # 单日最大涨幅12%
        'min_price': 1.0,  # 最小价格
        'max_price_ratio': 100.0,  # 价格极值比
        'min_volume': 0.0,  # 最小成交量
    }
    
    # 交叉验证阈值
    CROSS_VALIDATION_THRESHOLDS = {
        'daily_divergence': 0.30,  # 逐日差异30%触发
        'mean_divergence': 0.03,  # 均值差异3%
        'std_divergence': 0.10,  # 标准差差异10%
        'max_daily_divergence_ratio': 0.10,  # 允许10%日期超阈值
    }
    
    def __init__(self):
        """初始化数据质量检查器"""
        self._check_history: List[DataQualityReport] = []
        self._validation_history: List[CrossValidationResult] = []
    
    def check_quality(self, 
                     data: pd.DataFrame,
                     index_id: str = 'unknown',
                     expected_days: Optional[int] = None) -> DataQualityReport:
        """
        数据质量多维度检查
        
        Args:
            data: 数据DataFrame（必须包含date, close, volume列）
            index_id: 指数代码（用于日志）
            expected_days: 期望天数（用于完整性检查）
        
        Returns:
            数据质量报告
        """
        issues = []
        
        # 1. 完整性检查
        completeness_score = self._check_completeness(data, expected_days, issues)
        
        # 2. 一致性检查（字段类型、取值范围）
        consistency_score = self._check_consistency(data, issues)
        
        # 3. 连续性检查（时间连续性、缺失值）
        continuity_score = self._check_continuity(data, issues)
        
        # 4. 合理性检查（价格波动、成交量异常）
        reasonableness_score = self._check_reasonableness(data, index_id, issues)
        
        # 计算总分（加权平均）
        overall_score = (
            completeness_score * 0.25 +
            consistency_score * 0.25 +
            continuity_score * 0.25 +
            reasonableness_score * 0.25
        )
        
        report = DataQualityReport(
            overall_score=overall_score,
            completeness=completeness_score,
            consistency=consistency_score,
            continuity=continuity_score,
            reasonableness=reasonableness_score,
            issues=issues,
            metadata={
                'index_id': index_id,
                'rows': len(data),
                'expected_days': expected_days,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        self._check_history.append(report)
        
        logger.info(f"数据质量检查完成: {index_id}, 总分={overall_score:.2f}, "
                   f"问题数={len(issues)}")
        
        return report
    
    def _check_completeness(self, data: pd.DataFrame, expected_days: Optional[int], 
                           issues: List[str]) -> float:
        """完整性检查"""
        if expected_days is None:
            return 1.0
        
        actual_days = len(data)
        completeness = min(1.0, actual_days / expected_days)
        
        if completeness < 0.9:
            issues.append(f"数据不完整: 期望{expected_days}天, 实际{actual_days}天")
        
        return completeness
    
    def _check_consistency(self, data: pd.DataFrame, issues: List[str]) -> float:
        """一致性检查"""
        score = 1.0
        
        # 检查必需字段
        required_fields = ['date', 'close', 'volume']
        missing_fields = [f for f in required_fields if f not in data.columns]
        if missing_fields:
            issues.append(f"缺失必需字段: {missing_fields}")
            score *= 0.5
        
        # 检查字段类型
        if 'close' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['close']):
                issues.append("close字段非数值类型")
                score *= 0.8
        
        if 'volume' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['volume']):
                issues.append("volume字段非数值类型")
                score *= 0.8
        
        return max(0.0, score)
    
    def _check_continuity(self, data: pd.DataFrame, issues: List[str]) -> float:
        """连续性检查"""
        score = 1.0
        
        # 检查缺失值
        if 'close' in data.columns:
            missing_count = data['close'].isna().sum()
            if missing_count > 0:
                missing_ratio = missing_count / len(data)
                issues.append(f"close字段有{missing_count}个缺失值({missing_ratio:.1%})")
                score *= (1.0 - missing_ratio)
        
        # 检查时间连续性（交易日）
        if 'date' in data.columns and len(data) > 1:
            dates = pd.to_datetime(data['date'])
            date_diff = dates.diff().dt.days.dropna()
            # 交易日间隔通常1-3天（考虑周末节假日）
            abnormal_gaps = (date_diff > 10).sum()
            if abnormal_gaps > 0:
                issues.append(f"发现{abnormal_gaps}个异常时间间隔(>10天)")
                score *= max(0.7, 1.0 - abnormal_gaps / len(data))
        
        return max(0.0, score)
    
    def _check_reasonableness(self, data: pd.DataFrame, index_id: str, 
                            issues: List[str]) -> float:
        """合理性检查"""
        score = 1.0
        
        if 'close' not in data.columns or len(data) < 2:
            return score
        
        prices = data['close'].values
        
        # 检查价格范围
        if (prices < self.REASONABLENESS_THRESHOLDS['min_price']).any():
            issues.append(f"存在异常低价(<{self.REASONABLENESS_THRESHOLDS['min_price']})")
            score *= 0.8
        
        price_ratio = prices.max() / max(prices.min(), 0.01)
        if price_ratio > self.REASONABLENESS_THRESHOLDS['max_price_ratio']:
            issues.append(f"价格极值比过大({price_ratio:.1f})")
            score *= 0.9
        
        # 检查收益率
        returns = np.diff(prices) / prices[:-1]
        max_abs_return = np.abs(returns).max()
        if max_abs_return > self.REASONABLENESS_THRESHOLDS['max_daily_return']:
            issues.append(f"存在异常波动({max_abs_return:.1%} > {self.REASONABLENESS_THRESHOLDS['max_daily_return']:.1%})")
            score *= 0.9
        
        # 检查成交量
        if 'volume' in data.columns:
            volumes = data['volume'].values
            if (volumes < self.REASONABLENESS_THRESHOLDS['min_volume']).any():
                zero_volume_count = (volumes <= 0).sum()
                issues.append(f"存在{zero_volume_count}个零成交量")
                score *= max(0.8, 1.0 - zero_volume_count / len(data))
        
        return max(0.0, score)
    
    def cross_validate(self,
                      data_a: pd.DataFrame,
                      data_b: pd.DataFrame,
                      source_a: str,
                      source_b: str) -> CrossValidationResult:
        """
        数据源交叉验证（专家answer.md第3轮5.1节双维度验证）
        
        验证维度：
        1. 逐日差异：30%触发，允许10%日期超阈值
        2. 窗口统计量：均值3%/标准差10%
        
        Args:
            data_a: 数据源A
            data_b: 数据源B
            source_a: 数据源A名称
            source_b: 数据源B名称
        
        Returns:
            交叉验证结果
        """
        # 按日期合并
        merged = pd.merge(
            data_a[['date', 'close']],
            data_b[['date', 'close']],
            on='date', how='inner', suffixes=('_a', '_b')
        )
        
        if len(merged) == 0:
            logger.warning(f"{source_a}与{source_b}无重叠数据")
            return CrossValidationResult(
                source_a=source_a,
                source_b=source_b,
                passed=False,
                overlap_days=0,
                daily_divergence={'count': 0, 'ratio': 0.0, 'threshold': 0.30, 'passed': False},
                mean_divergence={'diff_pct': 0.0, 'threshold': 0.03, 'passed': False},
                std_divergence={'diff_pct': 0.0, 'threshold': 0.10, 'passed': False},
                details={'error': 'no_overlap'}
            )
        
        # 1. 逐日差异检查
        daily_diff = abs(merged['close_a'] - merged['close_b']) / merged['close_a']
        daily_divergence_count = (daily_diff > self.CROSS_VALIDATION_THRESHOLDS['daily_divergence']).sum()
        daily_divergence_ratio = daily_divergence_count / len(merged)
        daily_passed = daily_divergence_ratio <= self.CROSS_VALIDATION_THRESHOLDS['max_daily_divergence_ratio']
        
        # 2. 窗口统计量检查
        mean_diff_pct = abs(merged['close_a'].mean() - merged['close_b'].mean()) / merged['close_a'].mean()
        std_diff_pct = abs(merged['close_a'].std() - merged['close_b'].std()) / merged['close_a'].std()
        
        mean_passed = mean_diff_pct <= self.CROSS_VALIDATION_THRESHOLDS['mean_divergence']
        std_passed = std_diff_pct <= self.CROSS_VALIDATION_THRESHOLDS['std_divergence']
        
        # 总体通过判定：逐日差异通过 AND (均值通过 OR 标准差通过)
        passed = daily_passed and (mean_passed or std_passed)
        
        result = CrossValidationResult(
            source_a=source_a,
            source_b=source_b,
            passed=passed,
            overlap_days=len(merged),
            daily_divergence={
                'count': int(daily_divergence_count),
                'ratio': float(daily_divergence_ratio),
                'threshold': self.CROSS_VALIDATION_THRESHOLDS['daily_divergence'],
                'passed': daily_passed
            },
            mean_divergence={
                'diff_pct': float(mean_diff_pct),
                'threshold': self.CROSS_VALIDATION_THRESHOLDS['mean_divergence'],
                'passed': mean_passed
            },
            std_divergence={
                'diff_pct': float(std_diff_pct),
                'threshold': self.CROSS_VALIDATION_THRESHOLDS['std_divergence'],
                'passed': std_passed
            },
            details={
                'max_daily_diff': float(daily_diff.max()),
                'avg_daily_diff': float(daily_diff.mean())
            }
        )
        
        self._validation_history.append(result)
        
        logger.info(f"交叉验证: {source_a} vs {source_b}, "
                   f"重叠{len(merged)}天, {'通过' if passed else '未通过'}")
        
        return result
    
    def get_check_history(self, limit: int = 10) -> List[DataQualityReport]:
        """获取质量检查历史"""
        return self._check_history[-limit:]
    
    def get_validation_history(self, limit: int = 10) -> List[CrossValidationResult]:
        """获取交叉验证历史"""
        return self._validation_history[-limit:]
