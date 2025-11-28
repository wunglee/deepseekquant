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
from core_bak_refactored.core.share.market_enums import MarketCode

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
    # 分市场合理性阈值（专家答复）
    MARKET_SPECIFIC_THRESHOLDS = {
        MarketCode.CN.value: {'max_daily_return': 0.10},
        MarketCode.US.value: {'max_daily_return': 0.15},
        MarketCode.HK.value: {'max_daily_return': 0.08},
        MarketCode.JP.value: {'max_daily_return': 0.06},
        MarketCode.EU.value: {'max_daily_return': 0.07},
        MarketCode.SG.value: {'max_daily_return': 0.10},
    }
    # 连续性检查异常间隔阈值（天）
    ABNORMAL_GAP_THRESHOLD = {
        'default': 10,
        MarketCode.CN.value: 7,
        MarketCode.US.value: 3,
        MarketCode.HK.value: 5,
        MarketCode.JP.value: 5,
        MarketCode.EU.value: 5,
        MarketCode.SG.value: 5,
    }
    
    # 交叉验证阈值（默认）
    CROSS_VALIDATION_THRESHOLDS = {
        'daily_divergence': 0.30,  # 逐日差异30%触发
        'mean_divergence': 0.03,  # 均值差异3%
        'std_divergence': 0.10,  # 标准差差异10%
        'max_daily_divergence_ratio': 0.10,  # 允许10%日期超阈值
    }
    # 分市场交叉验证阈值（专家答复）
    PER_MARKET_CROSS_VALIDATION_THRESHOLDS = {
        MarketCode.CN.value: {
            'daily_divergence': 0.30,
            'mean_divergence': 0.03,
            'std_divergence': 0.10,
            'max_daily_divergence_ratio': 0.10,
        },
        MarketCode.US.value: {
            'daily_divergence': 0.25,
            'mean_divergence': 0.02,
            'std_divergence': 0.08,
            'max_daily_divergence_ratio': 0.05,
        },
        MarketCode.HK.value: {
            'daily_divergence': 0.30,
            'mean_divergence': 0.03,
            'std_divergence': 0.12,
            'max_daily_divergence_ratio': 0.10,
        },
        MarketCode.JP.value: {
            'daily_divergence': 0.28,
            'mean_divergence': 0.025,
            'std_divergence': 0.09,
            'max_daily_divergence_ratio': 0.08,
        },
        MarketCode.EU.value: {
            'daily_divergence': 0.28,
            'mean_divergence': 0.025,
            'std_divergence': 0.09,
            'max_daily_divergence_ratio': 0.08,
        },
        MarketCode.SG.value: {
            'daily_divergence': 0.35,
            'mean_divergence': 0.04,
            'std_divergence': 0.15,
            'max_daily_divergence_ratio': 0.15,
        },
    }
    
    def __init__(self):
        """初始化数据质量检查器"""
        self._check_history: List[DataQualityReport] = []
        self._validation_history: List[CrossValidationResult] = []
        # 数据源可信度评分（初始化100分，专家第4轮建议）
        self._source_ratings: Dict[str, int] = {}
        # 紧急回退缓存（专家第7轮问题6-2）
        self._emergency_cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def check_quality(self, 
                     data: pd.DataFrame,
                     index_id: str = 'unknown',
                     expected_days: Optional[int] = None,
                     market: Optional[str] = None) -> DataQualityReport:
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
        continuity_score = self._check_continuity(data, issues, market)
        
        # 4. 合理性检查（价格波动、成交量异常）
        reasonableness_score = self._check_reasonableness(data, index_id, issues, market)
        
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
    
    def _check_continuity(self, data: pd.DataFrame, issues: List[str], market: Optional[str] = None) -> float:
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
            # 交易日间隔阈值按市场差异化
            mk = market if market in self.ABNORMAL_GAP_THRESHOLD else 'default'
            gap_threshold = self.ABNORMAL_GAP_THRESHOLD.get(mk, self.ABNORMAL_GAP_THRESHOLD['default'])
            abnormal_gaps = (date_diff > gap_threshold).sum()
            if abnormal_gaps > 0:
                issues.append(f"发现{abnormal_gaps}个异常时间间隔(>{gap_threshold}天)")
                score *= max(0.7, 1.0 - abnormal_gaps / len(data))
        
        return max(0.0, score)
    
    def _check_reasonableness(self, data: pd.DataFrame, index_id: str, 
                            issues: List[str], market: Optional[str] = None) -> float:
        """合理性检查"""
        score = 1.0
        thresholds = dict(self.REASONABLENESS_THRESHOLDS)
        if market and market in self.MARKET_SPECIFIC_THRESHOLDS:
            thresholds.update(self.MARKET_SPECIFIC_THRESHOLDS[market])
        
        if 'close' not in data.columns or len(data) < 2:
            return score
        
        prices = data['close'].values
        
        # 检查价格范围
        if (prices < thresholds['min_price']).any():
            issues.append(f"存在异常低价(<{thresholds['min_price']})")
            score *= 0.8
        
        price_ratio = prices.max() / max(prices.min(), 0.01)
        if price_ratio > thresholds['max_price_ratio']:
            issues.append(f"价格极值比过大({price_ratio:.1f})")
            score *= 0.9
        
        # 检查收益率
        returns = np.diff(prices) / prices[:-1]
        max_abs_return = np.abs(returns).max()
        if max_abs_return > thresholds['max_daily_return']:
            issues.append(f"存在异常波动({max_abs_return:.1%} > {thresholds['max_daily_return']:.1%})")
            score *= 0.9
        
        # 检查成交量
        if 'volume' in data.columns:
            volumes = data['volume'].values
            if (volumes < thresholds['min_volume']).any():
                zero_volume_count = (volumes <= 0).sum()
                issues.append(f"存在{zero_volume_count}个零成交量")
                score *= max(0.8, 1.0 - zero_volume_count / len(data))
        
        return max(0.0, score)
    
    def cross_validate(self,
                      data_a: pd.DataFrame,
                      data_b: pd.DataFrame,
                      source_a: str,
                      source_b: str,
                      market: Optional[str] = None,
                      data_freshness_hours: Optional[Dict[str, float]] = None) -> CrossValidationResult:
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
        daily_divergence_count = (daily_diff > (self.PER_MARKET_CROSS_VALIDATION_THRESHOLDS.get(market, self.CROSS_VALIDATION_THRESHOLDS)['daily_divergence'])).sum()
        daily_divergence_ratio = daily_divergence_count / len(merged)
        daily_passed = daily_divergence_ratio <= (self.PER_MARKET_CROSS_VALIDATION_THRESHOLDS.get(market, self.CROSS_VALIDATION_THRESHOLDS)['max_daily_divergence_ratio'])
        
        # 2. 窗口统计量检查
        mean_a = merged['close_a'].mean()
        mean_b = merged['close_b'].mean()
        if abs(mean_a) <= 1e-12:
            mean_diff_pct = 0.0 if abs(mean_b) <= 1e-12 else float('inf')
        else:
            mean_diff_pct = abs(mean_a - mean_b) / abs(mean_a)
        
        std_a = merged['close_a'].std()
        std_b = merged['close_b'].std()
        if abs(std_a) <= 1e-12:
            std_diff_pct = 0.0 if abs(std_b) <= 1e-12 else float('inf')
        else:
            std_diff_pct = abs(std_a - std_b) / abs(std_a)
        
        mean_passed = mean_diff_pct <= (self.PER_MARKET_CROSS_VALIDATION_THRESHOLDS.get(market, self.CROSS_VALIDATION_THRESHOLDS)['mean_divergence'])
        std_passed = std_diff_pct <= (self.PER_MARKET_CROSS_VALIDATION_THRESHOLDS.get(market, self.CROSS_VALIDATION_THRESHOLDS)['std_divergence'])
        
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
                'threshold': self.PER_MARKET_CROSS_VALIDATION_THRESHOLDS.get(market, self.CROSS_VALIDATION_THRESHOLDS)['daily_divergence'],
                'passed': daily_passed
            },
            mean_divergence={
                'diff_pct': float(mean_diff_pct),
                'threshold': self.PER_MARKET_CROSS_VALIDATION_THRESHOLDS.get(market, self.CROSS_VALIDATION_THRESHOLDS)['mean_divergence'],
                'passed': mean_passed
            },
            std_divergence={
                'diff_pct': float(std_diff_pct),
                'threshold': self.PER_MARKET_CROSS_VALIDATION_THRESHOLDS.get(market, self.CROSS_VALIDATION_THRESHOLDS)['std_divergence'],
                'passed': std_passed
            },
            details={
                'max_daily_diff': float(daily_diff.max()),
                'avg_daily_diff': float(daily_diff.mean())
            }
        )
        
        self._validation_history.append(result)
        
        # 更新数据源可信度评分（专家第4/5轮）
        if not hasattr(self, '_source_ratings'):
            self._source_ratings = {}
        self._source_ratings.setdefault(source_a, 100)
        self._source_ratings.setdefault(source_b, 100)
        
        def _apply_penalty(src: str, check_type: str, passed_flag: bool):
            if passed_flag:
                return
            penalty_scores = {
                'daily_divergence': 10,
                'mean_divergence': 5,
                'std_divergence': 8,
                'data_freshness': 5,
            }
            penalty = penalty_scores.get(check_type, 5)
            self._source_ratings[src] = max(0, self._source_ratings[src] - penalty)
        
        if not daily_passed:
            _apply_penalty(source_a, 'daily_divergence', False)
            _apply_penalty(source_b, 'daily_divergence', False)
        if not mean_passed:
            _apply_penalty(source_a, 'mean_divergence', False)
            _apply_penalty(source_b, 'mean_divergence', False)
        if not std_passed:
            _apply_penalty(source_a, 'std_divergence', False)
            _apply_penalty(source_b, 'std_divergence', False)
        
        # 数据更新及时性评分（延迟>2小时扣5分）
        if data_freshness_hours:
            for src, delay in data_freshness_hours.items():
                self._source_ratings.setdefault(src, 100)
                if delay is not None and delay > 2.0:
                    _apply_penalty(src, 'data_freshness', False)
        
        # 评分<=60触发第三数据源复核（仅记录在details，具体复核流程由上层处理）
        low_score_sources = {
            src: score for src, score in self._source_ratings.items() if score <= 60
        }
        if low_score_sources:
            result.details['third_source_review_required'] = True
            result.details['low_score_sources'] = low_score_sources
        
        logger.info(f"交叉验证: {source_a} vs {source_b}, "
                   f"重叠{len(merged)}天, {'通过' if passed else '未通过'} | "
                   f"ratings: {source_a}={self._source_ratings[source_a]}, {source_b}={self._source_ratings[source_b]}")
        
        return result
    
    def get_check_history(self, limit: int = 10) -> List[DataQualityReport]:
        """获取质量检查历史"""
        return self._check_history[-limit:]
    
    def get_validation_history(self, limit: int = 10) -> List[CrossValidationResult]:
        """获取交叉验证历史"""
        return self._validation_history[-limit:]
    
    def get_source_health_summary(self, 
                                   primary_source: Optional[str] = None,
                                   backup_source: Optional[str] = None,
                                   monitoring_sources: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        数据源健康度摘要（专家第5轮建议，第6轮问题4增强）
        第6轮问题4：主备监控三级全披露
        """
        if not hasattr(self, '_source_ratings'):
            return {}
        
        def _get_level(score: int) -> str:
            if score >= 90:
                return '优秀'
            elif score >= 70:
                return '良好'
            elif score >= 60:
                return '警告'
            else:
                return '危险'
        
        summary: Dict[str, Any] = {}
        
        # 1. 主数据源
        if primary_source and primary_source in self._source_ratings:
            score = self._source_ratings[primary_source]
            summary['primary_source'] = {'name': primary_source, 'score': score, 'level': _get_level(score), 'penalties': []}
        
        # 2. 备用数据源
        if backup_source and backup_source in self._source_ratings:
            score = self._source_ratings[backup_source]
            summary['backup_source'] = {'name': backup_source, 'score': score, 'level': _get_level(score), 'penalties': []}
        
        # 3. 监控数据源
        if monitoring_sources:
            summary['monitoring_sources'] = [
                {'name': src, 'score': self._source_ratings[src], 'level': _get_level(self._source_ratings[src])}
                for src in monitoring_sources if src in self._source_ratings
            ]
        
        # 4. 危险档位数据源（专家第6轮问题4：评分≤60）
        dangerous = [
            {'name': source, 'score': score, 'level': _get_level(score), 'action': '暂停参与新回测 + 显式披露'}
            for source, score in self._source_ratings.items() if score <= 60
        ]
        if dangerous:
            summary['dangerous_sources'] = dangerous
        
        # 5. 所有数据源总览（兼容旧逻辑）
        summary['all_sources'] = {source: {'score': score, 'level': _get_level(score)} for source, score in self._source_ratings.items()}
        
        return summary
    
    def should_pause_source(self, source_name: str) -> bool:
        """
        检查数据源是否应该暂停使用（专家第7轮问题4-1）
        
        Args:
            source_name: 数据源名称
        
        Returns:
            是否应该暂停
        """
        return self._source_ratings.get(source_name, 100) <= 60
    
    def get_source_switch_recommendation(self, primary_source: str, backup_source: str) -> Dict[str, Any]:
        """
        获取数据源切换建议（专家第7轮问题4-2）
        
        Args:
            primary_source: 主数据源名称
            backup_source: 备用数据源名称
        
        Returns:
            切换建议信息
        """
        primary_rating = self._source_ratings.get(primary_source, 100)
        backup_rating = self._source_ratings.get(backup_source, 100)
        
        return {
            'switch_recommended': primary_rating <= 60 and backup_rating > 70,
            'primary_rating': primary_rating,
            'backup_rating': backup_rating,
            'auto_switch_allowed': primary_rating <= 50,  # 严重危险可自动切换
            'manual_approval_required': 50 < primary_rating <= 60  # 一般危险需人工确认
        }
    
    def cache_emergency_data(self, source_name: str, data: pd.DataFrame) -> None:
        """
        缓存紧急回退数据（专家第7轮问题6-2）
        
        Args:
            source_name: 数据源名称
            data: 数据DataFrame
        """
        self._emergency_cache[source_name] = data.copy()
        self._cache_timestamps[source_name] = datetime.now()
        logger.info(f"已缓存数据源{source_name}的紧急回退数据，{len(data)}行")
    
    def get_emergency_fallback_data(self, source_name: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        """
        获取紧急回退数据（专家第7轮问题6-2）
        
        Args:
            source_name: 数据源名称
            max_age_hours: 最大缓存年龄（小时）
        
        Returns:
            缓存数据或None
        """
        if source_name not in self._emergency_cache:
            logger.warning(f"数据源{source_name}无紧急回退缓存")
            return None
        
        cache_age = (datetime.now() - self._cache_timestamps[source_name]).total_seconds() / 3600
        if cache_age > max_age_hours:
            logger.warning(f"数据源{source_name}缓存已过期（{cache_age:.1f}小时 > {max_age_hours}小时）")
            return None
        
        logger.info(f"使用数据源{source_name}的紧急回退缓存（年龄：{cache_age:.1f}小时）")
        return self._emergency_cache[source_name].copy()
