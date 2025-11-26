"""
数据质量检查器 - Phase 5B-5 重构提取
从 historical_data_provider.py 提取，消除代码重复

职责：
- 统一的数据质量验证逻辑
- 完整性、一致性、连续性、合理性检查
- 可被数据层和验收层复用

设计原则：
- 单一职责：只负责质量检查，不负责数据获取
- 可复用：静态方法，无状态
- 可测试：独立的单元测试
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger('DeepSeekQuant.DataQuality')


@dataclass
class DataQualityReport:
    """数据质量报告"""
    overall_score: float  # 总体评分 (0-1)
    completeness_score: float  # 完整性评分
    consistency_score: float  # 一致性评分
    continuity_score: float  # 连续性评分
    reasonableness_score: float  # 合理性评分
    passed: bool  # 是否通过（≥0.6）
    details: Dict[str, Any]  # 详细信息


class DataQualityChecker:
    """
    数据质量检查器（重构提取）
    
    基于专家answer.md第1轮5.1节：数据质量评分≥90%
    
    评分体系：
    - 完整性（30%）：缺失值比例
    - 一致性（30%）：负价格、异常大价格
    - 连续性（20%）：时间序列断档
    - 合理性（20%）：收益率合理性
    """
    
    # 质量阈值配置
    COMPLETENESS_WEIGHT = 0.3
    CONSISTENCY_WEIGHT = 0.3
    CONTINUITY_WEIGHT = 0.2
    REASONABLENESS_WEIGHT = 0.2
    
    PASS_THRESHOLD = 0.6  # 及格线
    EXCELLENT_THRESHOLD = 0.9  # 优秀线
    
    @staticmethod
    def check_quality(data: pd.DataFrame, 
                     source: str = 'unknown',
                     cache_key: str = None) -> DataQualityReport:
        """
        检查数据质量
        
        Args:
            data: 待检查的数据框
            source: 数据源名称
            cache_key: 缓存键（可选）
        
        Returns:
            DataQualityReport
        """
        if data is None or data.empty:
            return DataQualityReport(
                overall_score=0.0,
                completeness_score=0.0,
                consistency_score=0.0,
                continuity_score=0.0,
                reasonableness_score=0.0,
                passed=False,
                details={'error': 'empty_data', 'source': source}
            )
        
        try:
            # 1. 完整性检查
            completeness = DataQualityChecker._check_completeness(data)
            
            # 2. 一致性检查
            consistency = DataQualityChecker._check_consistency(data)
            
            # 3. 连续性检查
            continuity = DataQualityChecker._check_continuity(data)
            
            # 4. 合理性检查
            reasonableness = DataQualityChecker._check_reasonableness(data)
            
            # 计算总分
            overall_score = (
                completeness * DataQualityChecker.COMPLETENESS_WEIGHT +
                consistency * DataQualityChecker.CONSISTENCY_WEIGHT +
                continuity * DataQualityChecker.CONTINUITY_WEIGHT +
                reasonableness * DataQualityChecker.REASONABLENESS_WEIGHT
            )
            
            # 确保评分在[0, 1]范围内
            overall_score = max(0.0, min(1.0, overall_score))
            
            passed = overall_score >= DataQualityChecker.PASS_THRESHOLD
            
            report = DataQualityReport(
                overall_score=overall_score,
                completeness_score=completeness,
                consistency_score=consistency,
                continuity_score=continuity,
                reasonableness_score=reasonableness,
                passed=passed,
                details={
                    'source': source,
                    'total_rows': len(data),
                    'columns': list(data.columns),
                    'cache_key': cache_key
                }
            )
            
            level = "优秀" if overall_score >= DataQualityChecker.EXCELLENT_THRESHOLD else ("及格" if passed else "不及格")
            logger.info(f"数据质量检查完成: {source}, 总分={overall_score:.4f} ({level})")
            
            return report
            
        except Exception as e:
            logger.error(f"数据质量检查失败: {e}")
            return DataQualityReport(
                overall_score=0.0,
                completeness_score=0.0,
                consistency_score=0.0,
                continuity_score=0.0,
                reasonableness_score=0.0,
                passed=False,
                details={'error': str(e), 'source': source}
            )
    
    @staticmethod
    def _check_completeness(data: pd.DataFrame) -> float:
        """
        完整性检查：缺失值比例
        
        Returns:
            评分 (0-1)，1表示无缺失
        """
        total_cells = len(data) * len(data.columns)
        if total_cells == 0:
            return 0.0
        
        missing_count = data.isnull().sum().sum()
        completeness = 1.0 - (missing_count / total_cells)
        
        logger.debug(f"完整性: {completeness:.4f} (缺失{missing_count}/{total_cells})")
        return completeness
    
    @staticmethod
    def _check_consistency(data: pd.DataFrame) -> float:
        """
        一致性检查：负价格、异常大价格
        
        Returns:
            评分 (0-1)，1表示完全一致
        """
        consistency = 1.0
        
        if 'close' not in data.columns or len(data) == 0:
            return consistency
        
        close_prices = data['close'].dropna()
        if len(close_prices) == 0:
            return 0.5  # 无价格数据，给部分分
        
        # 检查负价格
        negative_prices = (close_prices < 0).sum()
        if negative_prices > 0:
            penalty = min(0.5, 0.5 * (negative_prices / len(close_prices)))
            consistency -= penalty
            logger.debug(f"检测到{negative_prices}个负价格，扣分{penalty:.4f}")
        
        # 检查异常大价格（超过均值10倍）
        mean_price = close_prices.mean()
        if mean_price > 0:
            extreme_prices = (close_prices > mean_price * 10).sum()
            if extreme_prices > 0:
                penalty = min(0.3, 0.3 * (extreme_prices / len(close_prices)))
                consistency -= penalty
                logger.debug(f"检测到{extreme_prices}个异常大价格，扣分{penalty:.4f}")
        
        return max(0.0, consistency)
    
    @staticmethod
    def _check_continuity(data: pd.DataFrame) -> float:
        """
        连续性检查：时间序列断档
        
        Returns:
            评分 (0-1)，1表示完全连续
        """
        if 'date' not in data.columns or len(data) <= 1:
            return 0.5  # 无法验证，给部分分
        
        try:
            date_series = pd.to_datetime(data['date'])
            date_diffs = date_series.diff().dt.days
            
            # 过滤NaT和负值
            valid_diffs = date_diffs[date_diffs.notna() & (date_diffs > 0)]
            if len(valid_diffs) == 0:
                return 0.5
            
            # 检查长间隔（>10天视为断档）
            long_gaps = (valid_diffs > 10).sum()
            continuity = 1.0 - (long_gaps / len(valid_diffs))
            
            logger.debug(f"连续性: {continuity:.4f} (断档{long_gaps}/{len(valid_diffs)})")
            return continuity
            
        except Exception as e:
            logger.warning(f"连续性检查失败: {e}")
            return 0.5
    
    @staticmethod
    def _check_reasonableness(data: pd.DataFrame) -> float:
        """
        合理性检查：收益率合理性
        
        Returns:
            评分 (0-1)，1表示完全合理
        """
        reasonableness = 1.0
        
        if 'close' not in data.columns or len(data) <= 1:
            return reasonableness
        
        try:
            close_prices = data['close'].dropna()
            if len(close_prices) <= 1:
                return reasonableness
            
            returns = close_prices.pct_change().dropna()
            if len(returns) == 0:
                return reasonableness
            
            # 日收益率应在-50%到+50%之间（极端但合理）
            unreasonable = ((returns < -0.5) | (returns > 0.5)).sum()
            if unreasonable > 0:
                penalty = min(0.2, 0.2 * (unreasonable / len(returns)))
                reasonableness -= penalty
                logger.debug(f"检测到{unreasonable}个极端收益率，扣分{penalty:.4f}")
            
            return max(0.0, reasonableness)
            
        except Exception as e:
            logger.warning(f"合理性检查失败: {e}")
            return 0.8  # 降级评分
