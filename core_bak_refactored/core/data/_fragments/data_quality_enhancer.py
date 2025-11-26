"""
数据质量增强器
从第6轮专家指导实施
职责: 多源数据验证、质量评分、备选数据源切换

设计原则：
- 多源数据交叉验证
- 质量评分机制
- 自动切换备选数据源
"""

import pandas as pd
from typing import List, Dict, Any, Union
from datetime import datetime
import logging
from dataclasses import dataclass, field

logger = logging.getLogger('DeepSeekQuant.DataQualityEnhancer')


@dataclass
class DataQualityReport:
    """数据质量报告"""
    completeness_score: float = 0.0  # 完整性评分 (0-1)
    consistency_score: float = 0.0   # 一致性评分 (0-1)
    accuracy_score: float = 0.0      # 准确性评分 (0-1)
    outliers_detected: int = 0       # 检测到的异常值数量
    total_rows: int = 0              # 总行数
    missing_values: int = 0          # 缺失值数量
    overall_score: float = 0.0       # 综合评分 (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataQualityEnhancer:
    """数据质量增强器"""
    
    def __init__(self, primary_source, backup_sources: List = None):
        """
        初始化数据质量增强器
        
        Args:
            primary_source: 主数据源
            backup_sources: 备选数据源列表
        """
        self.primary = primary_source
        self.backups = backup_sources or []
    
    def get_enhanced_prices(self, index_id: str, start_date: Union[str, datetime], 
                           end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        获取增强数据（多源验证）
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame: 增强后的数据
        """
        # 1. 从主源获取数据
        try:
            primary_data = self.primary.get_index_prices(index_id, start_date, end_date)
            logger.info(f"从主数据源获取到 {len(primary_data)} 条数据")
        except Exception as e:
            logger.warning(f"主数据源获取失败: {e}")
            primary_data = None
        
        # 2. 数据质量检查
        if primary_data is not None and not primary_data.empty:
            quality_report = self.validate_data_quality(primary_data)
            logger.info(f"主数据源质量评分: {quality_report.overall_score:.2f}")
            
            # 3. 质量不足时使用备源
            if quality_report.overall_score < 0.8:  # 质量阈值80%
                logger.warning(f"主数据源质量不足 ({quality_report.overall_score:.2f})，尝试备选数据源")
                for i, backup in enumerate(self.backups):
                    try:
                        backup_data = backup.get_index_prices(index_id, start_date, end_date)
                        backup_quality = self.validate_data_quality(backup_data)
                        logger.info(f"备选数据源 {i+1} 质量评分: {backup_quality.overall_score:.2f}")
                        
                        if backup_quality.overall_score > quality_report.overall_score:
                            logger.info(f"使用备选数据源 {i+1} 的数据（质量更好）")
                            return backup_data
                    except Exception as e:
                        logger.warning(f"备选数据源 {i+1} 获取失败: {e}")
                        continue
        
        # 4. 返回主源数据（即使质量不高也返回）
        if primary_data is not None:
            return primary_data
        
        # 5. 如果主源失败且有备选源，尝试备选源
        for i, backup in enumerate(self.backups):
            try:
                backup_data = backup.get_index_prices(index_id, start_date, end_date)
                logger.info(f"主源失败，使用备选数据源 {i+1} 的数据")
                return backup_data
            except Exception as e:
                logger.warning(f"备选数据源 {i+1} 获取失败: {e}")
                continue
        
        # 6. 所有数据源都失败
        raise ValueError(f"所有数据源都无法获取 {index_id} 的数据")
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        数据质量验证
        
        Args:
            data: 待验证的数据
            
        Returns:
            DataQualityReport: 数据质量报告
        """
        if data is None or data.empty:
            return DataQualityReport(
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                outliers_detected=0,
                total_rows=0,
                missing_values=0,
                overall_score=0.0
            )
        
        total_rows = len(data)
        
        # 计算缺失值
        missing_values = data.isnull().sum().sum()
        
        # 完整性评分 (基于缺失值比例)
        completeness_score = 1.0 - (missing_values / (total_rows * len(data.columns))) if total_rows > 0 else 0.0
        
        # 一致性评分 (检查数据类型一致性)
        consistency_score = self._calculate_consistency_score(data)
        
        # 准确性评分 (基于数据范围检查)
        accuracy_score = self._calculate_accuracy_score(data)
        
        # 异常值检测
        outliers_detected = self._detect_outliers(data)
        
        # 综合评分（加权平均）
        overall_score = (
            0.3 * completeness_score +
            0.3 * consistency_score +
            0.2 * accuracy_score +
            0.2 * (1.0 - min(1.0, outliers_detected / max(1, total_rows)))  # 异常值越少越好
        )
        
        return DataQualityReport(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            outliers_detected=int(outliers_detected),
            total_rows=int(total_rows),
            missing_values=int(missing_values),
            overall_score=overall_score
        )
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算数据一致性评分"""
        if data.empty:
            return 0.0
            
        # 检查数值列的数据类型一致性
        numeric_columns = data.select_dtypes(include=['number']).columns
        if len(numeric_columns) == 0:
            return 0.5  # 如果没有数值列，给中等分数
            
        # 检查是否有混合类型的数据
        consistency_issues = 0
        for col in numeric_columns:
            if data[col].dtype == 'object':
                # 尝试转换为数值
                try:
                    pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    consistency_issues += 1
        
        # 计算一致性评分
        consistency_score = 1.0 - (consistency_issues / len(numeric_columns)) if len(numeric_columns) > 0 else 1.0
        return max(0.0, consistency_score)  # 确保不为负数
    
    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """计算数据准确性评分"""
        if data.empty:
            return 0.0
            
        accuracy_score = 1.0
        
        # 检查价格数据是否合理
        if 'close' in data.columns:
            close_prices = data['close']
            # 检查是否有负价格
            negative_prices = (close_prices < 0).sum()
            if negative_prices > 0:
                accuracy_score -= 0.2 * (negative_prices / len(close_prices))
            
            # 检查是否有异常大的价格
            if len(close_prices) > 0:
                mean_price = close_prices.mean()
                if mean_price > 0:
                    # 检查超过均值10倍的价格
                    extreme_prices = (close_prices > mean_price * 10).sum()
                    if extreme_prices > 0:
                        accuracy_score -= 0.1 * (extreme_prices / len(close_prices))
        
        # 确保评分在0-1范围内
        return max(0.0, min(1.0, accuracy_score))
    
    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """检测异常值数量"""
        if data.empty:
            return 0
            
        outliers = 0
        
        # 对数值列进行异常值检测
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            series = data[col]
            # 使用IQR方法检测异常值
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outliers += col_outliers
            
        return outliers