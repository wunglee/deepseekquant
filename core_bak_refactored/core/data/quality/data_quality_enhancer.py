"""[专家碎片] 数据质量增强器 - 第6轮专家指导(迁移至 quality 子域)

职责:
- 多源数据智能切换(基于质量评分)
- 数据质量验证与评分
- 质量对比选择(选择质量最高的源)

设计原则:
- 主源优先, 备源降级
- 质量驱动切换(阈值可配置)
- 完整的质量评分体系(完整性+一致性+准确性+异常检测)
"""

import logging
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger('DeepSeekQuant.DataQualityEnhancer')

# 数据质量阈值常量
DEFAULT_QUALITY_THRESHOLD = 0.8  # 默认质量阈值
MIN_DATA_ROWS = 1  # 最少数据行数


@dataclass
class DataQualityReport:
    """数据质量报告
    
    Attributes:
        completeness_score: 完整性评分(0-1)
        consistency_score: 一致性评分(0-1)
        accuracy_score: 准确性评分(0-1)
        outliers_detected: 检测到的异常值数量
        total_rows: 总行数
        missing_values: 缺失值总数
        overall_score: 综合评分(0-1)
    """
    completeness_score: float
    consistency_score: float
    accuracy_score: float
    outliers_detected: int
    total_rows: int
    missing_values: int
    overall_score: float

class DataQualityEnhancer:
    """数据质量增强器
    
    职责:
    - 从多个数据源获取数据
    - 基于质量评分选择最佳数据源
    - 主源质量不足时自动切换到备源
    
    设计模式:
    - 策略模式: 支持多种数据源策略
    - 责任链模式: 主源失败逐级尝试备源
    
    Args:
        primary_source: 主数据源(实现HistoricalDataProvider接口)
        backup_sources: 备用数据源列表
        quality_threshold: 质量阈值(默认0.8),低于此值触发备源切换
    """

    def __init__(
        self,
        primary_source,
        backup_sources: List = None,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    ):
        """初始化数据质量增强器
        
        Args:
            primary_source: 主数据源提供者
            backup_sources: 备用数据源列表
            quality_threshold: 质量阈值(0-1)
        """
        if quality_threshold < 0 or quality_threshold > 1:
            raise ValueError(f"quality_threshold必须在0-1之间,当前值: {quality_threshold}")
        
        self.primary = primary_source
        self.backups = backup_sources or []
        self.quality_threshold = quality_threshold
        
        logger.info(
            f"DataQualityEnhancer初始化: 主源={type(primary_source).__name__}, "
            f"备源数量={len(self.backups)}, 质量阈值={quality_threshold}"
        )
    
    def get_enhanced_prices(
        self,
        index_id: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """获取质量增强的价格数据
        
        流程:
        1. 从主源获取数据
        2. 评估数据质量
        3. 质量不足时尝试备源
        4. 返回最佳质量的数据+质量报告
        
        Args:
            index_id: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            (数据DataFrame, 质量报告DataQualityReport)元组
        
        Raises:
            ValueError: 所有数据源都无法获取数据
        """
        logger.info(f"开始获取{index_id}数据: {start_date} 到 {end_date}")
        
        # 尝试从主数据源获取
        primary_data = None
        primary_quality = None
        
        try:
            primary_data = self.primary.get_index_prices(index_id, start_date, end_date)
            if primary_data is not None and not primary_data.empty:
                primary_quality = self.validate_data_quality(primary_data)
                logger.info(
                    f"主源数据: {len(primary_data)}行, 质量评分={primary_quality.overall_score:.3f}"
                )
            else:
                logger.warning("主数据源返回空数据")
        except Exception as e:
            logger.error(f"主数据源获取失败: {type(e).__name__}: {e}")
        
        # 如果主源质量达标,直接返回
        if primary_data is not None and not primary_data.empty:
            if primary_quality.overall_score >= self.quality_threshold:
                logger.info(f"主源质量达标({primary_quality.overall_score:.3f}>={self.quality_threshold}),直接使用")
                return primary_data, primary_quality
            else:
                logger.warning(
                    f"主源质量不足({primary_quality.overall_score:.3f}<{self.quality_threshold}),尝试备源"
                )
                # 尝试备选数据源
                best_backup_data = None
                best_backup_quality = primary_quality
                
                for i, backup in enumerate(self.backups):
                    try:
                        backup_data = backup.get_index_prices(index_id, start_date, end_date)
                        if backup_data is None or backup_data.empty:
                            logger.debug(f"备源{i+1}返回空数据")
                            continue
                        
                        backup_quality = self.validate_data_quality(backup_data)
                        logger.info(
                            f"备源{i+1}: {len(backup_data)}行, 质量={backup_quality.overall_score:.3f}"
                        )
                        
                        # 选择质量最高的备源
                        if backup_quality.overall_score > best_backup_quality.overall_score:
                            best_backup_data = backup_data
                            best_backup_quality = backup_quality
                            logger.info(f"备源{i+1}质量更优,暂存为最佳备源")
                    except Exception as e:
                        logger.error(f"备源{i+1}获取失败: {type(e).__name__}: {e}")
                        continue
                
                # 如果找到更好的备源,使用备源数据
                if best_backup_data is not None and best_backup_quality.overall_score > primary_quality.overall_score:
                    logger.info(
                        f"使用备源数据(质量{best_backup_quality.overall_score:.3f} > "
                        f"主源{primary_quality.overall_score:.3f})"
                    )
                    return best_backup_data, best_backup_quality
        
        # 主源可用(即使质量不足),优先返回主源
        if primary_data is not None:
            return primary_data, primary_quality if primary_quality else self.validate_data_quality(primary_data)
        # 主源失败,逐个尝试备源
        logger.warning("主源失败,逐个尝试备源")
        for i, backup in enumerate(self.backups):
            try:
                backup_data = backup.get_index_prices(index_id, start_date, end_date)
                if backup_data is not None and not backup_data.empty:
                    backup_quality = self.validate_data_quality(backup_data)
                    logger.info(f"备源{i+1}成功: {len(backup_data)}行, 质量={backup_quality.overall_score:.3f}")
                    return backup_data, backup_quality
            except Exception as e:
                logger.error(f"备源{i+1}失败: {type(e).__name__}: {e}")
                continue
        
        # 所有数据源都失败
        error_msg = f"所有数据源({1 + len(self.backups)})都无法获取{index_id}的数据"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def validate_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """验证数据质量
        
        评分维度:
        - 完整性(30%):缺失值比例
        - 一致性(30%):数据类型一致性
        - 准确性(20%):负价格, 极端价格检测
        - 异常值(20%):IQR方法检测异常值
        
        Args:
            data: 待验证的DataFrame
        
        Returns:
            DataQualityReport对象
        """
        # 空数据处理
        if data is None or data.empty:
            logger.debug("数据为空,返回零分质量报告")
            return DataQualityReport(
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                outliers_detected=0,
                total_rows=0,
                missing_values=0,
                overall_score=0.0
            )
        # 计算各维度评分
        total_rows = len(data)
        total_cells = total_rows * len(data.columns)
        missing_values = data.isnull().sum().sum()
        
        completeness_score = (
            1.0 - (missing_values / total_cells)
            if total_cells > 0 else 0.0
        )
        consistency_score = self._calculate_consistency_score(data)
        accuracy_score = self._calculate_accuracy_score(data)
        outliers_detected = self._detect_outliers(data)
        
        # 综合评分(加权平均)
        outlier_penalty = min(1.0, outliers_detected / max(1, total_rows))
        overall_score = (
            0.3 * completeness_score +
            0.3 * consistency_score +
            0.2 * accuracy_score +
            0.2 * (1.0 - outlier_penalty)
        )
        
        report = DataQualityReport(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            outliers_detected=int(outliers_detected),
            total_rows=int(total_rows),
            missing_values=int(missing_values),
            overall_score=overall_score
        )
        
        logger.debug(
            f"质量评分: 总分={overall_score:.3f}, "
            f"完整性={completeness_score:.3f}, "
            f"一致性={consistency_score:.3f}, "
            f"准确性={accuracy_score:.3f}, "
            f"异常值={outliers_detected}"
        )
        
        return report
    
    def _calculate_consistency_score(self, data: pd.DataFrame) -> float:
        """计算数据一致性评分
        
        检查数值列的类型一致性
        
        Args:
            data: 待检查的DataFrame
        
        Returns:
            一致性评分(0-1)
        """
        if data.empty:
            return 0.0
        numeric_columns = data.select_dtypes(include=['number']).columns
        if len(numeric_columns) == 0:
            return 0.5
        consistency_issues = 0
        for col in numeric_columns:
            if data[col].dtype == 'object':
                try:
                    pd.to_numeric(data[col], errors='raise')
                except (ValueError, TypeError):
                    consistency_issues += 1
        consistency_score = 1.0 - (consistency_issues / len(numeric_columns)) if len(numeric_columns) > 0 else 1.0
        return max(0.0, consistency_score)
    
    def _calculate_accuracy_score(self, data: pd.DataFrame) -> float:
        """计算数据准确性评分
        
        检查项:
        - 负价格(扣0.2 * 比例)
        - 极端价格(超过均值10倍,扣0.1 * 比例)
        
        Args:
            data: 待检查的DataFrame
        
        Returns:
            准确性评分(0-1)
        """
        if data.empty:
            return 0.0
        accuracy_score = 1.0
        if 'close' in data.columns:
            close_prices = data['close']
            negative_prices = (close_prices < 0).sum()
            if negative_prices > 0:
                accuracy_score -= 0.2 * (negative_prices / len(close_prices))
            if len(close_prices) > 0:
                mean_price = close_prices.mean()
                if mean_price > 0:
                    extreme_prices = (close_prices > mean_price * 10).sum()
                    if extreme_prices > 0:
                        accuracy_score -= 0.1 * (extreme_prices / len(close_prices))
        return max(0.0, min(1.0, accuracy_score))
    
    def _detect_outliers(self, data: pd.DataFrame) -> int:
        """检测异常值(IQR方法)
        
        使用四分位距(IQR)检测异常值:
        - 异常值定义:< Q1 - 1.5*IQR 或 > Q3 + 1.5*IQR
        
        Args:
            data: 待检测的DataFrame
        
        Returns:
            异常值总数
        """
        if data.empty:
            return 0
        outliers = 0
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            series = data[col]
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = ((series < lower_bound) | (series > upper_bound)).sum()
            outliers += col_outliers
        return outliers
