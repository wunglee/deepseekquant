"""[专家碎片] 数据质量验证器 - 第6轮专家指导(迁移至 quality 子域)

架构变更说明（2025-12-06）：
- 原名：DataQualityEnhancer（数据质量增强器）
- 原职责：多源数据智能切换、质量驱动切换、主源优先备源降级
- 新职责：单纯的数据质量验证与评分，不再负责数据源切换
- 设计原则：单一职责原则（SRP），仅评估数据质量，不参与数据获取决策

新架构规则（2025-12-06）：
- 用户指定单一数据源（primary_source），不再自动切换
- 本类仅负责数据质量评分，返回质量报告供调用方决策
- 移除了backup_sources参数和多源切换逻辑

职责:
- 数据质量验证与评分
- 完整的质量评分体系(完整性+一致性+准确性+异常检测)
- 提供标准化的质量报告

设计模式:
- 策略模式: 支持多种质量评估策略
- 报告模式: 标准化的质量报告输出
"""

import logging
import pandas as pd
import yaml
from typing import List, Dict, Any, Union, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from core_bak_refactored.core.data.quality.quality_types import DataQualityReport
from core_bak_refactored.core.data.quality.data_quality_utils import calculate_consistency_score, calculate_accuracy_score, detect_outliers

logger = logging.getLogger('DeepSeekQuant.DataQualityEnhancer')


def _load_data_quality_config() -> Dict[str, Any]:
    """从配置文件加载数据质量配置
    
    Returns:
        Dict: 数据质量配置字典
    
    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置文件格式错误或缺少必需字段
    """
    config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'dev' / 'data_quality.yml'
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"数据质量配置文件不存在: {config_path}\n"
            f"请确保配置文件存在: core_bak_refactored/config/dev/data_quality.yml"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证必需字段
        if 'quality_threshold' not in config:
            raise ValueError("配置文件缺少必需字段: quality_threshold")
        if 'min_data_rows' not in config:
            raise ValueError("配置文件缺少必需字段: min_data_rows")
        if 'score_weights' not in config:
            raise ValueError("配置文件缺少必需字段: score_weights")
        
        return config
    
    except yaml.YAMLError as e:
        raise ValueError(f"数据质量配置文件格式错误: {e}")


# 从配置文件加载常量
_DATA_QUALITY_CONFIG = _load_data_quality_config()
DEFAULT_QUALITY_THRESHOLD = _DATA_QUALITY_CONFIG['quality_threshold']
MIN_DATA_ROWS = _DATA_QUALITY_CONFIG['min_data_rows']
SCORE_WEIGHTS = _DATA_QUALITY_CONFIG['score_weights']


class DataQualityEnhancer:
    """数据质量验证器
    
    新架构设计（2025-12-06）：
    - 仅负责对数据进行质量评估
    - 返回DataQualityReport供调用方决策
    
    职责:
    - 数据质量验证与评分
    - 提供标准化的质量报告
    
    Args:
        quality_threshold: 质量阈值(默认0.8)，仅用于报告对比
    """

    def __init__(
        self,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    ):
        """初始化数据质量验证器
        
        Args:
            quality_threshold: 质量阈值(0-1)
        """
        if quality_threshold < 0 or quality_threshold > 1:
            raise ValueError(f"quality_threshold必须在0-1之间,当前值: {quality_threshold}")
        
        self.quality_threshold = quality_threshold
        
        logger.info(
            f"DataQualityEnhancer初始化: 质量阈值={quality_threshold} "
            "(仅用于报告对比)"
        )

    
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
                missing_values=0
            )
        # 计算各维度评分
        completeness_score, total_rows, missing_values = self._validate_data_completeness(data)
        consistency_score = calculate_consistency_score(data)
        accuracy_score = calculate_accuracy_score(data)
        outliers_detected = detect_outliers(data)
        
        report = DataQualityReport(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            outliers_detected=int(outliers_detected),
            total_rows=int(total_rows),
            missing_values=int(missing_values)
        )
        
        logger.debug(
            f"质量评分: 总分={report.overall_score:.3f}, "
            f"完整性={completeness_score:.3f}, "
            f"一致性={consistency_score:.3f}, "
            f"准确性={accuracy_score:.3f}, "
            f"异常值={outliers_detected}"
        )
        
        return report
    

    def _validate_data_completeness(self, data: pd.DataFrame) -> Tuple[float, int, int]:
        """验证数据完整性
        
        Args:
            data: 待验证的DataFrame
            
        Returns:
            Tuple[完整性评分, 总行数, 缺失值总数]
        """
        if data.empty:
            return 0.0, 0, 0
            
        total_rows = len(data)
        total_cells = total_rows * len(data.columns)
        missing_values = data.isnull().sum().sum()
        completeness_score = 1.0 - (missing_values / total_cells) if total_cells > 0 else 0.0
        
        return completeness_score, total_rows, int(missing_values)
