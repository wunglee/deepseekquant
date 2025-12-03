"""
数据分析通用工具

职责：
- 时间序列对齐
- 类型保障转换
- 数据有效性验证

来源：从 core/data/data_utils.py 迁移而来（属于跨模块共享的业务能力）

使用方：
- core/risk/risk_preprocessor.py
- core/backtest（事件分析）
- core/portfolio（风险计算）
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger('DeepSeekQuant.DataAnalysisUtils')


class DataAnalysisUtils:
    """
    数据分析通用工具类
    
    提供跨模块共享的数据处理能力
    """
    
    @staticmethod
    def align_time_series(
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        对齐两个时间序列（按索引交集）
        
        业务规则：
        - 优先按索引交集对齐
        - 无交集时按长度对齐（取尾部）并重置索引
        
        Args:
            series1: 第一个序列
            series2: 第二个序列
        
        Returns:
            (对齐后的序列1, 对齐后的序列2)
        
        Examples:
            >>> s1 = pd.Series([1, 2, 3], index=['2020-01-01', '2020-01-02', '2020-01-03'])
            >>> s2 = pd.Series([4, 5], index=['2020-01-02', '2020-01-03'])
            >>> aligned1, aligned2 = DataAnalysisUtils.align_time_series(s1, s2)
            >>> # 返回交集: ['2020-01-02', '2020-01-03']
        """
        # 取交集索引
        common_index = series1.index.intersection(series2.index)
        
        if len(common_index) == 0:
            # 业务规则：无交集时按长度对齐
            logger.warning("两个序列无交集，按长度对齐（取尾部）")
            min_len = min(len(series1), len(series2))
            return (
                series1.iloc[-min_len:].reset_index(drop=True),
                series2.iloc[-min_len:].reset_index(drop=True)
            )
        
        return series1.loc[common_index], series2.loc[common_index]
    
    @staticmethod
    def ensure_series(data: any, name: str = "data") -> pd.Series:
        """
        确保数据是 pandas Series 类型（容错转换）
        
        支持类型：
        - pd.Series: 直接返回
        - list/np.ndarray: 转换为Series
        - 其他类型: 尝试转换，失败时记录警告
        
        Args:
            data: 输入数据
            name: 数据名称（用于日志）
        
        Returns:
            pandas Series
        
        Examples:
            >>> data = [1, 2, 3]
            >>> series = DataAnalysisUtils.ensure_series(data)
            >>> isinstance(series, pd.Series)
            True
        """
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, (list, np.ndarray)):
            return pd.Series(data)
        else:
            # 降级日志：类型异常但尝试转换
            logger.warning(f"{name}类型无效: {type(data)}，尝试转换为Series")
            return pd.Series(data)
    
    @staticmethod
    def validate_dataframe(
        data: pd.DataFrame,
        required_columns: list = None,
        min_rows: int = 1
    ) -> Tuple[bool, str]:
        """
        验证DataFrame的有效性（接口前置校验）
        
        校验规则：
        - 必须是有效的DataFrame
        - 行数满足最小要求
        - 包含所有必需列
        
        Args:
            data: 待验证的DataFrame
            required_columns: 必需的列名列表
            min_rows: 最小行数
        
        Returns:
            (是否有效, 错误信息)
        
        Examples:
            >>> df = pd.DataFrame({'close': [100, 110]})
            >>> valid, msg = DataAnalysisUtils.validate_dataframe(df, ['close'], min_rows=2)
            >>> assert valid
        """
        if data is None or not isinstance(data, pd.DataFrame):
            return False, "数据不是有效的DataFrame"
        
        if len(data) < min_rows:
            return False, f"数据行数不足：{len(data)} < {min_rows}"
        
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                return False, f"缺少必需列：{missing_cols}"
        
        return True, ""
