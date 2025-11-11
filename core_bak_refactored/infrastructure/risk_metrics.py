"""
风险指标计算 - 基础设施层
从 core_bak/risk_manager.py 拆分
职责: 提供通用的风险指标计算函数（VaR、CVaR、波动率等）
"""

import numpy as np

from typing import Dict, List, Optional, Tuple
import scipy.stats as stats
import logging

logger = logging.getLogger('DeepSeekQuant.Infrastructure.RiskMetrics')


class StatisticalCalculator:
    """通用统计计算器（纯数学/统计），不包含业务术语或年化逻辑
    
    设计原则：
    - 纯numpy实现，不依赖pandas
    - 所有方法为纯函数，无副作用
    - 仅提供数学计算，业务逻辑由上层处理
    """
    
    @staticmethod
    def calculate_standard_deviation(returns: np.ndarray, window: Optional[int] = None, ddof: int = 1) -> float:
        """
        计算标准差（纯数学），不做年化
        
        Args:
            returns: 收益率序列
            window: 可选窗口大小；None表示使用全序列
            ddof: 自由度，默认1为样本标准差
        """
        if returns is None or len(returns) == 0:
            return 0.0
        if window is not None:
            if len(returns) < window:
                return 0.0
            return float(np.std(returns[-window:], ddof=ddof))
        return float(np.std(returns, ddof=ddof))
    
    @staticmethod
    def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """
        计算对数收益率（纯数学）
        
        公式：log_return[i] = log(price[i]) - log(price[i-1]) = log(price[i] / price[i-1])
        
        Args:
            prices: 价格序列
            
        Returns:
            对数收益率序列（长度比价格序列少1）
        """
        if prices is None or len(prices) < 2:
            return np.array([])
        return np.diff(np.log(prices))
    
    @staticmethod
    def calculate_simple_returns(prices: np.ndarray) -> np.ndarray:
        """
        计算简单收益率（纯数学）
        
        公式：simple_return[i] = (price[i] - price[i-1]) / price[i-1] = price[i] / price[i-1] - 1
        
        数学特性：
        - 与对数收益率的区别：r_simple = exp(r_log) - 1
        - 可加性：组合收益可以通过个股简单收益加权平均计算
        - 对称性：+10%/-10%不对称（对数收益对称）
        
        Args:
            prices: 价格序列
            
        Returns:
            简单收益率序列（长度比价格序列少1）
            
        Note:
            如果价格序列包含非正值，结果可能包含NaN或Inf
        """
        if prices is None or len(prices) < 2:
            return np.array([])
        
        # 检查价格有效性（记录警告但不中断计算）
        if np.any(prices <= 0):
            logger.warning(
                "价格序列包含非正值，简单收益率可能产生异常值（Inf/NaN）。"
                f"非正值数量: {np.sum(prices <= 0)}/{len(prices)}"
            )
        
        # 使用 (p[t] / p[t-1]) - 1 计算简单收益
        return (prices[1:] / prices[:-1]) - 1
    
    @staticmethod
    def calculate_quantile(values: np.ndarray, q: float, interpolation: str = 'linear') -> float:
        """
        计算分位数（纯数学）
        
        Args:
            values: 数值序列
            q: 分位点 (0-1)
            interpolation: 插值方法 'linear', 'lower', 'higher', 'nearest', 'midpoint'
        """
        if values is None or len(values) == 0:
            return 0.0
        # 使用百分位形式计算
        return float(np.quantile(values, q, method='linear' if interpolation == 'linear' else interpolation))
    
    @staticmethod
    def calculate_cvar(values: np.ndarray, confidence_level: float = 0.95) -> float:
        """
        计算条件在险价值（CVaR/ES）的数学形式：VaR以下值的均值
        """
        if values is None or len(values) == 0:
            return 0.0
        var = StatisticalCalculator.calculate_quantile(values, 1 - confidence_level)
        tail = values[values <= var]
        return float(np.mean(tail)) if len(tail) > 0 else float(var)
    
    @staticmethod
    def calculate_cumulative_peak_deviation(values: np.ndarray) -> np.ndarray:
        """
        计算相对于累积峰值的偏离（纯数学）
        返回偏离序列（值≤0），业务层据此计算最大回撤
        
        Args:
            values: 累积值序列，如累计收益或价格累计乘积
        """
        if values is None or len(values) == 0:
            return np.array([])
        cummax = np.maximum.accumulate(values)
        return values - cummax
    
    @staticmethod
    def calculate_covariance_matrix(data: np.ndarray) -> np.ndarray:
        """计算协方差矩阵（纯数学）
        
        Args:
            data: 二维数组，每列为一个变量
            
        Returns:
            协方差矩阵
            
        Note:
            如果矩阵接近奇异（条件数过大），会记录警告日志
        """
        if data is None or data.size == 0:
            return np.array([])
        
        cov_matrix = np.cov(data, rowvar=False)
        
        # 检测矩阵条件数（数值稳定性指标）
        if cov_matrix.size > 0 and cov_matrix.ndim == 2:
            try:
                condition_number = np.linalg.cond(cov_matrix)
                if condition_number > 1e6:
                    logger.warning(
                        f"协方差矩阵接近奇异，条件数={condition_number:.2e}（阈值1e6）。"
                        "可能存在完全共线的变量，数值计算可能不稳定。"
                    )
            except np.linalg.LinAlgError:
                logger.warning("协方差矩阵奇异，无法计算条件数。")
        
        return cov_matrix

    @staticmethod
    def calculate_correlation_matrix(data: np.ndarray) -> np.ndarray:
        """计算相关系数矩阵（纯数学）
        
        Args:
            data: 二维数组，每列为一个变量
        """
        if data is None or data.size == 0:
            return np.array([])
        return np.corrcoef(data, rowvar=False)

    @staticmethod
    def calculate_covariance_variance_ratio(series1: np.ndarray, series2: np.ndarray) -> float:
        """计算协方差/方差比率（β的数学形式）
        
        公式：cov(X, Y) / var(Y)
        """
        if series1 is None or series2 is None or len(series1) == 0 or len(series1) != len(series2):
            return np.nan
        cov = np.cov(series1, series2)[0, 1]
        var = np.var(series2)
        return float(cov / var) if var != 0 else np.nan
    
    @staticmethod
    def calculate_downside_deviation(values: np.ndarray, baseline: float = 0.0, ddof: int = 1) -> float:
        """计算下行标准差（标准半方差公式）
        
        公式：sqrt(sum(min(values - baseline, 0)^2) / (n - ddof))
        注意：这是标准的半方差（semi-variance）计算方法，而非简单的下行值标准差
        
        数学定义：
        - 仅计算低于基准值的偏差的平方均值再开方
        - 与全样本标准差的区别：只考虑负偏差（values < baseline）
        - ddof=1时为样本半方差，ddof=0时为总体半方差
        
        Args:
            values: 数值序列
            baseline: 基准值（用于判断偏差方向的阈值，默认0）
            ddof: 自由度，默认1为样本半方差
        """
        if values is None or len(values) == 0:
            return 0.0
        # 计算超额收益（负数为下行）
        excess = values - baseline
        downside = np.minimum(excess, 0)  # 取负超额收益
        # 计算半方差：下行偏差的平方均值
        n = len(values)
        if n <= ddof:
            return 0.0
        semi_variance = np.sum(downside**2) / (n - ddof)
        return float(np.sqrt(semi_variance))
    
    @staticmethod
    def calculate_residual(y: np.ndarray, x: np.ndarray, slope: float) -> np.ndarray:
        """计算线性回归残差（纯数学）
        
        公式：residual = y - slope * x
        
        Args:
            y: 因变量
            x: 自变量
            slope: 斜率（β系数）
        """
        if y is None or x is None or len(y) != len(x):
            return np.array([])
        return y - slope * x
    
    @staticmethod
    def calculate_correlation(returns1: np.ndarray, 
                              returns2: np.ndarray) -> float:
        """
        计算相关系数
        
        Args:
            returns1: 收益率序列1
            returns2: 收益率序列2
            
        Returns:
            相关系数
        """
        if len(returns1) != len(returns2) or len(returns1) == 0:
            return 0.0
        
        return float(np.corrcoef(returns1, returns2)[0, 1])
    
    @staticmethod
    def calculate_mean_std_ratio(values: np.ndarray, baseline: float = 0.0, ddof: int = 1) -> float:
        """
        计算均值与标准差比率（纯数学）
        不做年化，业务层自行处理
        """
        if values is None or len(values) == 0:
            return 0.0
        excess = values - baseline
        std = np.std(excess, ddof=ddof)
        if std == 0:
            return 0.0
        return float(np.mean(excess) / std)
    
    @staticmethod
    def calculate_tail_risk(returns: np.ndarray, 
                           threshold: float = -0.05) -> float:
        """
        计算尾部风险
        
        Args:
            returns: 收益率序列
            threshold: 阈值
            
        Returns:
            尾部风险概率
        """
        if len(returns) == 0:
            return 0.0
        
        tail_events = returns[returns < threshold]
        return float(len(tail_events) / len(returns))
