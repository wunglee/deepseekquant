"""
风险管理系统 - VaR和ES计算器
拆分自: core_bak/risk_manager.py (line 698-900)
职责: 计算在险价值(VaR)和预期短缺(ES)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy import stats
import logging

logger = logging.getLogger('DeepSeekQuant.VarCalculator')


class VarCalculator:
    """VaR和ES计算器 - 单一职责：风险值计算"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化VaR计算器
        
        Args:
            config: 配置字典
        """
        self.var_config = config.get('var', {
            'method': 'historical',
            'confidence_level': 0.95,
            'lookback_period': 504,
            'monte_carlo_simulations': 10000
        })
        
        self.es_config = config.get('expected_shortfall', {
            'method': 'historical',
            'confidence_level': 0.975,
            'lookback_period': 504
        })
        
        logger.info(f"VaR计算器初始化: method={self.var_config['method']}, "
                   f"confidence={self.var_config['confidence_level']}")
    
    def calculate_var(self, returns: pd.Series, 
                     confidence_level: Optional[float] = None,
                     method: Optional[str] = None) -> float:
        """
        计算在险价值(VaR)
        来源: core_bak/risk_manager.py line 698-723
        
        Args:
            returns: 收益序列
            confidence_level: 置信水平
            method: 计算方法 (historical, parametric, monte_carlo)
            
        Returns:
            VaR值 (负数表示损失)
        """
        if confidence_level is None:
            confidence_level = self.var_config['confidence_level']
        if method is None:
            method = self.var_config['method']
            
        try:
            if len(returns) < 20:
                logger.warning("收益数据不足，返回默认VaR")
                return -0.1
            
            if method == 'historical':
                return self._calculate_historical_var(returns, confidence_level)
            elif method == 'parametric':
                return self._calculate_parametric_var(returns, confidence_level)
            elif method == 'monte_carlo':
                # 蒙特卡洛需要协方差矩阵，此方法仅适用于单资产
                return self._calculate_historical_var(returns, confidence_level)
            else:
                return self._calculate_historical_var(returns, confidence_level)
                
        except Exception as e:
            logger.error(f"VaR计算失败: {e}")
            return -0.1
    
    def _calculate_historical_var(self, returns: pd.Series, 
                                  confidence_level: float) -> float:
        """
        历史模拟法计算VaR
        来源: core_bak/risk_manager.py line 724-736
        """
        try:
            if len(returns) < 20:
                return -0.1
            
            # 计算指定置信水平的分位数
            var = np.percentile(returns, (1 - confidence_level) * 100)
            return float(var)
            
        except Exception as e:
            logger.error(f"历史VaR计算失败: {e}")
            return -0.1
    
    def _calculate_parametric_var(self, returns: pd.Series, 
                                  confidence_level: float) -> float:
        """
        参数法计算VaR (正态分布假设)
        来源: core_bak/risk_manager.py line 738-753
        """
        try:
            if len(returns) < 20:
                return -0.1
            
            # 正态分布假设
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = mean_return + z_score * std_return
            
            return float(var)
            
        except Exception as e:
            logger.error(f"参数法VaR计算失败: {e}")
            return -0.1
    
    def calculate_monte_carlo_var(self, mean_returns: np.ndarray,
                                  cov_matrix: np.ndarray,
                                  weights: np.ndarray,
                                  confidence_level: Optional[float] = None,
                                  n_simulations: Optional[int] = None) -> float:
        """
        蒙特卡洛模拟法计算VaR
        来源: core_bak/risk_manager.py line 755-787
        
        Args:
            mean_returns: 资产平均收益向量
            cov_matrix: 协方差矩阵
            weights: 资产权重向量
            confidence_level: 置信水平
            n_simulations: 模拟次数
            
        Returns:
            VaR值
        """
        if confidence_level is None:
            confidence_level = self.var_config['confidence_level']
        if n_simulations is None:
            n_simulations = self.var_config['monte_carlo_simulations']
            
        try:
            if n_simulations < 1000:
                n_simulations = 1000
            
            # 生成随机收益
            np.random.seed(42)  # 可重复性
            simulated_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, n_simulations
            )
            
            # 计算组合收益
            portfolio_simulated_returns = simulated_returns @ weights
            
            # 计算VaR
            var = np.percentile(portfolio_simulated_returns, 
                              (1 - confidence_level) * 100)
            return float(var)
            
        except Exception as e:
            logger.error(f"蒙特卡洛VaR计算失败: {e}")
            return -0.1
    
    def calculate_expected_shortfall(self, returns: pd.Series,
                                    var_value: Optional[float] = None,
                                    confidence_level: Optional[float] = None) -> float:
        """
        计算预期短缺(ES/CVaR)
        来源: core_bak/risk_manager.py line 789-813
        
        Args:
            returns: 收益序列
            var_value: 已计算的VaR值 (可选)
            confidence_level: 置信水平
            
        Returns:
            ES值 (负数表示损失)
        """
        if confidence_level is None:
            confidence_level = self.es_config['confidence_level']
            
        try:
            if len(returns) < 20:
                logger.warning("收益数据不足，返回默认ES")
                return -0.15
            
            # 如果没有提供VaR，先计算
            if var_value is None:
                var_value = self.calculate_var(returns, confidence_level)
            
            # 计算超过VaR的平均损失
            tail_returns = returns[returns <= var_value]
            if len(tail_returns) > 0:
                es = np.mean(tail_returns)
            else:
                es = var_value * 1.2  # 保守估计
            
            return float(es)
            
        except Exception as e:
            logger.error(f"预期短缺计算失败: {e}")
            return -0.15
    
    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算最大回撤
        来源: core_bak/risk_manager.py line 815-835
        
        Args:
            returns: 收益序列
            
        Returns:
            最大回撤值 (负数)
        """
        try:
            if len(returns) < 20:
                logger.warning("收益数据不足，返回默认回撤")
                return -0.2
            
            # 计算累积收益
            cumulative_returns = np.cumprod(1 + returns) - 1
            
            # 计算回撤
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / (1 + peak)
            
            max_drawdown = np.min(drawdown)
            return float(max_drawdown)
            
        except Exception as e:
            logger.error(f"最大回撤计算失败: {e}")
            return -0.2
