"""
风险计算器 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 协调器 - 统一风险计算入口，委托给业务服务层
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

from .risk_metrics_service import RiskMetricsService
from .risk_models import RiskMetric

# 导入数据预处理器
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from infrastructure.data_preprocessor import RiskDataPreprocessor

logger = logging.getLogger('DeepSeekQuant.RiskCalculator')


class RiskCalculator:
    """
    风险计算器 - 纯协调器
    
    职责：
    - 提供统一的风险计算入口
    - 委托给 RiskMetricsService 进行实际计算
    - 使用 RiskDataPreprocessor 处理数据提取
    
    设计原则：
    - 不实现具体算法，仅负责委托
    - 不直接处理数据，委托给预处理器
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_metrics_service = RiskMetricsService(config)
        self.preprocessor = RiskDataPreprocessor()
        logger.info("风险计算器初始化完成")
    
    def calculate_volatility(self, returns: pd.Series, window: Optional[int] = None, annualize: bool = True) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_volatility(returns, window, annualize)

    def calculate_correlation_matrix(self, asset_returns: pd.DataFrame) -> pd.DataFrame:
        """相关性矩阵"""
        if asset_returns is None or asset_returns.empty:
            return pd.DataFrame()
        return asset_returns.corr().fillna(0.0)

    def calculate_var_historical(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_value_at_risk(returns, confidence_level, 'historical')

    def calculate_var_parametric(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_value_at_risk(returns, confidence_level, 'parametric')


    def calculate_var_monte_carlo(self, portfolio_state, market_data: Dict[str, Any], confidence_level: float) -> float:
        """
        蒙特卡洛法VaR（简化实现）
        
        注：此方法待移至 RiskMetricsService，当前保留兼容性
        """
        logger.warning("蒙特卡洛 VaR 计算待优化，当前使用简化实现")
        try:
            n_simulations = int(self.config.get('monte_carlo_sims', 1000))
            if n_simulations < 1000:
                n_simulations = 1000
            symbols = list(portfolio_state.allocations.keys())
            returns_data = {}
            for symbol in symbols:
                prices = market_data['prices'][symbol].get('close', [])
                if len(prices) >= 20:
                    # 使用预处理器计算收益
                    returns_data[symbol] = self.preprocessor.extract_returns_from_prices(np.array(prices))
            if not returns_data:
                return 0.0
            min_len = min(len(v) for v in returns_data.values())
            aligned = np.column_stack([v[-min_len:] for v in returns_data.values()])
            mean_vec = aligned.mean(axis=0)
            cov_mat = np.cov(aligned.T)
            np.random.seed(42)
            sims = np.random.multivariate_normal(mean_vec, cov_mat, n_simulations)
            weights = np.array([alloc.weight for alloc in portfolio_state.allocations.values()])
            portfolio_sims = sims @ weights
            var = np.percentile(portfolio_sims, (1 - confidence_level) * 100)
            return float(var)
        except Exception:
            return 0.0


    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """委托给 RiskMetricsService"""
        return self.risk_metrics_service.calculate_max_drawdown(returns)
    
    def calculate_all_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        计算所有风险指标
        
        职责：
        - 委托 RiskDataPreprocessor 提取数据
        - 委托 RiskMetricsService 计算指标
        """
        try:
            # 数据提取委托给预处理器
            returns = self.preprocessor.extract_returns_from_dict(data)
            market_returns = self.preprocessor.extract_market_returns_from_dict(data)
            
            # 验证数据有效性
            if not self.preprocessor.validate_returns_data(returns, min_length=20):
                logger.warning("收益数据不足，无法计算风险指标")
                return {}
            
            # 计算委托给服务层
            return self.risk_metrics_service.calculate_all_metrics(returns, market_returns)
            
        except Exception as e:
            logger.error(f"风险指标计算失败: {e}")
            return {}
    
    def simulate_correlation_breakdown(self, scenario, portfolio_state, market_data):
        """迁移到 StressTester，暂不在此实现"""
        raise NotImplementedError("Use StressTester.simulate_correlation_breakdown")


