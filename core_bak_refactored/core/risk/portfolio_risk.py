"""
组合风险分析 - 业务层
从 core_bak/risk_manager.py 拆分
职责: 组合层面的风险分析、风险贡献度
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .risk_metrics_service import RiskMetricsService
from infrastructure.risk_metrics import StatisticalCalculator

logger = logging.getLogger('DeepSeekQuant.PortfolioRisk')


class PortfolioRiskAnalyzer:
    """组合风险分析器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_metrics_service = RiskMetricsService(config)
    
    def calculate_portfolio_returns(self, portfolio_state, market_data: Dict[str, Any]) -> pd.Series:
        """计算组合收益序列"""
        try:
            # 获取组合中所有资产
            symbols = list(portfolio_state.allocations.keys())
            if not symbols:
                logger.warning("组合中没有资产")
                return pd.Series()

            # 获取价格数据并确保时间对齐
            price_data = {}
            min_length = float('inf')

            for symbol in symbols:
                if symbol in market_data['prices']:
                    closes = market_data['prices'][symbol].get('close', [])
                    if len(closes) > 0:
                        price_data[symbol] = closes
                        min_length = min(min_length, len(closes))
                    else:
                        logger.warning(f"符号 {symbol} 没有价格数据，跳过")
                        continue
                else:
                    logger.warning(f"市场数据中缺少符号 {symbol}，跳过")
                    continue

            # 若所有资产均不可用，返回空序列
            if not price_data:
                logger.warning("有效价格数据为空，返回空序列")
                return pd.Series()
            if min_length < 2:
                logger.warning("价格数据不足")
                return pd.Series()

            # 截取相同长度的价格序列
            aligned_prices = {}
            for symbol, prices in price_data.items():
                aligned_prices[symbol] = prices[-min_length:]

            # 计算对数收益（使用基础设施层统一方法）
            returns_data = {}
            for symbol, prices in aligned_prices.items():
                log_returns = StatisticalCalculator.calculate_log_returns(np.array(prices))
                returns_data[symbol] = log_returns

            # 创建DataFrame
            returns_df = pd.DataFrame(returns_data)

            # 获取用于计算的符号列
            symbols_used = list(returns_df.columns)
            # 获取权重（仅针对有效符号）
            weights = np.array([portfolio_state.allocations[s].weight for s in symbols_used])

            # 计算加权组合收益
            portfolio_returns = returns_df.dot(weights)

            # 转换为Series并设置时间索引
            if 'timestamp' in market_data and len(market_data['timestamp']) >= len(portfolio_returns):
                # 使用最后一个时间戳作为索引（假设时间戳是升序排列）
                timestamps = market_data['timestamp'][-len(portfolio_returns):]
                portfolio_returns = pd.Series(portfolio_returns.values, index=timestamps)
            else:
                # 使用数值索引作为备选
                portfolio_returns = pd.Series(portfolio_returns.values)

            logger.debug(f"组合收益计算完成: 数据点={len(portfolio_returns)}, 期间={len(portfolio_returns)}天")
            return portfolio_returns

        except Exception as e:
            logger.error(f"组合收益计算失败: {e}")
            return pd.Series()
    
    def calculate_risk_contributions(self, portfolio_state, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算风险贡献度（基于相关性矩阵与权重）"""
        risk_contributions = {}
        try:
            if correlation_matrix is None or correlation_matrix.empty:
                return risk_contributions
            
            symbols = list(portfolio_state.allocations.keys())
            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            
            # 确保相关性矩阵维度匹配
            if len(weights) != correlation_matrix.shape[0]:
                logger.warning("权重与相关性矩阵维度不匹配")
                return risk_contributions
            
            # 计算组合方差
            portfolio_variance = weights.T @ correlation_matrix.values @ weights
            
            if portfolio_variance > 0:
                # 计算边际风险贡献
                marginal_risk = (correlation_matrix.values @ weights) / np.sqrt(portfolio_variance)
                
                # 计算风险贡献度
                for i, symbol in enumerate(symbols):
                    risk_contribution = weights[i] * marginal_risk[i]
                    risk_contributions[symbol] = float(risk_contribution)
            
            return risk_contributions
        
        except Exception as e:
            logger.error(f"风险贡献度计算失败: {e}")
            return {}
    
    def calculate_risk_contributions_covariance(self, portfolio_state, covariance_matrix: pd.DataFrame) -> Dict[str, float]:
        """基于协方差矩阵计算风险贡献度（边际风险贡献法）"""
        contributions: Dict[str, float] = {}
        try:
            if covariance_matrix is None or covariance_matrix.empty:
                return contributions
            symbols = list(portfolio_state.allocations.keys())
            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            if len(weights) != covariance_matrix.shape[0]:
                logger.warning("权重与协方差矩阵维度不匹配")
                return contributions
            portfolio_variance = float(weights.T @ covariance_matrix.values @ weights)
            if portfolio_variance <= 0:
                return contributions
            marginal_risk = (covariance_matrix.values @ weights) / np.sqrt(portfolio_variance)
            for i, symbol in enumerate(symbols):
                contributions[symbol] = float(weights[i] * marginal_risk[i])
            return contributions
        except Exception as e:
            logger.error(f"协方差风险贡献度计算失败: {e}")
            return {}
    
    def calculate_factor_risk_attribution(self, portfolio_state, factor_exposures: pd.DataFrame, 
                                          factor_covariance: pd.DataFrame) -> Dict[str, Any]:
        """
        计算因子级风险归因分解
        
        基于 Barra 风险模型：组合方差 = X' * F * X + Δ
        
        Args:
            portfolio_state: 组合状态
            factor_exposures: 因子暴露矩阵 (N资产 x K因子)
            factor_covariance: 因子协方差矩阵 (K x K)
            
        Returns:
            {
                'market_risk': 市场因子风险,
                'industry_risk': 行业因子风险,
                'style_risk': 风格因子风险,
                'specific_risk': 特质风险,
                'total_risk': 总风险,
                'factor_contributions': 各因子贡献明细
            }
        """
        try:
            if factor_exposures is None or factor_exposures.empty:
                logger.warning("因子暴露数据为空")
                return {}
            
            symbols = list(portfolio_state.allocations.keys())
            weights = np.array([portfolio_state.allocations[symbol].weight for symbol in symbols])
            
            # 确保符号匹配
            common_symbols = [s for s in symbols if s in factor_exposures.index]
            if not common_symbols:
                logger.warning("没有匹配的符号")
                return {}
            
            # 调整权重与暴露
            matched_weights = np.array([portfolio_state.allocations[s].weight for s in common_symbols])
            matched_exposures = factor_exposures.loc[common_symbols].values
            
            # 组合因子暴露：X_p = w' * X
            portfolio_factor_exposure = matched_weights @ matched_exposures
            
            # 因子风险贡献：X_p' * F * X_p
            factor_variance = float(portfolio_factor_exposure @ factor_covariance.values @ portfolio_factor_exposure.T)
            
            # 分解到各个因子
            factor_contributions = {}
            for i, factor_name in enumerate(factor_covariance.columns):
                # 单个因子的边际贡献
                marginal_contribution = 2 * portfolio_factor_exposure[i] * (factor_covariance.iloc[i, :].values @ portfolio_factor_exposure)
                factor_contributions[factor_name] = float(marginal_contribution)
            
            # 按因子类型聚合（假设因子命名约定）
            market_risk = sum(v for k, v in factor_contributions.items() if k.startswith('market'))
            industry_risk = sum(v for k, v in factor_contributions.items() if k.startswith('industry'))
            style_risk = sum(v for k, v in factor_contributions.items() if k.startswith('style'))
            
            # 特质风险（简化：假设为10%总风险）
            specific_risk = factor_variance * 0.1
            
            total_risk = factor_variance + specific_risk
            
            return {
                'market_risk': float(market_risk),
                'industry_risk': float(industry_risk),
                'style_risk': float(style_risk),
                'specific_risk': float(specific_risk),
                'total_risk': float(total_risk),
                'factor_contributions': factor_contributions
            }
            
        except Exception as e:
            logger.error(f"因子风险归因失败: {e}")
            return {}
    
    def analyze(self, data: Dict[str, Any], risk_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        综合分析组合风险（P1增强：7维度分析）
        
        根据专家指导，返回完整的7维度风险分析：
        1. total_risk - 组合总风险（波动率）
        2. volatility - 组合波动率（年化）
        3. var_95 - 95% VaR
        4. cvar_95 - 95% CVaR
        5. sharpe_ratio - 夏普比率
        6. max_drawdown - 最大回撤
        7. risk_contributions - 各资产风险贡献
        """
        # 初始化结果（专家推荐的7维度结构）
        result = {
            'total_risk': 0.0,              # 组合总风险（使用波动率，具有可加性）
            'volatility': 0.0,              # 组合波动率（年化）
            'var_95': 0.0,                  # 95% VaR
            'cvar_95': 0.0,                 # 95% CVaR
            'sharpe_ratio': 0.0,            # 夏普比率
            'max_drawdown': 0.0,            # 最大回撤
            'risk_contributions': {},       # 各资产风险贡献
            # 额外保留的原有字段
            'portfolio_returns': pd.Series(),
            'concentration_risk': 0.0
        }
        
        try:
            portfolio_state = data.get('portfolio_state')
            market_data = data.get('market_data')
            
            if not portfolio_state:
                logger.warning("组合状态为空")
                return result
            
            # 1. 计算组合收益序列
            portfolio_returns = pd.Series()
            if market_data:
                portfolio_returns = self.calculate_portfolio_returns(portfolio_state, market_data)
                result['portfolio_returns'] = portfolio_returns
            
            # 2. 计算波动率（总风险）
            if len(portfolio_returns) > 1:
                daily_volatility = float(portfolio_returns.std())
                annual_volatility = daily_volatility * np.sqrt(self.config.get('trading_days_per_year', 252))
                result['volatility'] = annual_volatility
                result['total_risk'] = annual_volatility  # 专家指导：总风险=波动率
            
            # 3. 计算VaR和CVaR（使用RiskMetricsService）
            if len(portfolio_returns) > 1:
                # 转换为Series类型（RiskMetricsService需要Series）
                if not isinstance(portfolio_returns, pd.Series):
                    portfolio_returns = pd.Series(portfolio_returns)
                
                var_95 = self.risk_metrics_service.calculate_value_at_risk(
                    portfolio_returns, 
                    confidence_level=0.95
                )
                cvar_95 = self.risk_metrics_service.calculate_expected_shortfall(
                    portfolio_returns, 
                    confidence_level=0.95
                )
                result['var_95'] = abs(var_95)  # VaR通常为负值，取绝对值表示损失
                result['cvar_95'] = abs(cvar_95)
            
            # 4. 计算夏普比率（使用增强版，考虑市场风险溢价）
            if len(portfolio_returns) > 1:
                # 使用增强版夏普比率（国际化支持）
                enhanced_result = self.risk_metrics_service.calculate_sharpe_ratio_enhanced(
                    portfolio_returns,
                    risk_free_rate=None,  # 使用动态无风险利率
                    include_market_premium=True,  # 包含市场溢价
                    adjust_for_anomalies=True,  # 调整市场异常
                    prices=None
                )
                result['sharpe_ratio'] = enhanced_result['enhanced_sharpe']
            
            # 5. 计算最大回撤
            if len(portfolio_returns) > 1:
                max_dd = self.risk_metrics_service.calculate_max_drawdown(portfolio_returns)
                result['max_drawdown'] = abs(max_dd)  # 取绝对值表示回撤幅度
            
            # 6. 计算风险贡献度（优先使用协方差矩阵）
            cov_matrix = data.get('covariance_matrix')
            if cov_matrix is not None:
                result['risk_contributions'] = self.calculate_risk_contributions_covariance(
                    portfolio_state, cov_matrix
                )
            else:
                corr_matrix = data.get('correlation_matrix')
                if corr_matrix is not None:
                    result['risk_contributions'] = self.calculate_risk_contributions(
                        portfolio_state, corr_matrix
                    )
            
            # 7. 计算集中度风险（HHI）
            weights_list = [alloc.weight for alloc in portfolio_state.allocations.values()]
            if weights_list:
                hhi = sum(w ** 2 for w in weights_list)
                result['concentration_risk'] = float(min(hhi, 1.0))
            
            logger.debug(f"组合风险分析完成: 波动率={result['volatility']:.4f}, VaR={result['var_95']:.4f}, 夏普={result['sharpe_ratio']:.2f}")
            return result
        
        except Exception as e:
            logger.error(f"组合风险分析失败: {e}")
            return result


