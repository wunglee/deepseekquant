"""
组合再平衡器 - 业务层
从 core_bak/portfolio_manager.py 拆分
职责: 组合再平衡、权重调整
"""

from typing import Dict
import logging

logger = logging.getLogger("DeepSeekQuant.PortfolioRebalancer")


class PortfolioRebalancer:
    """组合再平衡器"""

    def __init__(self, config: Dict):
        self.config = config

    def _market_cap_optimization(self, portfolio: PortfolioState, signals: Dict,
                                 market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """市值加权优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols:
                return {}

            # 获取市值数据
            market_caps = {}
            total_market_cap = 0

            for symbol in symbols:
                # 这里应该从市场数据获取实际市值
                # 简化实现：使用价格*假设的流通股本
                if symbol in market_data['prices']:
                    price = market_data['prices'][symbol]['close'][-1]
                    # 假设流通股本（实际中应该从基本面数据获取）
                    shares_outstanding = self._estimate_shares_outstanding(symbol)
                    market_cap = price * shares_outstanding
                    market_caps[symbol] = market_cap
                    total_market_cap += market_cap

            if total_market_cap == 0:
                return self._equal_weight_optimization(portfolio, signals, market_data, objective)

            # 计算市值权重
            optimized_weights = {}
            for symbol, market_cap in market_caps.items():
                optimized_weights[symbol] = market_cap / total_market_cap

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"市值加权优化失败: {e}")
            return self._equal_weight_optimization(portfolio, signals, market_data, objective)

    def _min_variance_optimization(self, portfolio: PortfolioState, signals: Dict,
                                   market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """最小方差优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None:
                return self._equal_weight_optimization(portfolio, signals, market_data, objective)

            # 确保协方差矩阵包含所有符号
            missing_symbols = set(symbols) - set(self.covariance_matrix.columns)
            if missing_symbols:
                logger.warning(f"协方差矩阵缺少符号: {missing_symbols}, 使用等权重优化")
                return self._equal_weight_optimization(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt进行最小方差优化
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]

            ef = EfficientFrontier(None, cov_matrix, weight_bounds=(0, 1))
            ef.min_volatility()
            weights = ef.clean_weights()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"最小方差优化失败: {e}")
            return self._equal_weight_optimization(portfolio, signals, market_data, objective)

    def _max_sharpe_optimization(self, portfolio: PortfolioState, signals: Dict,
                                 market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """最大夏普比率优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None or self.expected_returns is None:
                return self._min_variance_optimization(portfolio, signals, market_data, objective)

            # 确保数据完整性
            missing_cov = set(symbols) - set(self.covariance_matrix.columns)
            missing_returns = set(symbols) - set(self.expected_returns.index)

            if missing_cov or missing_returns:
                logger.warning(f"数据不完整, 使用最小方差优化")
                return self._min_variance_optimization(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt进行最大夏普优化
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]
            expected_returns = self.expected_returns[symbols]

            ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0, 1))
            ef.max_sharpe()
            weights = ef.clean_weights()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"最大夏普比率优化失败: {e}")
            return self._min_variance_optimization(portfolio, signals, market_data, objective)

    def _risk_parity_optimization(self, portfolio: PortfolioState, signals: Dict,
                                  market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """风险平价优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None:
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 确保协方差矩阵包含所有符号
            missing_symbols = set(symbols) - set(self.covariance_matrix.columns)
            if missing_symbols:
                logger.warning(f"协方差矩阵缺少符号: {missing_symbols}, 使用最大夏普优化")
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 使用Riskfolio-Lib进行风险平价优化
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]
            returns_data = self._get_historical_returns(symbols, market_data)

            if returns_data is None or returns_data.empty:
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 创建风险平价组合
            port = rp.Portfolio(returns=returns_data)
            port.assets_stats(method_mu='hist', method_cov='hist')
            port.rp_optimization(model='Classic', objective='Risk', rm='MV')
            weights = port.w.to_dict()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols if symbol in weights}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"风险平价优化失败: {e}")
            return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

    def _black_litterman_optimization(self, portfolio: PortfolioState, signals: Dict,
                                      market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """Black-Litterman模型优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None or self.expected_returns is None:
                return self._risk_parity_optimization(portfolio, signals, market_data, objective)

            # 获取市场均衡收益（先验收益）
            market_returns = self.expected_returns[symbols]
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]

            # 基于信号生成观点
            views, confidence = self._generate_black_litterman_views(signals, symbols, market_data)

            if not views:
                logger.info("没有生成有效的观点，使用市场均衡收益")
                return self._max_sharpe_optimization(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt的Black-Litterman模型
            from pypfopt import BlackLittermanModel

            bl = BlackLittermanModel(cov_matrix, pi=market_returns, absolute_views=views)
            bl_returns = bl.bl_returns()
            bl_cov = bl.bl_cov()

            # 使用Black-Litterman后的收益和协方差进行优化
            ef = EfficientFrontier(bl_returns, bl_cov, weight_bounds=(0, 1))

            if objective == PortfolioObjective.MAXIMIZE_SHARPE:
                ef.max_sharpe()
            elif objective == PortfolioObjective.MINIMIZE_RISK:
                ef.min_volatility()
            else:
                ef.max_sharpe()

            weights = ef.clean_weights()
            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"Black-Litterman优化失败: {e}")
            return self._risk_parity_optimization(portfolio, signals, market_data, objective)

    def _hierarchical_risk_parity(self, portfolio: PortfolioState, signals: Dict,
                                  market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """分层风险平价优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None:
                return self._black_litterman_optimization(portfolio, signals, market_data, objective)

            # 确保协方差矩阵包含所有符号
            missing_symbols = set(symbols) - set(self.covariance_matrix.columns)
            if missing_symbols:
                logger.warning(f"协方差矩阵缺少符号: {missing_symbols}, 使用Black-Litterman优化")
                return self._black_litterman_optimization(portfolio, signals, market_data, objective)

            # 使用Riskfolio-Lib进行HRP优化
            returns_data = self._get_historical_returns(symbols, market_data)

            if returns_data is None or returns_data.empty:
                return self._black_litterman_optimization(portfolio, signals, market_data, objective)

            port = rp.Portfolio(returns=returns_data)
            port.assets_stats(method_mu='hist', method_cov='hist')
            port.hrp_optimization(model='HRP')
            weights = port.w.to_dict()

            optimized_weights = {symbol: weights[symbol] for symbol in symbols if symbol in weights}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"分层风险平价优化失败: {e}")
            return self._black_litterman_optimization(portfolio, signals, market_data, objective)

    def _critical_line_algorithm(self, portfolio: PortfolioState, signals: Dict,
                                 market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """关键线算法优化"""
        try:
            symbols = list(portfolio.allocations.keys())
            if not symbols or self.covariance_matrix is None or self.expected_returns is None:
                return self._hierarchical_risk_parity(portfolio, signals, market_data, objective)

            # 使用PyPortfolioOpt的关键线算法
            cov_matrix = self.covariance_matrix.loc[symbols, symbols]
            expected_returns = self.expected_returns[symbols]

            ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0, 1))

            if objective == PortfolioObjective.MAXIMIZE_SHARPE:
                ef.max_sharpe()
            elif objective == PortfolioObjective.MINIMIZE_RISK:
                ef.min_volatility()
            elif objective == PortfolioObjective.MAXIMIZE_UTILITY:
                risk_aversion = self.optimization_config.get('risk_aversion', 1.0)
                ef.max_quadratic_utility(risk_aversion=risk_aversion)
            else:
                ef.max_sharpe()

            weights = ef.clean_weights()
            optimized_weights = {symbol: weights[symbol] for symbol in symbols}

            # 考虑现金余额
            cash_weight = portfolio.cash_balance / portfolio.total_value if portfolio.total_value > 0 else 0
            if cash_weight > 0:
                adjustment_factor = 1 - cash_weight
                optimized_weights = {symbol: weight * adjustment_factor for symbol, weight in optimized_weights.items()}
                optimized_weights['CASH'] = cash_weight

            return optimized_weights

        except Exception as e:
            logger.error(f"关键线算法优化失败: {e}")
            return self._hierarchical_risk_parity(portfolio, signals, market_data, objective)

    def _custom_optimization(self, portfolio: PortfolioState, signals: Dict,
                             market_data: Dict, objective: PortfolioObjective) -> Dict[str, float]:
        """自定义优化"""
        try:
            # 获取自定义优化配置
            custom_config = self.optimization_config.get('custom_parameters', {})
            optimization_type = custom_config.get('type', 'signal_based')
